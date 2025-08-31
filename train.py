#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Retraining script with improved data format for clean sentiment analysis responses.
This addresses the streaming issue by using a better training format.
"""

import os
import sys
import argparse
from pathlib import Path

# Check if we have the necessary dependencies
try:
    '''
    PyTorch is the deep learning framework.

    Provides:

    Tensors (GPU/CPU accelerated arrays).

    Autograd for backpropagation.

    Optimizers, neural net layers, loss functions.

    In training scripts, it underpins all model computations.
    '''
    import torch

    '''
    Datasets is a library for loading and preprocessing datasets.

    Provides:

    Data loading, preprocessing, and transformations.

    Common datasets and benchmarks.
    From 🤗 Hugging Face Datasets library.

    load_dataset() fetches pre-packaged or custom datasets (e.g., "wikitext", "imdb").

    Supports streaming large datasets and automatic caching.
    '''
    from datasets import load_dataset

    '''
    Transformers is a library for natural language processing.

    Provides:

    Pre-trained models, tokenizers, and training utilities.

    AutoModelForCausalLM loads pre-trained models.

    AutoTokenizer handles tokenization: Loads the correct tokenizer for a given model name (e.g., GPT-2, LLaMA, BERT). Handles converting raw text → token IDs for training.

    TrainingArguments Defines hyperparameters and settings for training (learning rate, batch size, logging, checkpoints, etc.).

    Trainer handles training loop: Manages the training loop, including data loading, batching, and model updates.

    DataCollatorForLanguageModeling prepares batches: Converts raw text into token IDs and handles padding.

    '''
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        DataCollatorForLanguageModeling
    )

    '''
    PEFT library by Hugging Face.

    Enables lightweight fine-tuning (LoRA, Prefix Tuning, etc.), so you don't update the full model (billions of params), only small adapter weights.

    LoraConfig: Configuration object defining LoRA rank, alpha, dropout, target modules, etc.

    get_peft_model: Wraps a base transformer model with PEFT modules (injects trainable low-rank adapters).

    TaskType: Enum defining task type (CAUSAL_LM, SEQ_CLS, TOKEN_CLS, etc.), so PEFT knows how to attach adapters.
    '''
    from peft import LoraConfig, get_peft_model, TaskType

    '''
    Evaluate: Hugging Face's Evaluate library.

    Provides standardized implementations of evaluation metrics (accuracy, perplexity, BLEU, ROUGE, etc.).
    '''
    import evaluate

    '''
    tqdm adds progress bars to loops.

    Commonly used during dataset preprocessing or evaluation loops:
    '''
    from tqdm import tqdm

    import subprocess

    DEPENDENCIES_OK = True
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("   Install with: pip install torch transformers datasets peft evaluate tqdm")
    DEPENDENCIES_OK = False

def improved_data_format():
    """
    Load and format data with improved training format for better sentiment responses.

    This function loads the tweet_eval sentiment dataset and formats it with better training
    data structure to prevent streaming responses by adding proper stop tokens and clear
    instruction-response format.

    Returns:
        tuple: (train_dataset, val_dataset) formatted for improved sentiment analysis
    """
    print("📊 Loading and formatting data with improved format...")

    # Load the dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    # Define labels
    LABEL_NAMES = ["negative", "neutral", "positive"]

    def to_chat_improved(sample):
        """Improved format that prevents streaming."""
        tweet = sample['text']
        label = LABEL_NAMES[sample["label"]]

        # Better format: Clear instruction + single response + end token
        return {
            "text": f"Classify the sentiment of this tweet: {tweet}\n\nSentiment: {label}<|endoftext|>",
            "label": sample["label"]
        }

    # Apply improved formatting
    train_raw = ds["train"].map(to_chat_improved)
    val_raw = ds["validation"].map(to_chat_improved)

    print(f"✅ Data formatted - Train: {len(train_raw)}, Val: {len(val_raw)}")
    return train_raw, val_raw

def create_better_modelfile(gguf_file: str, model_name: str, output_dir: str):
    """
    Create a better Modelfile optimized for single responses.

    This function creates an optimized Modelfile for Ollama deployment with parameters
    specifically tuned to prevent streaming and ensure single-word sentiment responses:
    - Temperature 0.0 for deterministic responses
    - num_predict 1 to generate exactly one token
    - top_k 1 for greedy decoding
    - Clear system prompt for single-word responses

    Args:
        gguf_file: Path to the GGUF model file
        model_name: Name for the model in Ollama
        output_dir: Directory to save the Modelfile

    Returns:
        str: Path to the created Modelfile
    """
    modelfile_content = f"""FROM {os.path.basename(gguf_file)}
TEMPLATE "{{{{ .Prompt }}}}"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "###"
PARAMETER temperature 0.0
PARAMETER top_p 1.0
PARAMETER top_k 1
PARAMETER num_predict 1
PARAMETER repeat_penalty 1.0
SYSTEM You are a sentiment analysis expert. When given a tweet, classify it as exactly one word: "positive", "negative", or "neutral". Give only that single word as your response.
"""

    modelfile_path = os.path.join(output_dir, "Modelfile_Improved")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"📄 Created improved Modelfile: {modelfile_path}")
    print("   Features:")
    print("   • Temperature 0.0 for deterministic responses")
    print("   • num_predict 1 to generate exactly 1 token")
    print("   • top_k 1 for greedy decoding")
    print("   • Clear system prompt for single-word responses")

    return modelfile_path

def retrain_with_better_format(train_samples=1000, val_samples=200):
    """
    Retrain the sentiment model with improved format to prevent streaming responses.

    This function implements the complete retraining pipeline with memory-optimized settings
    for CPU training, using improved data formatting and LoRA fine-tuning to address
    the streaming response issue.

    Args:
        train_samples: Number of training samples to use (default: 1000)
        val_samples: Number of validation samples to use (default: 200)

    Returns:
        bool: True if retraining completed successfully, False otherwise
    """
    if not DEPENDENCIES_OK:
        print("❌ Dependencies not satisfied. Please install required packages.")
        return False

    print("🔄 Starting retraining with improved format...")
    print(f"   Train samples: {train_samples}")
    print(f"   Val samples: {val_samples}")

    # Load improved data
    train_raw, val_raw = improved_data_format()

    # Sample the data (ensure minimum sizes for training stability)
    min_train_samples = 100  # Minimum for meaningful training
    min_val_samples = 20     # Minimum for validation

    if train_samples:
        train_samples = max(min_train_samples, train_samples)
        train_raw = train_raw.select(range(min(train_samples, len(train_raw))))
    if val_samples:
        val_samples = max(min_val_samples, val_samples)
        val_raw = val_raw.select(range(min(val_samples, len(val_raw))))

    print(f"✅ Final dataset sizes: {len(train_raw)} train, {len(val_raw)} validation")

    # Load tokenizer
    model_name = "google/gemma-3-270m-it"
    print(f"📝 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize datasets
    print("🔄 Tokenizing datasets...")
    train_tokenized = train_raw.map(
        lambda x: tokenize_function(x, tokenizer, 512),
        batched=True,
        remove_columns=train_raw.column_names
    )
    val_tokenized = val_raw.map(
        lambda x: tokenize_function(x, tokenizer, 512),
        batched=True,
        remove_columns=val_raw.column_names
    )

    # Load model with LoRA (faster training than full)
    print("🔥 Loading model with LoRA...")
    model = load_model_with_lora(model_name, "cpu", 16, 16, 0.0)

    # Training arguments optimized for CPU memory constraints
    training_args = TrainingArguments(
        output_dir="sentiment_improved_model",
        num_train_epochs=2,  # Reduced from 3 to 2
        per_device_train_batch_size=1,  # Very small batch size
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Much higher accumulation
        learning_rate=2e-4,  # Slightly lower LR
        lr_scheduler_type="cosine",
        warmup_steps=25,  # Reduced warmup
        logging_steps=25,  # More frequent logging
        save_steps=100,  # More frequent saving
        fp16=False,  # CPU training
        seed=42,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=None,
        dataloader_num_workers=0,
        weight_decay=0.01,
        adam_beta2=0.999,
        max_grad_norm=1.0,
        # Memory optimization settings
        dataloader_drop_last=True,  # Drop incomplete batches
        # group_by_length=True,  # Disabled due to batch size issues
        # length_column_name="input_ids",  # For group_by_length
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train with memory management
    print("🚀 Starting improved training...")
    print("💡 Memory-optimized settings:")
    print("   • Batch size: 1")
    print("   • Gradient accumulation: 16")
    print("   • Group by length: enabled")
    print("   • This will take longer but use less memory")

    try:
        trainer.train()
    except RuntimeError as e:
        if "not enough memory" in str(e):
            print("❌ Out of memory! Try:")
            print("   • Reduce train_samples further")
            print("   • Use even smaller batch size")
            print("   • Close other applications")
            return False
        else:
            raise e

    # Memory cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Evaluate
    print("📊 Evaluating improved model...")
    accuracy = evaluate_model(trainer.model, tokenizer, val_raw, "cpu")
    print(f"Validation Accuracy: {accuracy:.4f}")
    # Save model - merge LoRA weights first
    print("💾 Saving improved model...")
    print("   Merging LoRA adapters with base model...")

    # Merge LoRA weights with base model
    merged_model = trainer.model.merge_and_unload()
    merged_dir = os.path.join("sentiment_improved_model", "merged")
    Path(merged_dir).mkdir(parents=True, exist_ok=True)

    # Save merged model
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)

    print("   ✅ Merged model saved to:", merged_dir)

    # Convert to GGUF
    print("🔄 Converting to GGUF...")
    gguf_success = convert_to_gguf(merged_dir, "sentiment_improved_model")

    if gguf_success:
        gguf_dir = os.path.join("sentiment_improved_model", "gguf_converted")
        print("\n🎉 Improved model ready!")
        print(f"   GGUF Location: {gguf_dir}")
        print("   Should give clean, single-word responses!")

        # Create better Modelfile
        create_better_modelfile(
            os.path.join(gguf_dir, "sentiment_improved_model.gguf"),
            "sentiment_improved",
            gguf_dir
        )

        return True
    else:
        print("❌ GGUF conversion failed!")
        print("   Model saved in Hugging Face format only.")
        print(f"   Location: {merged_dir}")
        return False

# Import functions from the main scripts
def tokenize_function(examples, tokenizer, max_length):
    """
    Tokenize input text and prepare labels for language modeling training.

    This function handles the tokenization process, converting raw text into token IDs
    with proper padding and truncation, and creates labels for supervised learning.

    Args:
        examples: Dictionary containing text data from dataset
        tokenizer: Pre-trained tokenizer for the model
        max_length: Maximum sequence length for tokenization

    Returns:
        dict: Tokenized inputs with input_ids, attention_mask, and labels
    """
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    result["labels"] = result["input_ids"].clone()
    return result

def load_model_with_lora(model_name: str, device: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    """
    Load a pre-trained model with LoRA adapters for efficient fine-tuning.

    This function loads a base transformer model and applies LoRA (Low-Rank Adaptation)
    for parameter-efficient fine-tuning. Only the LoRA adapter weights are trainable,
    significantly reducing memory requirements and training time.

    Args:
        model_name: Name or path of the pre-trained model
        device: Device to load the model on ("cpu" or "cuda")
        lora_r: LoRA rank (dimension of the low-rank matrices)
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout probability for LoRA layers

    Returns:
        PeftModel: Model with LoRA adapters applied
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"
    )

    if device == "cpu":
        model = model.to(device)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model

def evaluate_model(model, tokenizer, val_raw, device):
    """
    Evaluate the fine-tuned model on validation data.

    This function evaluates the model's performance on sentiment classification by generating
    predictions for validation samples and computing accuracy metrics.

    Args:
        model: Fine-tuned model to evaluate
        tokenizer: Tokenizer for text processing
        val_raw: Validation dataset
        device: Device for model inference

    Returns:
        float: Accuracy score on validation set
    """
    model.eval()
    acc_metric = evaluate.load("accuracy")

    y_true, y_pred = [], []

    with torch.no_grad():
        for example in tqdm(val_raw, desc="Evaluating"):
            prompt = f"Classify this tweet sentiment: {example['text']}\nSentiment:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=1.0,
                    top_k=50,
                    temperature=1.0,
                    use_cache=True
                )

            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

            response_lower = response.strip().lower()
            if "negative" in response_lower or response_lower.startswith("neg"):
                pred = 0
            elif "positive" in response_lower or response_lower.startswith("pos"):
                pred = 2
            else:
                pred = 1

            y_pred.append(pred)
            y_true.append(example["label"])

    accuracy = acc_metric.compute(predictions=y_pred, references=y_true)["accuracy"]
    return accuracy

def convert_to_gguf(merged_dir: str, output_dir: str):
    """
    Convert merged model to GGUF format for Ollama compatibility.

    This function converts a Hugging Face model to GGUF format using either the standalone
    conversion script or the llama.cpp conversion tools. GGUF format is optimized for
    CPU inference and compatible with Ollama and LM Studio.

    Args:
        merged_dir: Path to directory containing merged model
        output_dir: Output directory for GGUF file

    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        # Use the standalone script if it exists
        script_path = "convert_to_gguf_standalone.py"
        if os.path.exists(script_path):
            result = subprocess.run([
                sys.executable, script_path,
                "--model_path", merged_dir,
                "--output_dir", output_dir,
                "--quantization", "f16"
            ], capture_output=True, text=True)

            return result.returncode == 0

        # Fallback: Try to use llama.cpp directly
        script_path = os.path.join("llama.cpp", "convert_hf_to_gguf.py")
        if os.path.exists(script_path):
            gguf_file = os.path.join(output_dir, "gguf", f"{os.path.basename(merged_dir)}.gguf")

            result = subprocess.run([
                sys.executable, script_path,
                merged_dir,
                "--outfile", gguf_file,
                "--outtype", "f16"
            ], capture_output=True, text=True)

            return result.returncode == 0

        print("❌ No conversion script found")
        return False

    except Exception as e:
        print(f"❌ GGUF conversion failed: {e}")
        return False

def main():
    """
    Main entry point for the improved sentiment retraining script.

    This function parses command-line arguments and orchestrates the complete retraining
    pipeline including data preparation, model training, evaluation, and GGUF conversion
    to create a model that gives clean single-word sentiment responses.
    """
    parser = argparse.ArgumentParser(description="Retraining with improved sentiment format")
    parser.add_argument("--train_samples", type=int, default=300, help="Number of training samples (default: 300 for memory safety)")
    parser.add_argument("--val_samples", type=int, default=50, help="Number of validation samples (default: 50 for memory safety)")

    args = parser.parse_args()

    print("🚀 Retraining Sentiment Model with Improved Format")
    print("=" * 60)
    print("This will fix the streaming response issue by:")
    print("• Using better training data format")
    print("• Adding <|endoftext|> stop tokens")
    print("• Training for single-word responses")
    print("• Memory-optimized for CPU training")
    print("=" * 60)
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print("Batch size: 1, Gradient accumulation: 16")
    print("=" * 60)

    success = retrain_with_better_format(args.train_samples, args.val_samples)

    if success:
        print("\n🎉 Retraining completed successfully!")
        print("   Your new model should give clean, single-word responses.")
        print("   Load the GGUF file in LM Studio/Ollama for testing.")
    else:
        print("\n❌ Retraining failed. Check error messages above.")

if __name__ == "__main__":
    main()
