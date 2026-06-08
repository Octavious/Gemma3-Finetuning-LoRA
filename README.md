<a href="https://www.buymeacoffee.com/ArabicAI" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

# 🚀 Gemma 3 Sentiment Retraining - Improved Format Guide

## 📋 Table of Contents
- [Solution Overview](#solution-overview-)
- [Environment Setup](#environment-setup-)
- [Installation & Setup](#installation--setup-)
- [Usage Guide](#usage-guide-)
- [Technical Deep Dive](#technical-deep-dive-)
- [Expected Results](#expected-results-)
- [Troubleshooting](#troubleshooting-)
- [Files Overview](#files-overview-)

## Solution Overview ✅

This project provides a **specialized retraining solution** for Gemma 3 models focused on **sentiment analysis** with improved single-word responses:

- **Memory-optimized CPU training** (batch size 1, gradient accumulation 16)
- **LoRA fine-tuning** for efficient parameter updates
- **Improved data formatting** to prevent streaming responses
- **Automatic GGUF conversion** for Ollama/LM Studio compatibility
- **Deterministic single-word outputs** (positive/negative/neutral)
- **Cross-platform compatibility** (CPU, GPU, Apple MPS)

## Environment Setup 🏗️

### Prerequisites

**Required Software:**
- **Python**: 3.8 - 3.11 (3.10 recommended)
- **Git**: For cloning repositories and model downloads
- **pip**: Python package manager (latest version)

**Operating System Support:**
- ✅ **Windows 10/11** (recommended)
- ✅ **macOS** (Intel/Apple Silicon)
- ✅ **Linux** (Ubuntu 18.04+, CentOS 7+)
- ✅ **WSL2** (Windows Subsystem for Linux)

### Hardware Requirements

**Minimum Requirements:**
- **RAM**: 8GB (16GB recommended for smooth training)
- **Storage**: 2GB free space (models can be 500MB-2GB each)
- **CPU**: Any modern CPU (4+ cores recommended)
- **Network**: Stable internet for model downloads

**Recommended Hardware:**
- **RAM**: 16GB+ for optimal performance
- **Storage**: SSD with 10GB+ free space
- **CPU**: 8+ cores, modern architecture
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, speeds up training)

### Virtual Environment Setup

**Why Virtual Environment?**
Isolates dependencies and prevents conflicts with system packages.

**Option 1: venv (Built-in Python)**
```bash
# Create virtual environment
python -m venv gemma_env

# Activate environment
# Windows:
gemma_env\Scripts\activate

# Linux/macOS:
source gemma_env/bin/activate

# Verify activation
python --version
pip --version
```

**Option 2: conda/miniconda**
```bash
# Create conda environment
conda create -n gemma_env python=3.10

# Activate environment
conda activate gemma_env

# Verify activation
python --version
conda list
```

### Environment Variables (Optional)

**Set these for better performance and debugging:**

**Windows:**
```cmd
# Set environment variables
set HF_HOME=%USERPROFILE%\.cache\huggingface
set TRANSFORMERS_CACHE=%USERPROFILE%\.cache\huggingface\transformers
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set CUDA_VISIBLE_DEVICES=0
```

**Linux/macOS:**
```bash
# Add to your shell profile (~/.bashrc or ~/.zshrc)
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_CACHE="$HOME/.cache/huggingface/transformers"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_VISIBLE_DEVICES=0
```

**Common Environment Variables:**
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers model cache
- `PYTORCH_CUDA_ALLOC_CONF`: GPU memory optimization
- `CUDA_VISIBLE_DEVICES`: Specific GPU selection

### System Dependencies

**Windows:**
```cmd
# Install build tools (if needed for some packages)
# Download and install: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Optional: Install Git for Windows
# Download from: https://git-scm.com/download/win
```

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update

# Install essential build tools
sudo apt install -y build-essential git curl

# Install Python development headers (if needed)
sudo apt install -y python3-dev python3-pip
```

**macOS:**
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (optional, but recommended)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install essential tools
brew install git python
```

### Testing Environment Setup

**Run these commands to verify everything is working:**

```bash
# 1. Check Python version
python --version

# 2. Check pip version
pip --version

# 3. Test basic imports
python -c "import sys; print(f'Python path: {sys.executable}')"

# 4. Test PyTorch installation (after installing requirements.txt)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 5. Test Transformers installation
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Troubleshooting Environment Issues

**Common Issues:**

**"Module not found" errors:**
```bash
# Reinstall in virtual environment
pip uninstall package_name
pip install package_name
```

**Permission errors:**
```bash
# Use --user flag or run as administrator
pip install --user package_name
```

**CUDA/GPU issues:**
```bash
# Check GPU status
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Install CPU-only version if GPU issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Training Architecture 🔧

### Main Retraining Pipeline (`retrain_with_better_format()`)

**Complete workflow**:
```python
def retrain_with_better_format():
    # 1. Load improved sentiment dataset
    train_raw, val_raw = improved_data_format()

    # 2. Sample data for memory efficiency
    train_raw = train_raw.select(range(train_samples))
    val_raw = val_raw.select(range(val_samples))

    # 3. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
    model = load_model_with_lora("google/gemma-3-270m-it", "cpu", 16, 16, 0.0)

    # 4. Memory-optimized training
    trainer = Trainer(model, training_args, train_dataset, eval_dataset)

    # 5. Evaluate and save
    accuracy = evaluate_model(trainer.model, tokenizer, val_raw)
    merged_model = trainer.model.merge_and_unload()

    # 6. Convert to GGUF for Ollama
    convert_to_gguf(merged_dir, output_dir)
    create_better_modelfile(gguf_file, model_name, output_dir)
```

### Command-Line Interface

**Simple argument parsing** for memory-optimized retraining:

**Available Parameters**:
```bash
--train_samples 300    # Number of training samples (default: 300, memory-safe)
--val_samples 50       # Number of validation samples (default: 50, memory-safe)
```

**Usage Examples**:
```bash
# Quick test with minimal data
python train.py --train_samples 100 --val_samples 20

# Balanced training (recommended)
python train.py --train_samples 500 --val_samples 100

# Full dataset (requires more memory/time)
python train.py --train_samples 2000 --val_samples 300
```

**Why Simple Arguments?**
- **Memory Safety**: Pre-configured for CPU training with optimal batch size (1) and gradient accumulation (16)
- **Reliability**: Fixed hyperparameters that work well for sentiment retraining
- **Simplicity**: Focus on data size rather than complex tuning

### 3. **Device Detection & Reproducibility (`get_device()`, `set_seed()`)**
**Why**: 
- **Hardware Optimization**: Ensures training runs on available hardware
- **Reproducibility**: Guarantees consistent results across runs

**What it does**:
- **Device Selection**: Prioritizes CUDA → MPS → CPU
- **Seed Setting**: Fixes randomness in PyTorch, NumPy, and Python
- **Hardware Detection**: Automatically detects GPU capabilities

**Device Priority**:
```python
if torch.cuda.is_available():
    return "cuda"                    # NVIDIA GPU
elif torch.backends.mps.is_available():
    return "mps"                     # Apple Silicon
else:
    return "cpu"                     # CPU fallback
```

### Improved Data Formatting (`improved_data_format()`)

**Why**: Prevents streaming responses through better training data structure.

**What it does**:
- **Dataset Loading**: Loads TweetEval sentiment dataset (cardiffnlp/tweet_eval)
- **Enhanced Formatting**: Uses `<|endoftext|>` stop tokens to prevent continuation
- **Clear Instructions**: Explicit sentiment classification prompts
- **Memory Sampling**: Configurable data subset for CPU training

**Improved Data Transformation**:
```python
# Before: Basic format (causes streaming)
"Classify this tweet sentiment: Great movie!\nSentiment: positive"

# After: Enhanced format (prevents streaming)
"Classify the sentiment of this tweet: Great movie!\n\nSentiment: positive<|endoftext|>"
```

**Key Improvements**:
- **Stop Token**: `<|endoftext|>` prevents the model from continuing after "positive"
- **Clear Separation**: Double newline (`\n\n`) separates instruction from response
- **Single Response**: Training format encourages one-word answers
- **Robust Parsing**: Handles partial matches ("pos" → "positive")

**Dataset Details**:
- **Source**: TweetEval sentiment analysis benchmark
- **Total Samples**: ~14,000 tweets
- **Classes**: negative (0), neutral (1), positive (2)
- **Balanced Distribution**: ~33% each class
- **Average Length**: ~50 tokens per tweet

### Tokenization Process

**Standard Gemma-3 tokenization** with memory-optimized settings:

**What it does**:
- **Tokenizer Loading**: AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
- **Pad Token Handling**: Sets pad_token = eos_token if None
- **Batch Tokenization**: Processes datasets efficiently
- **Sequence Handling**: Max length 512 tokens with truncation

**Tokenization Implementation**:
```python
def tokenize_function(examples, tokenizer, max_length):
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,  # 512 tokens
        padding="max_length",   # Pad to max length
        return_tensors="pt"
    )
    result["labels"] = result["input_ids"].clone()  # Causal LM training
    return result
```

**Technical Specifications**:
- **Model**: google/gemma-3-270m-it
- **Vocabulary**: 256K+ tokens
- **Max Length**: 512 tokens (fits most tweets)
- **Padding**: Right padding to max length
- **Labels**: Same as input_ids for next-token prediction

### LoRA Model Loading

**Memory-optimized LoRA configuration** for CPU training:

**What it does**:
- **Base Model**: Loads google/gemma-3-270m-it (270M parameters)
- **LoRA Adapters**: Adds trainable low-rank matrices to attention + MLP layers
- **Memory Efficiency**: Only ~1% of parameters are trainable
- **CPU Optimized**: Uses float32 precision, eager attention

**LoRA Configuration**:
```python
def load_model_with_lora(model_name, device, lora_r=16, lora_alpha=16, lora_dropout=0.0):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # CPU compatible
        trust_remote_code=True,
        attn_implementation="eager"  # CPU optimized
    )

    lora_config = LoraConfig(
        r=16,                    # Rank: 16 (balanced capacity)
        lora_alpha=16,           # Alpha: 16 (1:1 ratio with rank)
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj"      # MLP layers
        ],
        lora_dropout=0.0,        # No dropout for small dataset
        bias="none",             # Freeze all biases
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    return model
```

**Efficiency Metrics**:
- **Base Model**: 270M parameters (frozen)
- **Trainable**: ~2M parameters (LoRA adapters)
- **Memory Usage**: ~4-8GB RAM during training
- **Training Time**: 2-30 minutes depending on data size

### Memory-Optimized Training

**CPU-focused training configuration** with gradient accumulation:

**Training Arguments**:
```python
training_args = TrainingArguments(
    output_dir="sentiment_improved_model",
    num_train_epochs=2,              # 2 epochs for sentiment task
    per_device_train_batch_size=1,   # Small batch size for CPU
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=16,  # Effective batch size = 1 * 16 = 16
    learning_rate=2e-4,              # Standard for LoRA fine-tuning
    lr_scheduler_type="cosine",      # Smooth learning rate decay
    warmup_steps=25,                 # Short warmup period
    logging_steps=25,                # Frequent logging
    save_steps=100,                  # Save checkpoints regularly
    fp16=False,                      # CPU training (no mixed precision)
    seed=42,                         # Reproducibility
    dataloader_pin_memory=False,     # CPU optimization
    remove_unused_columns=False,
    report_to=None,                  # Disable external logging
    dataloader_num_workers=0,        # Single-threaded for stability
    weight_decay=0.01,               # L2 regularization
    adam_beta2=0.999,
    max_grad_norm=1.0,               # Gradient clipping
)
```

**Memory Optimization Features**:
- **Batch Size**: 1 sample per step (CPU compatible)
- **Gradient Accumulation**: 16 steps = effective batch size 16
- **No Mixed Precision**: FP32 for CPU stability
- **Single Threading**: `num_workers=0` for reliability
- **Memory Cleanup**: Garbage collection after training

### Model Evaluation

**Robust sentiment evaluation** with flexible parsing:

**Evaluation Process**:
```python
def evaluate_model(model, tokenizer, val_raw, device):
    model.eval()
    acc_metric = evaluate.load("accuracy")

    with torch.no_grad():
        for example in val_raw:
            # Create evaluation prompt
            prompt = f"Classify this tweet sentiment: {example['text']}\nSentiment:"

            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=5,        # Short response expected
    do_sample=False,         # Greedy decoding
                temperature=1.0,         # Deterministic
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                top_p=1.0, top_k=50,    # Focused generation
                use_cache=True
            )

            # Parse response with flexible matching
            response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:])
            response_lower = response.strip().lower()

            # Robust sentiment parsing
            if "negative" in response_lower or response_lower.startswith("neg"):
                pred = 0
            elif "positive" in response_lower or response_lower.startswith("pos"):
                pred = 2
            else:
                pred = 1  # neutral/default

    return accuracy_score
```

**Evaluation Features**:
- **Flexible Parsing**: Handles partial matches ("pos" → "positive")
- **Greedy Decoding**: Deterministic predictions for consistency
- **Short Responses**: `max_new_tokens=5` for single-word answers
- **Robust Defaults**: Falls back to neutral for unclear responses

### Model Saving & GGUF Conversion

**Complete model persistence** pipeline with Ollama deployment:

**Saving Process**:
```python
# 1. Merge LoRA weights with base model
merged_model = trainer.model.merge_and_unload()

# 2. Save merged model
merged_dir = os.path.join("sentiment_improved_model", "merged")
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

# 3. Convert to GGUF format
convert_to_gguf(merged_dir, "sentiment_improved_model")

# 4. Create optimized Modelfile
create_better_modelfile(gguf_file, "sentiment_improved", gguf_dir)
```

**Output Structure**:
```
sentiment_improved_model/
├── merged/                     # Hugging Face format
│   ├── config.json
│   ├── model.safetensors       # ~500MB merged model
│   └── tokenizer files...
└── gguf_converted/            # Ollama ready
    ├── merged.gguf            # Quantized model (~500MB)
    └── Modelfile              # Configuration file
```

**GGUF Conversion Features**:
- **Automatic Detection**: Uses standalone script or llama.cpp
- **F16 Base**: Converts to float16 first, then quantizes
- **Multiple Formats**: Supports q4_k_m, q5_k_m, etc.
- **Ollama Ready**: Creates Modelfile for immediate deployment

## Installation & Setup 🛠️

### 1. **Install Dependencies** (1 minute)
```bash
pip install -r requirements.txt
```

**Key Dependencies**:
- `transformers>=4.40.0`: Hugging Face model library
- `torch>=2.0.0`: PyTorch deep learning framework
- `datasets>=2.14.0`: Dataset loading and processing
- `peft>=0.7.0`: Parameter-efficient fine-tuning
- `evaluate>=0.4.0`: Evaluation metrics
- `accelerate>=0.20.0`: Distributed training support

### 2. **HuggingFace Authentication** (30 seconds)
```bash
hf auth login --token [insert HF token]
```

**Why Required**: Gemma models require authentication for download.

### 3. **Hardware Verification**
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check MPS availability (Apple Silicon)
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```


## Usage Guide 📖

### Quick Start Commands

**Basic retraining** (recommended for most users):
```bash
# Install dependencies
pip install -r requirements.txt

# Quick test (5-10 minutes)
python train.py --train_samples 100 --val_samples 20

# Balanced training (15-30 minutes)
python train.py --train_samples 500 --val_samples 100

# Full training (30-60 minutes)
python train.py --train_samples 2000 --val_samples 300
```

### Expected Output

After training completes, you'll get:
```
sentiment_improved_model/
├── merged/                     # Hugging Face format
│   ├── config.json
│   ├── model.safetensors       # ~500MB
│   └── tokenizer files...
└── gguf_converted/            # Ollama ready
    ├── merged.gguf            # ~500MB quantized model
    └── Modelfile              # Configuration file
```

### Ollama Deployment

**Deploy your trained model**:

```bash
# Navigate to GGUF directory
cd sentiment_improved_model/gguf_converted

# Create Ollama model
ollama create sentiment-improved -f Modelfile

# Test the model
ollama run sentiment-improved

# Interactive testing
echo "Classify this tweet sentiment: I love this movie!" | ollama run sentiment-improved
```

### Standalone GGUF Conversion

**Convert existing models** to GGUF format:

```bash
# Convert any Hugging Face model
python convert_to_gguf_standalone.py \
    --model_path /path/to/model \
    --output_dir ./converted_model \
    --quantization q4_k_m

# Available quantization options
# q4_k_m (recommended), q5_k_m, q8_0, f16, etc.
```

## Technical Deep Dive 🔬

### LoRA (Low-Rank Adaptation) Details

**Mathematical Foundation**:
LoRA approximates weight updates using low-rank decomposition:
```
ΔW = BA
```
Where:
- `W` = Original weight matrix (d × k)
- `B` = Low-rank matrix (d × r), where r << min(d,k)
- `A` = Low-rank matrix (r × k)
- `ΔW` = Weight update (d × k)

**Benefits**:
- **Memory Efficiency**: Reduces trainable parameters by 90%+
- **Training Speed**: ~10x faster than full fine-tuning
- **Storage Efficiency**: Adapters are ~1% of base model size
- **Modularity**: Can be easily swapped or combined

**Hyperparameter Tuning**:
- **Rank (r)**: Higher = more capacity, more parameters
  - `r=8`: Lightweight, good for simple tasks
  - `r=16`: Balanced (default)
  - `r=32`: High capacity, good for complex tasks
- **Alpha (α)**: Scaling factor, typically α = r
- **Dropout**: Regularization, typically 0.0-0.1

### Model Architecture Details

**Gemma-3-270M Specifications**:
- **Parameters**: 270 million
- **Layers**: 28 transformer layers
- **Attention Heads**: 16 heads
- **Hidden Size**: 1,536 dimensions
- **Vocabulary**: 256K tokens
- **Context Length**: 8,192 tokens

**Training Configuration**:
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: 2e-4 (typical for LoRA)
- **Scheduler**: Cosine annealing with warmup
- **Batch Size**: Effective batch size = device_batch × accumulation
- **Mixed Precision**: FP16 for GPU, FP32 for CPU

### Data Processing Pipeline

**Dataset Statistics**:
- **Source**: TweetEval sentiment analysis
- **Total Examples**: ~14,000 tweets
- **Class Distribution**: ~33% each (balanced)
- **Average Length**: ~50 tokens
- **Max Length**: 512 tokens (truncated)

**Preprocessing Steps**:
1. **Text Cleaning**: Remove URLs, mentions, hashtags
2. **Format Conversion**: Convert to instruction format
3. **Tokenization**: Convert to token IDs
4. **Padding**: Pad sequences to max length
5. **Label Creation**: Set labels for causal LM training

## Expected Results 📊

### Performance Benchmarks

**Training Performance** (CPU with memory optimization):
- **Memory Usage**: 4-8GB RAM (batch size 1 + gradient accumulation)
- **Training Time**: 5-60 minutes depending on data size
- **CPU Cores**: 4-8 cores utilized
- **Disk Space**: ~600MB for final model

**Model Performance** (sentiment classification):
- **Baseline**: 33.3% (random guessing for 3 classes)
- **Quick Test**: 50-65% (100 samples, 5-10 minutes)
- **Balanced**: 65-75% (500 samples, 15-30 minutes)
- **High Quality**: 75-85% (2000+ samples, 45-60 minutes)

**File Sizes**:
- **Merged Model**: ~500MB (Hugging Face format)
- **GGUF Model**: ~200-400MB (quantized for Ollama)
- **Training Data**: Minimal (TweetEval dataset downloaded automatically)

### Performance Scaling

**Sample Size Impact**:
```
Samples | Accuracy | Training Time | Memory Usage
--------|----------|---------------|-------------
100     | 50-65%  | 5-10 min     | ~4GB RAM
300     | 60-70%  | 10-20 min    | ~5GB RAM
500     | 65-75%  | 15-30 min    | ~6GB RAM
1000    | 70-80%  | 25-45 min    | ~7GB RAM
2000    | 75-85%  | 45-60 min    | ~8GB RAM
```

**Key Factors**:
- **Data Quality**: TweetEval provides clean, balanced sentiment data
- **Convergence**: Model typically converges within 2 epochs
- **Memory**: Gradient accumulation enables effective training on limited RAM
- **Determinism**: Fixed seeds ensure reproducible results

## Troubleshooting 🔧

### Common Issues and Solutions

**1. Memory Errors (Out of Memory)**
```bash
# The training is already optimized for CPU with batch_size=1
# If you still get OOM errors, reduce data size further:
python train.py --train_samples 50 --val_samples 10

# Or run garbage collection manually:
import gc; gc.collect()
```

**2. Slow Training**
```bash
# Training is CPU-optimized, but if too slow:
# Use smaller dataset for testing
python train.py --train_samples 100 --val_samples 20

# Or check CPU utilization:
# The script uses single-threaded processing for stability
```

**3. Poor Model Performance**
```bash
# Increase training data size:
python train.py --train_samples 1000 --val_samples 200

# Note: Model typically converges well with 500+ samples
# The improved data format prevents streaming responses
```

**4. Authentication Errors**
```bash
# Required for Gemma model download:
hf auth login --token YOUR_HUGGINGFACE_TOKEN

# Get token from: https://huggingface.co/settings/tokens
```

**5. GGUF Conversion Fails**
```bash
# If llama.cpp conversion fails, try the standalone script:
python convert_to_gguf_standalone.py \
    --model_path sentiment_improved_model/merged \
    --output_dir sentiment_improved_model \
    --quantization f16
```

**6. Import Errors**
```bash
# Install missing dependencies:
pip install torch transformers datasets peft evaluate tqdm

# For GGUF conversion:
pip install llama-cpp-python
```

**7. Training Hangs or Freezes**
```bash
# The script has memory cleanup built-in
# If it hangs, try smaller batch or restart
python train.py --train_samples 50 --val_samples 10
```

## Advanced Configuration ⚙️

### Custom Datasets

**Format Requirements**:
```python
# Expected dataset format
{
    "text": "Your input text here",
    "label": 0  # Integer label (0, 1, 2 for sentiment)
}
```

**Loading Custom Data**:
```python
# Modify load_and_prepare_data() function
def load_custom_data():
    # Load your dataset
    dataset = load_dataset("your_dataset")
    
    # Apply formatting
    formatted = dataset.map(your_formatting_function)
    
    return formatted["train"], formatted["validation"]
```

### Hyperparameter Tuning

**Learning Rate Schedule**:
```python
# Available schedulers
lr_scheduler_type="cosine"      # Smooth decay (default)
lr_scheduler_type="linear"      # Linear decay
lr_scheduler_type="constant"    # Fixed learning rate
```

**LoRA Configuration**:
```python
# High capacity (more parameters)
lora_r=32, lora_alpha=32

# Lightweight (fewer parameters)
lora_r=8, lora_alpha=8

# Balanced (default)
lora_r=16, lora_alpha=16
```

### Model Variants

**Available Models**:
```bash
# Small model (270M parameters)
--model_name google/gemma-3-270m-it

# Medium model (2B parameters)
--model_name google/gemma-3-2b-it

# Large model (7B parameters) - requires more memory
--model_name google/gemma-3-7b-it
```

## Files Overview 📁

### Core Files

- **`train.py`** ← **MAIN SCRIPT** (sentiment retraining)
  - Complete sentiment analysis retraining pipeline
  - Memory-optimized for CPU training
  - Improved data formatting to prevent streaming responses
  - Automatic GGUF conversion with optimized Modelfile
  - Production-ready with comprehensive error handling

- **`convert_to_gguf_standalone.py`** (GGUF conversion utility)
  - Standalone script to convert any Hugging Face model to GGUF
  - Automatic llama.cpp repository management
  - Multiple quantization options (q4_k_m, q5_k_m, f16, etc.)
  - Ollama-ready Modelfile generation

- **`requirements.txt`** (dependencies)
  - All required packages for training and conversion
  - Includes torch, transformers, datasets, peft, evaluate
  - Optional GGUF conversion support

### Project Structure
```
Gemma3_Finetuning_Lora/
├── train.py                      # Main retraining script
├── convert_to_gguf_standalone.py # GGUF conversion utility
├── requirements.txt              # Python dependencies
└── README.md                     # This documentation
```

### Output Structure
```
sentiment_improved_model/
├── merged/                       # Hugging Face format
│   ├── config.json              # Model configuration
│   ├── model.safetensors        # Merged weights (~500MB)
│   └── tokenizer files...       # Tokenizer data
└── gguf/                         # Ollama deployment ready
    ├── sentiment_improved_model.gguf  # Quantized model
    └── Modelfile_Improved       # Optimized configuration
```

### Model Deployment Options

**Option 1: Ollama (Recommended)**
```bash
cd sentiment_improved_model/gguf
ollama create sentiment-improved -f Modelfile_Improved
ollama run sentiment-improved
```

**Option 2: Hugging Face Transformers**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load merged model
model = AutoModelForCausalLM.from_pretrained("sentiment_improved_model/merged")
tokenizer = AutoTokenizer.from_pretrained("sentiment_improved_model/merged")

# Use for inference
prompt = "Classify this tweet sentiment: I love this movie!\nSentiment:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=5)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## You're Ready! 🎉

This **sentiment retraining solution** is specifically designed to fix streaming response issues and provide clean single-word sentiment classifications:

✅ **Streaming Response Fix**: Improved data format prevents repetitive outputs
✅ **CPU-Optimized**: Memory-efficient training with gradient accumulation
✅ **Single-Word Responses**: Deterministic outputs (positive/negative/neutral)
✅ **Ollama Ready**: Automatic GGUF conversion with optimized Modelfile
✅ **Production Ready**: Comprehensive error handling and validation
✅ **Simple Interface**: Just specify training sample size

**Quick Start**:
```bash
# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace
hf auth login

# Run retraining
python train.py --train_samples 500 --val_samples 100

# Deploy to Ollama
cd sentiment_improved_model/gguf
ollama create sentiment-improved -f Modelfile_Improved
ollama run sentiment-improved
```

**What You'll Get**:
- **Clean Responses**: No more "positive positive positive..."
- **Fast Training**: 15-30 minutes for good performance
- **Easy Deployment**: Ready-to-use Ollama model
- **Memory Efficient**: Works on standard CPUs with 8GB RAM

**Support**: Check the troubleshooting section for common issues. The improved format ensures reliable single-word sentiment classifications!
