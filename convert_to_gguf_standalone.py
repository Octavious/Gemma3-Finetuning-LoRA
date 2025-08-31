#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to convert a Hugging Face model to GGUF format.
Uses the official convert-hf-to-gguf.py script from llama.cpp repository.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def get_conversion_script():
    """Get the llama.cpp repository and conversion script."""
    repo_dir = Path("llama.cpp")
    
    # Check if repository already exists and find the conversion script
    if repo_dir.exists():
        print(f"📁 Repository already exists: {repo_dir}")
        # Try different possible script names
        possible_scripts = ["convert.py", "convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]
        
        for script_name in possible_scripts:
            script_path = repo_dir / script_name
            if script_path.exists():
                print(f"✅ Found conversion script: {script_path}")
                return str(script_path)
        
        print("❌ No conversion script found in existing repository")
        print("   Removing existing directory and re-cloning...")
        
        # Remove existing directory and re-clone
        try:
            import shutil
            shutil.rmtree(repo_dir)
        except Exception as e:
            print(f"   Failed to remove existing directory: {e}")
            return None
    
    # Clone the repository
    print("📦 Cloning llama.cpp repository...")
    try:
        result = subprocess.run([
            "git", "clone", "--depth", "1", 
            "https://github.com/ggerganov/llama.cpp.git", 
            "llama.cpp"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            # Try different possible script names
            possible_scripts = ["convert.py", "convert-hf-to-gguf.py", "convert_hf_to_gguf.py"]
            
            for script_name in possible_scripts:
                script_path = repo_dir / script_name
                if script_path.exists():
                    print(f"✅ Found conversion script: {script_path}")
                    return str(script_path)
            
            print("❌ No conversion script found in cloned repository")
            print("   Available files in repository:")
            try:
                for file in repo_dir.iterdir():
                    if file.name.endswith('.py'):
                        print(f"     {file.name}")
            except:
                pass
            return None
        else:
            print(f"   Failed to clone repository: {result.stderr}")
            return None
    except Exception as e:
        print(f"   Failed to clone repository: {e}")
        return None

def convert_to_gguf_standalone(model_path: str, output_dir: str, quantization: str = "q4_k_m"):
    """
    Convert a Hugging Face model to GGUF format for Ollama compatibility.
    
    Args:
        model_path: Path to the Hugging Face model directory (e.g., 'cpu_4epochs_model/merged').
        output_dir: Output directory for the GGUF file and Modelfile.
        quantization: Quantization level (q4_k_m, q5_k_m, etc.).
    """
    print(f"[Attempting GGUF conversion for {model_path} with {quantization} quantization...]\n")

    # Get conversion script (download or clone if needed)
    script_path = get_conversion_script()
    if not script_path:
        return False

    # Create output directory for GGUF files
    gguf_output_dir = Path(output_dir) / "gguf_converted"
    gguf_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    model_name = Path(model_path).name
    gguf_file_name = f"{model_name}.gguf"
    gguf_file_path = gguf_output_dir / gguf_file_name
    
    try:
        # Step 1: Convert to f16 GGUF format first
        temp_f16_file = gguf_file_path.with_suffix('.f16.gguf')
        
        cmd1 = [
            sys.executable, script_path,
            model_path,
            "--outfile", str(temp_f16_file),
            "--outtype", "f16"
        ]
        
        print(f"Step 1 - Converting to f16: {' '.join(cmd1)}\n")
        result1 = subprocess.run(cmd1, capture_output=True, text=True)
        
        if result1.returncode != 0:
            print(f"❌ Step 1 failed with return code {result1.returncode}")
            print(f"   Stderr: {result1.stderr}")
            return False
        
        if not temp_f16_file.exists():
            print(f"❌ Step 1 completed but f16 file not found at {temp_f16_file}")
            return False
        
        print(f"✅ Step 1 complete: Created f16 GGUF file")
        
        # Step 2: Quantize if needed (skip if already f16)
        if quantization == "f16":
            # Just rename the file
            temp_f16_file.rename(gguf_file_path)
            final_file = gguf_file_path
        else:
            # Look for quantize executable in llama.cpp directory
            quantize_exe = Path("llama.cpp") / "quantize.exe"
            if not quantize_exe.exists():
                quantize_exe = Path("llama.cpp") / "quantize"
            
            if quantize_exe.exists():
                print(f"Step 2 - Quantizing to {quantization}...")
                cmd2 = [
                    str(quantize_exe),
                    str(temp_f16_file),
                    str(gguf_file_path),
                    quantization
                ]
                
                print(f"Running: {' '.join(cmd2)}")
                result2 = subprocess.run(cmd2, capture_output=True, text=True)
                
                if result2.returncode == 0 and gguf_file_path.exists():
                    print(f"✅ Step 2 complete: Quantized to {quantization}")
                    # Clean up temp file
                    try:
                        temp_f16_file.unlink()
                    except:
                        pass
                    final_file = gguf_file_path
                else:
                    print(f"⚠️  Step 2 failed, using f16 version instead")
                    temp_f16_file.rename(gguf_file_path)
                    final_file = gguf_file_path
            else:
                print(f"⚠️  quantize executable not found, using f16 version instead")
                temp_f16_file.rename(gguf_file_path)
                final_file = gguf_file_path
        
        if final_file.exists():
            file_size = os.path.getsize(final_file) / (1024 * 1024)  # MB
            print(f"✅ GGUF conversion successful!")
            print(f"   GGUF File: {final_file}")
            print(f"   Size: {file_size:.1f} MB")
            
            # Create Modelfile for Ollama
            create_modelfile(final_file, model_name, gguf_output_dir)
            
            return True
        else:
            print(f"❌ Final GGUF file not found")
            return False
            
    except Exception as e:
        print(f"❌ An error occurred during GGUF conversion: {e}")
        return False

def create_modelfile(gguf_file_path: Path, model_name: str, output_dir: Path):
    """
    Create a Modelfile for Ollama deployment.
    """
    modelfile_content = f"""FROM {gguf_file_path.name}
TEMPLATE "{{{{ .Prompt }}}}"
PARAMETER stop "###"
PARAMETER stop "<|endoftext|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.1
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_predict 10
SYSTEM You are a sentiment analysis model trained on tweets. You classify tweets as negative, neutral, or positive. Respond with only the sentiment label: positive, negative, or neutral.
"""
    
    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"📄 Created Modelfile: {modelfile_path}")
    print(f"   To use with Ollama:")
    print(f"   cd {output_dir}")
    print(f"   ollama create {model_name} -f Modelfile")
    print(f"   ollama run {model_name}")

def main():
    parser = argparse.ArgumentParser(description="Convert Hugging Face model to GGUF format.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the Hugging Face model directory (e.g., 'my_model/merged').")
    parser.add_argument("--output_dir", type=str, default="./",
                        help="Directory to save the GGUF file and Modelfile.")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                        choices=["q2_k", "q3_k_s", "q3_k_m", "q3_k_l", "q4_0", "q4_1", "q4_k_s", "q4_k_m", "q5_0", "q5_1", "q5_k_s", "q5_k_m", "q6_k", "q8_0"],
                        help="Quantization level for GGUF conversion.")
    
    args = parser.parse_args()
    
    convert_to_gguf_standalone(args.model_path, args.output_dir, args.quantization)

if __name__ == "__main__":
    main()