# MedGemma Pediatric Chest X-ray Fine-Tuning
# Google Colab Notebook
# Make sure to enable GPU: Runtime > Change runtime type > T4 GPU

# ============================================================================
# CELL 1: Check GPU and Install Dependencies
# ============================================================================

# Check if GPU is available
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: No GPU detected! Please enable GPU in Runtime settings.")

# ============================================================================
# CELL 2: Install Required Packages
# ============================================================================

!pip install -q accelerate peft transformers bitsandbytes datasets pillow tqdm

# ============================================================================
# CELL 3: Mount Google Drive
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

# TODO: After running export scripts locally, upload the 'pediatric_xray_dataset' 
# folder to your Google Drive. Update the path below:
DATASET_PATH = "/content/drive/MyDrive/pediatric_xray_dataset"

# ============================================================================
# CELL 4: Import Libraries
# ============================================================================

import json
import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# ============================================================================
# CELL 5: Create Custom Dataset Class
# ============================================================================

class PediatricXrayDataset(Dataset):
    """Custom dataset for pediatric chest X-rays with reports"""
    
    def __init__(self, jsonl_path, dataset_root, processor, max_length=512):
        self.dataset_root = Path(dataset_root)
        self.processor = processor
        self.max_length = max_length
        
        # Load data from JSONL
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load image
        img_path = self.dataset_root / item['image']
        image = Image.open(img_path).convert('RGB')
        
        # Create prompt with age information
        age_group = item['age_group']
        prompt = f"Analyze this pediatric chest X-ray (age group: {age_group}) and provide a detailed radiology report."
        
        # Target report
        report = item['report']
        
        # Process with the model's processor
        # Note: MedGemma uses Gemma architecture, adjust based on actual model
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Add labels (the report text)
        labels = self.processor.tokenizer(
            report,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )["input_ids"]
        
        encoding["labels"] = labels
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        return encoding

# ============================================================================
# CELL 6: Load Model with Quantization
# ============================================================================

# Model configuration
MODEL_NAME = "google/medgemma-2b"  # Adjust based on available MedGemma model
OUTPUT_DIR = "/content/drive/MyDrive/medgemma_pediatric_finetuned"

# Quantization config for memory efficiency (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
# Note: Adjust based on actual MedGemma model architecture
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True  # May be needed for custom models
)

print(f"Model loaded: {MODEL_NAME}")
print(f"Model size: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")

# ============================================================================
# CELL 7: Configure LoRA
# ============================================================================

# Prepare model for training
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA alpha
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Adjust based on model
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Print trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# ============================================================================
# CELL 8: Load Datasets
# ============================================================================

train_dataset = PediatricXrayDataset(
    jsonl_path=os.path.join(DATASET_PATH, "train.jsonl"),
    dataset_root=DATASET_PATH,
    processor=processor
)

val_dataset = PediatricXrayDataset(
    jsonl_path=os.path.join(DATASET_PATH, "val.jsonl"),
    dataset_root=DATASET_PATH,
    processor=processor
)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")

# ============================================================================
# CELL 9: Training Configuration
# ============================================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,  # Start with 3 epochs
    per_device_train_batch_size=1,  # Small batch for free Colab
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Effective batch size = 8
    learning_rate=2e-4,
    warmup_steps=100,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    fp16=True,  # Mixed precision training
    dataloader_num_workers=2,
    remove_unused_columns=False,
    report_to="none",  # Can change to "tensorboard" if desired
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

print("Training configuration:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size: {training_args.per_device_train_batch_size}")
print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"  Learning rate: {training_args.learning_rate}")

# ============================================================================
# CELL 10: Initialize Trainer and Start Training
# ============================================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n" + "="*50)
print("Starting training...")
print("="*50 + "\n")

# Train the model
trainer.train()

print("\n" + "="*50)
print("Training completed!")
print("="*50)

# ============================================================================
# CELL 11: Save Model
# ============================================================================

# Save the fine-tuned LoRA adapters
model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

print(f"\nModel saved to: {OUTPUT_DIR}")
print("You can now download this folder from Google Drive")

# ============================================================================
# CELL 12: Test Inference
# ============================================================================

# Load a test image
test_jsonl = os.path.join(DATASET_PATH, "test.jsonl")
with open(test_jsonl, 'r') as f:
    test_sample = json.loads(f.readline())

test_img_path = os.path.join(DATASET_PATH, test_sample['image'])
test_image = Image.open(test_img_path).convert('RGB')

# Create prompt
prompt = f"Analyze this pediatric chest X-ray (age group: {test_sample['age_group']}) and provide a detailed radiology report."

# Generate report
inputs = processor(text=prompt, images=test_image, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True
    )

generated_report = processor.decode(outputs[0], skip_special_tokens=True)

print("="*50)
print("TEST INFERENCE")
print("="*50)
print(f"\nAge group: {test_sample['age_group']}")
print(f"\nGround truth report:\n{test_sample['report']}")
print(f"\nGenerated report:\n{generated_report}")

# ============================================================================
# CELL 13: Evaluate on Test Set (Optional)
# ============================================================================

test_dataset = PediatricXrayDataset(
    jsonl_path=os.path.join(DATASET_PATH, "test.jsonl"),
    dataset_root=DATASET_PATH,
    processor=processor
)

print(f"Evaluating on {len(test_dataset)} test samples...")
test_results = trainer.evaluate(test_dataset)
print(f"Test loss: {test_results['eval_loss']:.4f}")
