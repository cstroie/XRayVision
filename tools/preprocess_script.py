#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with AI and WebSocket dashboard.
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
# 
# Preprocess X-ray images for MedGemma fine-tuning
# Run this after export_xray_data.py

import os
from pathlib import Path
from PIL import Image
import json
from tqdm import tqdm

def preprocess_images(dataset_dir="./pediatric_xray_dataset", target_size=(896, 896)):
    """
    Preprocess images: resize, normalize, convert to RGB
    
    Args:
        dataset_dir: Directory containing the exported dataset
        target_size: Target image size (width, height)
    """
    
    dataset_path = Path(dataset_dir)
    images_dir = dataset_path / "images"
    processed_dir = dataset_path / "images_processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = list(images_dir.glob("*.png"))
    
    print(f"Found {len(image_files)} images to preprocess")
    print(f"Target size: {target_size}")
    
    stats = {
        "processed": 0,
        "errors": 0,
        "original_sizes": []
    }
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Load image
            img = Image.open(img_path)
            
            # Track original size
            stats["original_sizes"].append(img.size)
            
            # Convert to RGB (in case of grayscale or RGBA)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize while maintaining aspect ratio
            img.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Create a new image with target size and paste the resized image
            new_img = Image.new('RGB', target_size, (0, 0, 0))
            paste_x = (target_size[0] - img.size[0]) // 2
            paste_y = (target_size[1] - img.size[1]) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            # Save processed image
            output_path = processed_dir / img_path.name
            new_img.save(output_path, 'PNG', optimize=True)
            
            stats["processed"] += 1
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            stats["errors"] += 1
    
    # Calculate statistics
    if stats["original_sizes"]:
        avg_width = sum(s[0] for s in stats["original_sizes"]) / len(stats["original_sizes"])
        avg_height = sum(s[1] for s in stats["original_sizes"]) / len(stats["original_sizes"])
    else:
        avg_width, avg_height = 0, 0
    
    print(f"\n=== Preprocessing Summary ===")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    print(f"Average original size: {avg_width:.0f}x{avg_height:.0f}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    print(f"Processed images saved to: {processed_dir}")
    
    # Update JSONL files to point to processed images
    for split in ["train", "val", "test"]:
        jsonl_path = dataset_path / f"{split}.jsonl"
        if not jsonl_path.exists():
            continue
            
        updated_entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                # Update image path to processed version
                entry["image"] = entry["image"].replace("images/", "images_processed/")
                updated_entries.append(entry)
        
        # Write updated JSONL
        with open(jsonl_path, 'w') as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + '\n')
        
        print(f"Updated {split}.jsonl with processed image paths")

if __name__ == "__main__":
    preprocess_images()
