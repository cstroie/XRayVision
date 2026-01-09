#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from PIL import Image

def resize_images(input_dir, target_width, target_height):
    """
    Resize all images in input_dir to target resolution if they don't already match.
    Saves processed images to 'output' subdirectory.
    
    Args:
        input_dir (str): Directory containing source images
        target_width (int): Target width in pixels
        target_height (int): Target height in pixels
    """
    # Create output directory
    output_dir = os.path.join(input_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(valid_extensions) and not f.startswith('.')]
    
    print(f"Found {len(image_files)} images in {input_dir}")
    processed = 0
    skipped = 0
    
    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # Check if output already exists
        if os.path.exists(output_path):
            skipped += 1
            continue
        
        try:
            # Open and convert to grayscale
            img = Image.open(input_path)
            if img.mode != 'L':
                img = img.convert('L')
            
            width, height = img.size
            print(f"Processing {filename}: original size {width}x{height}")
            
            # Check if resizing is needed based on specific rules:
            # 1. Both dimensions larger than target
            # 2. Both dimensions smaller than target
            # 3. One dimension equal to target, other dimension larger
            both_larger = width > target_width and height > target_height
            both_smaller = width < target_width and height < target_height
            one_equal_other_larger = (width == target_width and height > target_height) or \
                                     (height == target_height and width > target_width)
            
            needs_resize = both_larger or both_smaller or one_equal_other_larger
            
            print(f"Needs resize: {needs_resize}")
            
            if not needs_resize:
                # Already correct size or one dimension equal/smaller, just save with high compression
                img.save(output_path, 'PNG', compress_level=9)
            else:
                # Resize maintaining aspect ratio to fit within target dimensions
                target_ratio = target_width / target_height
                img_ratio = width / height
                
                if img_ratio > target_ratio:
                    # Image is wider - scale by width
                    new_width = target_width
                    new_height = int(height * (target_width / width))
                else:
                    # Image is taller - scale by height
                    new_height = target_height
                    new_width = int(width * (target_height / height))
                
                # Resize using high-quality resampling
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                                
                # Save with high compression
                img.save(output_path, 'PNG', compress_level=9)
            
            processed += 1
            print(f"Processed: {filename}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            skipped += 1
    
    print(f"\nComplete: {processed} processed, {skipped} skipped")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Resize images to target resolution')
    parser.add_argument('input_dir', help='Directory containing images to resize')
    parser.add_argument('--width', type=int, required=True, help='Target width in pixels')
    parser.add_argument('--height', type=int, required=True, help='Target height in pixels')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        exit(1)
    
    resize_images(args.input_dir, args.width, args.height)
