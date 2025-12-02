#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNG Normalization Tool

This tool applies grayscale normalization to PNG images similar to the 
processing used in XRayVision for DICOM to PNG conversion.
"""

import argparse
import cv2
import numpy as np
import os
import math
import logging

# Logger config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)

def apply_gamma_correction(image, gamma=1.2):
    """
    Apply gamma correction to an image to adjust brightness and contrast.

    Args:
        image: Input image as numpy array
        gamma: Gamma value for correction. If None, will be automatically calculated

    Returns:
        numpy array: Gamma-corrected image
    """
    # If gamma is None, compute it based on image statistics
    if gamma is None:
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.median(image)
        gamma = math.log(mid * 255) / math.log(mean)
        logging.debug(f"Calculated gamma is {gamma:.2f}")
    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def normalize_png(input_file, output_file, max_size=800):
    """
    Normalize a PNG image with preprocessing for optimal visualization.

    This function performs several important preprocessing steps:
    1. Reads the PNG pixel data
    2. Resizes the image while maintaining aspect ratio
    3. Applies percentile clipping to remove outliers
    4. Normalizes pixel values to 0-255 range
    5. Applies automatic gamma correction for better visualization
    6. Saves as PNG

    Args:
        input_file: Path to the input PNG file
        output_file: Path to the output normalized PNG file
        max_size: Maximum dimension for the output image (default: 800)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image from {input_file}")
        
        # Resize while maintaining aspect ratio
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            if height > width:
                new_height = max_size
                new_width = int(width * (max_size / height))
            else:
                new_width = max_size
                new_height = int(height * (max_size / width))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Clip to 1..99 percentiles to remove outliers and improve contrast
        minval = np.percentile(image, 1)
        maxval = np.percentile(image, 99)
        image = np.clip(image, minval, maxval)
        
        # Normalize image to 0-255
        image -= image.min()
        if image.max() != 0:
            image /= image.max()
        image *= 255.0
        
        # Save as 8 bit
        image = image.astype(np.uint8)
        
        # Auto adjust gamma
        image = apply_gamma_correction(image, None)
        
        # Save the normalized PNG file
        cv2.imwrite(output_file, image)
        logging.info(f"Normalized PNG saved to {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error normalizing PNG: {e}")
        return False


def main():
    """Main function to process PNG files."""
    parser = argparse.ArgumentParser(description="Normalize PNG images for better visualization")
    parser.add_argument("input", help="Input PNG file path")
    parser.add_argument("-o", "--output", help="Output PNG file path (default: add '_normalized' suffix)")
    parser.add_argument("--max-size", type=int, default=800, help="Maximum dimension for output image")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        logging.error(f"Input file does not exist: {args.input}")
        return 1
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        name, ext = os.path.splitext(args.input)
        output_file = f"{name}_normalized{ext}"
    
    # Process the image
    if normalize_png(args.input, output_file, args.max_size):
        logging.info("Normalization completed successfully")
        return 0
    else:
        logging.error("Normalization failed")
        return 1


if __name__ == "__main__":
    exit(main())
