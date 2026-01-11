#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with AI and WebSocket dashboard.
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# Export pediatric chest X-ray data from database for MedGemma fine-tuning
# Run this script locally on your machine with database access


import json
import os
from pathlib import Path
from datetime import datetime
import shutil
import sqlite3
import logging


def calculate_age_group(age_days):
    """
    Classify patient into age group based on days since birth.

    Maps patient age in days to standardized pediatric age groups used
    in medical imaging and research. These age groups align with common
    pediatric radiology classifications.

    Args:
        age_days (int): Patient age in days since birth

    Returns:
        str: Age group classification:
            - "neonate": 0-28 days (newborn period)
            - "infant": 29 days - 2 years
            - "preschool": 2-5 years
            - "school_age": 5-12 years
            - "adolescent": 12-18 years

    Note:
        Ages outside the valid range (negative or >18 years) default to "adolescent"
    """
    if age_days <= 28:
        return "neonate"
    elif age_days <= 730:  # 2 years = 365 * 2 = 730 days
        return "infant"
    elif age_days <= 1825:  # 5 years = 365 * 5 = 1825 days
        return "preschool"
    elif age_days <= 4380:  # 12 years = 365 * 12 = 4380 days
        return "school_age"
    else:
        return "adolescent"


def connect_to_database(db_path):
    """Connect to database and copy from source if needed."""
    # Ensure the export directory exists
    export_dir = os.path.dirname(db_path)
    if export_dir and not os.path.exists(export_dir):
        os.makedirs(export_dir, exist_ok=True)

    # Check if database exists, if not, copy from xrayvision-multi.db
    if not os.path.exists(db_path):
        source_db = "xrayvision-multi.db"
        if os.path.exists(source_db):
            logging.info(f"Database not found at {db_path}, copying from {source_db}")
            try:
                # Use SQLite backup API for secure copying with proper transaction handling
                with sqlite3.connect(source_db) as source_conn:
                    backup_conn = sqlite3.connect(db_path)
                    source_conn.backup(backup_conn)
                    backup_conn.close()
                logging.info(f"Successfully copied database to {db_path}")
            except (IOError, OSError, sqlite3.Error) as e:
                logging.error(f"Failed to copy database from {source_db} to {db_path}: {e}")
                raise FileNotFoundError(f"Could not create database at {db_path}: {e}")
        else:
            raise FileNotFoundError(f"Database file not found at {db_path} and source {source_db} does not exist")

    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        logging.info("Database connection established")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise


def query_records(conn, limit=None, region=None, age_group=None):
    """Query records from database with filters."""
    # Map age group to age range in days
    age_ranges = {
        'neonate': (0, 28),
        'infant': (29, 730),
        'preschool': (731, 1826),
        'school_age': (1827, 4383),
        'adolescent': (4384, 6574)
    }

    # Base query using CTE to calculate age first
    # This allows us to filter by the calculated age_days
    query = """
    WITH ExamData AS (
        SELECT
            e.uid as xray_id,
            e.uid as image_path,
            rr.text_en as report_text,
            rr.summary as report_summary,
            CASE
                WHEN p.birthdate IS NOT NULL THEN
                    CAST((julianday(e.created) - julianday(p.birthdate)) AS INTEGER)
                ELSE -1
            END as patient_age_days,
            p.sex as patient_sex,
            e.created as image_date,
            e.region
        FROM exams e
        INNER JOIN patients p ON e.cnp = p.cnp
        INNER JOIN rad_reports rr ON e.uid = rr.uid
        WHERE e.status = 'done'
        AND p.birthdate IS NOT NULL
        AND rr.text_en IS NOT NULL
        AND TRIM(rr.text_en) != ''
    )
    SELECT
        xray_id,
        image_path,
        report_text,
        report_summary,
        patient_age_days,
        patient_sex,
        image_date,
        region
    FROM ExamData
    WHERE patient_age_days > 0 AND patient_age_days <= 6574
    """

    # Add region filter if specified
    if region:
        query += f" AND region = '{region}'"

    # Add age group filter if specified using the calculated patient_age_days
    if age_group and age_group in age_ranges:
        min_age, max_age = age_ranges[age_group]
        query += f" AND patient_age_days BETWEEN {min_age} AND {max_age}"

    query += " ORDER BY image_date"

    if limit:
        query += f" LIMIT {limit}"
        logging.info(f"Exporting limited sample of {limit} records")

    cursor = conn.cursor()

    cursor.execute(query)
    records = cursor.fetchall()
    logging.info(f"Database query completed, found {len(records)} records")

    return records


def process_record(record, images_source_dir, split_dirs, stats, processed_count, skipped_count):
    """Process a single record: validate image, copy file, create entry."""
    xray_id, image_path, report, report_summary, age_days, sex, date, region = record

    # Calculate age group
    age_group = calculate_age_group(age_days)
    stats[age_group] += 1

    # Map sex to gender (boy/girl/child)
    if sex == 'M':
        gender = 'boy'
    elif sex == 'F':
        gender = 'girl'
    else:
        gender = 'child'

    # Construct source image path
    source_image_path = os.path.join(images_source_dir, f"{image_path}.png")

    # Verify image file exists before processing
    if not os.path.exists(source_image_path):
        logging.warning(f"Image file not found: {source_image_path}")
        return None, processed_count, skipped_count + 1

    # Create metadata entry
    entry = {
        "file_name": f"{xray_id}.png",
        "report": report,
        "report_summary": report_summary,
        "age_days": age_days,
        "age_group": age_group,
        "gender": gender,
        "date": date,
        "region": region
    }

    return entry, split_dirs, processed_count, skipped_count


def write_metadata_and_copy_images(data, split_name, output_path, images_source_dir):
    """
    Write metadata.jsonl and copy images to the appropriate split directory.

    Args:
        data: List of metadata entries for this split
        split_name: Name of the split (train, val, test)
        output_path: Base output directory
        images_source_dir: Source directory containing PNG images

    Returns:
        bool: True if successful, False otherwise
    """
    split_dir = output_path / split_name
    os.makedirs(split_dir, exist_ok=True)

    # Write metadata.jsonl
    metadata_path = split_dir / "metadata.jsonl"
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        logging.info(f"Wrote {len(data)} entries to {split_name}/metadata.jsonl")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write {split_name}/metadata.jsonl: {e}")
        return False

    # Copy images to split directory
    for entry in data:
        source_path = os.path.join(images_source_dir, f"{entry['file_name'].replace('.png', '')}.png")
        target_path = split_dir / entry['file_name']

        if not os.path.exists(target_path):
            try:
                shutil.copy2(source_path, target_path)
            except Exception as e:
                logging.error(f"Failed to copy image {source_path} to {target_path}: {e}")
                return False

    return True

def write_dataset_card(output_path, region=None):
    """
    Write dataset card (README.md) to the output directory.

    Args:
        output_path: Base output directory
        region: Optional region filter used for export
    """
    dataset_card_content = """---
license: cc-by-4.0
task_categories:
- image-classification
- image-to-text
---

## Dataset description

This dataset contains pediatric chest X-ray images with associated radiologist reports for medical AI research and model training.
The dataset is optimized for MedGemma fine-tuning and includes comprehensive metadata for each case.
"""

    if region:
        dataset_card_content += f"\n\n### Region Filter\nThis export is filtered for the anatomic region: `{region}`\n"

    dataset_card_content += """
## Dataset Structure

The dataset is organized in the standard Hugging Face image dataset format:

```
pediatric_xray_dataset/
├── train/
│   ├── metadata.jsonl      # Training set metadata
│   └── *.png               # Training images
├── val/
│   ├── metadata.jsonl      # Validation set metadata
│   └── *.png               # Validation images
├── test/
│   ├── metadata.jsonl      # Test set metadata
│   └── *.png               # Test images
├── dataset_stats.json      # Comprehensive dataset statistics
└── README.md               # This dataset card
```

## Metadata Fields

Each entry in the metadata.jsonl files contains:

- `file_name`: Image filename (e.g., "12345.png")
- `report`: Full radiologist report text in English
- `report_summary`: Summary of the radiologist findings
- `age_days`: Patient age in days since birth
- `age_group`: Pediatric age classification (neonate, infant, preschool, school_age, adolescent)
- `gender`: Patient gender (boy, girl, child)
- `date`: Exam date (YYYY-MM-DD)
- `region`: Anatomic region (e.g., "chest")

## Usage

This dataset is designed for fine-tuning medical vision-language models like MedGemma:

```python
from datasets import load_dataset

dataset = load_dataset("path/to/pediatric_xray_dataset")
```

## Citation

If you use this dataset in your research, please cite the original XRayVision project and the source medical institution.
"""

    try:
        with open(output_path / "README.md", 'w', encoding='utf-8') as f:
            f.write(dataset_card_content)
        logging.info("Wrote dataset card to README.md")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write dataset card: {e}")


def print_summary(records, processed_count, skipped_count, train_data, val_data, test_data, stats, output_path):
    """Print export summary and write statistics file."""
    # Write dataset statistics
    stats_data = {
        "export_metadata": {
            "total_records_found": len(records),
            "records_processed": processed_count,
            "records_skipped": skipped_count,
            "export_date": datetime.now().isoformat(),
            "output_dir": str(output_path.absolute()),
            "dataset_format": "image_dataset"
        },
        "dataset_splits": {
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "train_percentage": round(len(train_data) / max(1, processed_count) * 100, 1),
            "val_percentage": round(len(val_data) / max(1, processed_count) * 100, 1),
            "test_percentage": round(len(test_data) / max(1, processed_count) * 100, 1)
        },
        "age_distribution": stats,
        "age_distribution_percentages": {
            group: round(count / max(1, processed_count) * 100, 1)
            for group, count in stats.items()
        },
        "dataset_structure": {
            "train": "train/ - contains images and metadata.jsonl",
            "val": "val/ - contains images and metadata.jsonl",
            "test": "test/ - contains images and metadata.jsonl"
        }
    }

    try:
        with open(output_path / "dataset_stats.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2, ensure_ascii=False)
        logging.info("Wrote dataset statistics to dataset_stats.json")
    except (IOError, OSError) as e:
        logging.error(f"Failed to write dataset statistics: {e}")

    # Print summary
    print("\n" + "="*50)
    print("EXPORT SUMMARY")
    print("="*50)
    print(f"Total records found: {len(records)}")
    print(f"Records processed: {processed_count}")
    print(f"Records skipped: {skipped_count}")
    print(f"Success rate: {round(processed_count / max(1, len(records)) * 100, 1)}%")
    print()
    print("Dataset splits:")
    print(f"  Train: {len(train_data)} ({round(len(train_data) / max(1, processed_count) * 100, 1)}%)")
    print(f"  Validation: {len(val_data)} ({round(len(val_data) / max(1, processed_count) * 100, 1)}%)")
    print(f"  Test: {len(test_data)} ({round(len(test_data) / max(1, processed_count) * 100, 1)}%)")
    print()
    print("Age distribution:")
    for age_group, count in stats.items():
        percentage = round(count / max(1, processed_count) * 100, 1)
        print(f"  {age_group}: {count} ({percentage}%)")
    print()
    print(f"Export location: {output_path.absolute()}")
    print(f"Dataset format: image_dataset (train/val/test directories with metadata.jsonl)")
    print("="*50)


def export_data(output_dir="./export/pediatric_xray_dataset", limit=None, db_path="./export/xrayvision.db", images_source_dir="./images", region=None, age_group=None):
    """
    Export pediatric chest X-ray data from XRayVision database for MedGemma fine-tuning.

    This function extracts pediatric chest X-ray images and associated radiologist reports
    from the XRayVision database, organizes them into training/validation/test splits,
    and creates a structured dataset suitable for fine-tuning medical AI models.

    The export process includes:
    1. Database querying with pediatric age filtering (0-18 years)
    2. Image file validation and copying
    3. Reproducible dataset splitting (80/10/10 train/val/test)
    4. Metadata generation and statistics collection
    5. Image dataset format output compatible with Hugging Face datasets library

    Args:
        output_dir (str): Directory to save exported data (default: "./export/pediatric_xray_dataset")
        limit (int, optional): Maximum number of records to export (for testing/debugging)
        db_path (str): Path to XRayVision SQLite database file (default: "./export/xrayvision.db")
        images_source_dir (str): Directory containing source PNG image files (default: "./images")
        region (str, optional): Filter by anatomic region (e.g., "chest", "abdomen")
        age_group (str, optional): Filter by age group (e.g., "infant", "school_age")

    Returns:
        Path: Path to the created export directory

    Raises:
        sqlite3.Error: If database connection or query fails
        FileNotFoundError: If source image directory doesn't exist
        PermissionError: If output directory cannot be created or written to

    Example:
        >>> # Export full dataset
        >>> export_path = export_data()
        >>>
        >>> # Export limited sample for testing
        >>> export_path = export_data(limit=50)
        >>>
        >>> # Export with custom paths and filters
        >>> export_path = export_data(
        ...     output_dir="/data/medgemma_dataset",
        ...     db_path="/opt/xrayvision/data.db",
        ...     images_source_dir="/opt/xrayvision/images",
        ...     region="chest",
        ...     age_group="infant"
        ... )
    """

    # Validate input parameters
    if not os.path.exists(images_source_dir):
        raise FileNotFoundError(f"Source images directory not found: {images_source_dir}")

    # Create directory structure with region-specific naming
    if region:
        # Include region name in output directory
        region_safe = region.replace(" ", "_").lower()
        output_path = Path(f"{output_dir}_{region_safe}")
    else:
        output_path = Path(output_dir)

    # Create split directories (train, val, test)
    os.makedirs(output_path / "train", exist_ok=True)
    os.makedirs(output_path / "val", exist_ok=True)
    os.makedirs(output_path / "test", exist_ok=True)

    logging.info(f"Starting export to: {output_path.absolute()}")
    logging.info(f"Source database: {db_path}")
    logging.info(f"Source images: {images_source_dir}")

    # Connect to database
    conn = connect_to_database(db_path)

    try:
        # Query records
        records = query_records(conn, limit, region, age_group)

        if not records:
            logging.warning("No records found matching export criteria")
            return output_path

        # Prepare data for export
        train_data = []
        val_data = []
        test_data = []

        stats = {
            "neonate": 0,
            "infant": 0,
            "preschool": 0,
            "school_age": 0,
            "adolescent": 0
        }

        processed_count = 0
        skipped_count = 0

        logging.info("Processing records...")

        for idx, record in enumerate(records):
            try:
                entry, split_dirs, processed_count, skipped_count = process_record(
                    record, images_source_dir, None, stats, processed_count, skipped_count
                )

                if entry is None:
                    continue

                # Split into train/val/test (80/10/10) using hash for reproducibility
                # Using hash of filename ensures consistent splits across runs
                split_val = hash(entry["file_name"]) % 10
                if split_val < 8:
                    train_data.append(entry)
                elif split_val == 8:
                    val_data.append(entry)
                else:
                    test_data.append(entry)

            except (ValueError, TypeError) as e:
                logging.error(f"Error processing record {idx}: {e}")
                skipped_count += 1
                continue

        # Write metadata and copy images to split directories
        train_success = write_metadata_and_copy_images(train_data, "train", output_path, images_source_dir)
        val_success = write_metadata_and_copy_images(val_data, "val", output_path, images_source_dir)
        test_success = write_metadata_and_copy_images(test_data, "test", output_path, images_source_dir)

        # Write dataset card
        write_dataset_card(output_path, region)

        # Print summary
        print_summary(records, processed_count, skipped_count, train_data, val_data, test_data, stats, output_path)

        if not (train_success and val_success and test_success):
            logging.warning("Some files may not have been written successfully")

        return output_path

    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export pediatric X-ray data from XRayVision database")
    parser.add_argument("--limit", type=int, help="Maximum number of exams to export (default: all)")
    parser.add_argument("--region", type=str, help="Filter by anatomic region (e.g., chest, abdomen)")
    parser.add_argument("--age-group", type=str, choices=['neonate', 'infant', 'preschool', 'school_age', 'adolescent'],
                       help="Filter by age group")
    parser.add_argument("--output-dir", type=str, default="./export/pediatric_xray_dataset",
                       help="Output directory for exported data (default: ./export/pediatric_xray_dataset)")
    parser.add_argument("--db-path", type=str, default="./export/xrayvision.db",
                       help="Path to XRayVision database file (default: ./export/xrayvision.db)")
    parser.add_argument("--images-source-dir", type=str, default="./images",
                       help="Directory containing source PNG image files (default: ./images)")

    args = parser.parse_args()

    print("Starting export with the following parameters:")
    print(f"  Limit: {args.limit or 'all'}")
    print(f"  Region: {args.region or 'all'}")
    print(f"  Age group: {args.age_group or 'all'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Database path: {args.db_path}")
    print(f"  Images source directory: {args.images_source_dir}")
    print()

    # Run export with provided arguments
    export_data(
        output_dir=args.output_dir,
        limit=args.limit,
        db_path=args.db_path,
        images_source_dir=args.images_source_dir,
        region=args.region,
        age_group=args.age_group
    )
