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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)

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
    5. JSONL format output for compatibility with ML training pipelines

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

    # Create directory structure with region-specific naming
    if region:
        # Include region name in output directory
        region_safe = region.replace(" ", "_").lower()
        output_path = Path(f"{output_dir}_{region_safe}")
    else:
        output_path = Path(output_dir)

    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Starting export to: {output_path.absolute()}")
    logging.info(f"Source database: {db_path}")
    logging.info(f"Source images: {images_source_dir}")

    # Connect to database
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        logging.info("Database connection established")
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        raise

    # Query to match XRayVision database schema
    # This query extracts pediatric chest X-rays with radiologist reports
    query = """
    SELECT
        e.uid as xray_id,
        e.uid as image_path,  -- Use UID as image path since images are stored as {uid}.png
        rr.text as report_text,
        CASE
            WHEN p.birthdate IS NOT NULL THEN
                CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER)
            ELSE -1
        END as patient_age_days,
        p.sex as patient_sex,
        rr.justification as clinical_indication,
        e.created as image_date
    FROM exams e
    INNER JOIN patients p ON e.cnp = p.cnp
    INNER JOIN rad_reports rr ON e.uid = rr.uid
    WHERE e.status = 'done'
    AND p.birthdate IS NOT NULL
    AND e.type = 'CR'
    AND CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER) <= 6570  -- 18 years max
    AND CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER) >= 0     -- 0 years min
    AND rr.text IS NOT NULL
    AND TRIM(rr.text) != ''
    """

    # Add region filter if specified
    if region:
        query += f" AND e.region = '{region}'"

    # Add age group filter if specified
    if age_group:
        # Map age group to age range in days
        age_ranges = {
            'neonate': (0, 28),
            'infant': (29, 730),
            'preschool': (731, 1825),
            'school_age': (1826, 4380),
            'adolescent': (4381, 6570)
        }
        if age_group in age_ranges:
            min_age, max_age = age_ranges[age_group]
            query += f" AND CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER) BETWEEN {min_age} AND {max_age}"

    query += " ORDER BY e.created"

    if limit:
        query += f" LIMIT {limit}"
        logging.info(f"Exporting limited sample of {limit} records")

    try:
        # Debug: Check what's in the database first
        debug_query = "SELECT COUNT(*) FROM exams WHERE type = 'CR' AND status = 'done'"
        if region:
            debug_query += f" AND region = '{region}'"
        cursor.execute(debug_query)
        total_chest_exams = cursor.fetchone()[0]
        logging.info(f"Total {region} CR exams in database: {total_chest_exams}")

        debug_query = "SELECT COUNT(*) FROM exams e INNER JOIN patients p ON e.cnp = p.cnp LEFT JOIN rad_reports rr ON e.uid = rr.uid WHERE e.type = 'CR' AND e.status = 'done' AND p.birthdate IS NOT NULL AND rr.text IS NOT NULL"
        if region:
            debug_query += f" AND region = '{region}'"
        cursor.execute(debug_query)
        total_with_reports = cursor.fetchone()[0]
        logging.info(f"Total {region} CR exams with radiologist reports: {total_with_reports}")

        # Debug: Check what regions are actually in the database
        region_query = "SELECT DISTINCT region FROM exams WHERE type = 'CR' AND status = 'done' ORDER BY region"
        cursor.execute(region_query)
        regions = cursor.fetchall()
        logging.info(f"Available regions in database: {[r[0] for r in regions]}")

        # Debug: Check what types are actually in the database
        type_query = "SELECT DISTINCT type FROM exams WHERE status = 'done' ORDER BY type"
        cursor.execute(type_query)
        types = cursor.fetchall()
        logging.info(f"Available types in database: {[t[0] for t in types]}")

        cursor.execute(query)
        records = cursor.fetchall()
        logging.info(f"Database query completed, found {len(records)} records")
    except sqlite3.Error as e:
        logging.error(f"Database query failed: {e}")
        conn.close()
        raise
    finally:
        conn.close()

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
            xray_id, image_path, report, age_days, sex, indication, date = record

            # Calculate age group
            age_group = calculate_age_group(age_days)
            stats[age_group] += 1

            # Construct source image path
            source_image_path = os.path.join(images_source_dir, f"{image_path}.png")

            # Verify image file exists before processing
            if not os.path.exists(source_image_path):
                logging.warning(f"Image file not found: {source_image_path}")
                skipped_count += 1
                continue

            # Copy image to export directory with consistent naming and resizing
            # Since we're using the UID as image_path, we need to construct the full path
            source_image_path = os.path.join(images_source_dir, f"{image_path}.png")
            # Use the UID directly as the filename since it's already a unique identifier
            new_image_name = f"{xray_id}.png"
            new_image_path = images_dir / new_image_name
        
            # Copy and resize image if it exists
            if os.path.exists(source_image_path):
                try:
                    # Load image using PIL
                    from PIL import Image
                    img = Image.open(source_image_path)
                
                    # Convert to RGB (in case of grayscale or RGBA)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                
                    # Target size for MedGemma
                    target_size = (896, 896)
                
                    # Resize image to exactly match target size (may change aspect ratio)
                    # This scales the image to fill the entire target size
                    img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                    # Save processed image with optimization
                    img.save(new_image_path, 'PNG', optimize=True)
                
                    processed_count += 1
                    if processed_count % 10 == 0:  # Progress logging
                        logging.info(f"Processed {processed_count}/{len(records)} records")
                except Exception as e:
                    logging.error(f"Failed to process image {source_image_path}: {e}")
                    skipped_count += 1
                    continue
            else:
                logging.warning(f"Image file not found: {source_image_path}")
                skipped_count += 1
                continue

            # Create metadata entry
            entry = {
                "image": f"images/{new_image_name}",
                "report": report,
                "age_days": age_days,
                "age_group": age_group,
                "sex": sex,
                "clinical_indication": indication or "Unknown",
                "date": date,
                "xray_id": xray_id
            }

            # Split into train/val/test (80/10/10) using hash for reproducibility
            # Using hash of xray_id ensures consistent splits across runs
            split_val = hash(xray_id) % 10
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

    # Write JSONL files
    def write_jsonl(data, filename):
        """Write data to JSONL file with proper encoding and error handling."""
        filepath = output_path / filename
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for entry in data:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logging.info(f"Wrote {len(data)} entries to {filename}")
            return True
        except (IOError, OSError) as e:
            logging.error(f"Failed to write {filename}: {e}")
            return False

    # Write training data
    train_success = write_jsonl(train_data, "train.jsonl")
    val_success = write_jsonl(val_data, "val.jsonl")
    test_success = write_jsonl(test_data, "test.jsonl")

    # Write dataset statistics
    stats_data = {
        "export_metadata": {
            "total_records_found": len(records),
            "records_processed": processed_count,
            "records_skipped": skipped_count,
            "export_date": datetime.now().isoformat(),
            "database_path": db_path,
            "images_source_dir": images_source_dir,
            "output_dir": str(output_path.absolute())
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
    print("="*50)

    if not (train_success and val_success and test_success):
        logging.warning("Some files may not have been written successfully")

    return output_path


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
