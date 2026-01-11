#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with AI and WebSocket dashboard.
# Copyright (C) 2026 Costin Stroie <costinstroie@eridu.eu.org>
#
# Export pediatric chest X-ray data in MedGemma-optimized format
# Creates a dataset with images/ directory and data.jsonl file

import json
import os
import shutil
import sqlite3
import logging
import hashlib
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s'
)

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

def query_records(conn, limit=None, region=None):
    """Query records from database with filters."""
    # Base query to get exams with radiologist reports
    query = """
    SELECT
        e.uid as study_id,
        e.uid as image_name,
        rr.text_en as report_text,
        rr.summary as report_summary,
        e.region,
        e.created
    FROM exams e
    INNER JOIN patients p ON e.cnp = p.cnp
    INNER JOIN rad_reports rr ON e.uid = rr.uid
    WHERE e.status = 'done'
    AND rr.text_en IS NOT NULL
    AND TRIM(rr.text_en) != ''
    """

    # Add region filter if specified
    if region:
        query += f" AND e.region = '{region}'"

    query += " ORDER BY e.created"

    if limit:
        query += f" LIMIT {limit}"
        logging.info(f"Exporting limited sample of {limit} records")

    cursor = conn.cursor()
    cursor.execute(query)
    records = cursor.fetchall()
    logging.info(f"Database query completed, found {len(records)} records")

    return records

def generate_md5_filename(filename):
    """Generate MD5 hash from filename for consistent naming."""
    return hashlib.md5(filename.encode('utf-8')).hexdigest()

def create_medgemma_entry(record, images_source_dir):
    """Create a MedGemma-optimized dataset entry for report generation."""
    study_id, image_name, report_text, report_summary, region, created = record

    # Determine modality based on region
    if region.lower() == 'chest':
        modality = 'chest_xray'
    else:
        modality = f"{region.lower()}_xray"

    # Format the response as a structured radiology report
    # Use report summary if available, otherwise use full report
    if report_summary:
        response = report_summary
    else:
        response = report_text

    # Generate MD5 hash for filename
    md5_filename = generate_md5_filename(image_name)

    # Create entry with the new schema including summary field
    entry = {
        "image": f"images/{md5_filename}.png",
        "response": report_text,  # Use text_en for the response field
        "summary": report_summary,
        "modality": modality,
        "original_filename": image_name  # Keep original filename for reference
    }

    return entry

def export_medgemma_dataset(output_dir="./export/medgemma_dataset", limit=None, db_path="./export/xrayvision.db", images_source_dir="./images", region=None):
    """
    Export pediatric chest X-ray data in MedGemma-optimized format.

    Creates a dataset with:
    - images/ directory containing PNG images
    - data.jsonl file with entries in MedGemma format

    Args:
        output_dir (str): Directory to save exported data (default: "./medgemma_dataset")
        limit (int, optional): Maximum number of records to export
        db_path (str): Path to XRayVision SQLite database file
        images_source_dir (str): Directory containing source PNG image files
        region (str, optional): Filter by anatomic region (e.g., "chest")

    Returns:
        Path: Path to the created export directory
    """

    # Validate input parameters
    if not os.path.exists(images_source_dir):
        raise FileNotFoundError(f"Source images directory not found: {images_source_dir}")

    output_path = Path(output_dir)
    images_output_dir = output_path / "images"

    # Create output directory structure
    os.makedirs(images_output_dir, exist_ok=True)

    logging.info(f"Starting MedGemma dataset export to: {output_path.absolute()}")
    logging.info(f"Source database: {db_path}")
    logging.info(f"Source images: {images_source_dir}")

    # Connect to database
    conn = connect_to_database(db_path)

    try:
        # Query records
        records = query_records(conn, limit, region)

        if not records:
            logging.warning("No records found matching export criteria")
            return output_path

        # Process records and create dataset entries
        dataset_entries = []
        processed_count = 0
        skipped_count = 0

        logging.info("Processing records...")

        for idx, record in enumerate(records):
            try:
                entry = create_medgemma_entry(record, images_source_dir)
                dataset_entries.append(entry)

                # Generate MD5 filename for the image
                original_filename = record[1]
                md5_filename = generate_md5_filename(original_filename)

                # Copy image file with MD5 filename
                source_image_path = os.path.join(images_source_dir, f"{original_filename}.png")
                target_image_path = images_output_dir / f"{md5_filename}.png"

                if os.path.exists(source_image_path):
                    shutil.copy2(source_image_path, target_image_path)
                    processed_count += 1
                else:
                    logging.warning(f"Image file not found: {source_image_path}")
                    skipped_count += 1

            except Exception as e:
                logging.error(f"Error processing record {idx}: {e}")
                skipped_count += 1
                continue

        # Write data.jsonl file
        data_jsonl_path = output_path / "data.jsonl"
        try:
            with open(data_jsonl_path, 'w', encoding='utf-8') as f:
                for entry in dataset_entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            logging.info(f"Wrote {len(dataset_entries)} entries to data.jsonl")
        except (IOError, OSError) as e:
            logging.error(f"Failed to write data.jsonl: {e}")
            return output_path

        # Print summary
        print("\n" + "="*50)
        print("MEDGEMMA DATASET EXPORT SUMMARY")
        print("="*50)
        print(f"Total records found: {len(records)}")
        print(f"Records processed: {processed_count}")
        print(f"Records skipped: {skipped_count}")
        print(f"Success rate: {round(processed_count / max(1, len(records)) * 100, 1)}%")
        print()
        print(f"Export location: {output_path.absolute()}")
        print(f"Dataset format: MedGemma-optimized (images/ directory + data.jsonl)")
        print(f"Total entries in data.jsonl: {len(dataset_entries)}")
        print("="*50)

        return output_path

    finally:
        conn.close()

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Export data in MedGemma-optimized format")
    parser.add_argument("--limit", type=int, help="Maximum number of exams to export")
    parser.add_argument("--region", type=str, help="Filter by anatomic region (e.g., chest)")
    parser.add_argument("--output-dir", type=str, default="./export/medgemma_dataset",
                       help="Output directory for exported data")
    parser.add_argument("--db-path", type=str, default="./export/xrayvision.db",
                       help="Path to XRayVision database file")
    parser.add_argument("--images-source-dir", type=str, default="./images",
                       help="Directory containing source PNG image files")

    args = parser.parse_args()

    print("Starting MedGemma dataset export with the following parameters:")
    print(f"  Limit: {args.limit or 'all'}")
    print(f"  Region: {args.region or 'all'}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Database path: {args.db_path}")
    print(f"  Images source directory: {args.images_source_dir}")
    print()

    # Run export with provided arguments
    export_medgemma_dataset(
        output_dir=args.output_dir,
        limit=args.limit,
        db_path=args.db_path,
        images_source_dir=args.images_source_dir,
        region=args.region
    )
