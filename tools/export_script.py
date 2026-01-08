#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# XRayVision - Async DICOM processor with AI and WebSocket dashboard.
# Copyright (C) 2025 Costin Stroie <costinstroie@eridu.eu.org>
# 
# Export pediatric chest X-ray data from database for MedGemma fine-tuning
# Run this script locally on your machine with database access


import json
import os
from pathlib import Path
from datetime import datetime
import shutil
import sqlite3

def calculate_age_group(age_days):
    """Classify patient into age group based on days"""
    if age_days <= 28:
        return "neonate"
    elif age_days <= 730:  # 2 years
        return "infant"
    elif age_days <= 1825:  # 5 years
        return "preschool"
    elif age_days <= 4380:  # 12 years
        return "school_age"
    else:
        return "adolescent"

def export_data(output_dir="./export/pediatric_xray_dataset", limit=None):
    """
    Export X-ray images and reports from database
    
    Args:
        output_dir: Directory to save exported data
        limit: Optional limit on number of records (for testing)
    """
    
    # Create directory structure
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # SQLite (simpler for testing):
    conn = sqlite3.connect("./export/xrayvision.db")
    cursor = conn.cursor()
    
    # Query to match XRayVision database schema
    query = """
    SELECT 
        e.uid as xray_id,
        e.uid as image_path,  -- Use UID as image path since images are stored as {uid}.png
        ar.text as report_text,
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
    LEFT JOIN ai_reports ar ON e.uid = ar.uid
    LEFT JOIN rad_reports rr ON e.uid = rr.uid
    WHERE e.type = 'CR'  -- Computed Radiography
    AND e.region = 'chest'  -- Chest X-rays only
    AND e.status = 'done'  -- Only processed exams
    AND (
        p.birthdate IS NOT NULL 
        AND CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER) <= 6570  -- 18 years
        AND CAST((julianday(e.created) - julianday(p.birthdate)) * 365.25 AS INTEGER) >= 0
    )
    AND ar.text IS NOT NULL  -- Only exams with AI reports
    ORDER BY e.created
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    records = cursor.fetchall()
    
    # MOCK DATA for demonstration - replace with actual database query
    #records = [
    #    (1, "xray_001", "Clear lungs, no acute findings.", 45, "M", "Cough", "2024-01-15 10:30:00"),
    #    (2, "xray_002", "Right lower lobe consolidation...", 730, "F", "Fever", "2024-01-16 14:20:00"),
    #    # Add more mock records as needed
    #]
    
    print(f"Found {len(records)} records to export")
    
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
    
    for idx, record in enumerate(records):
        xray_id, image_path, report, age_days, sex, indication, date = record
        
        # Calculate age group
        age_group = calculate_age_group(age_days)
        stats[age_group] += 1
        
        # Copy image to export directory with consistent naming
        # Since we're using the UID as image_path, we need to construct the full path
        source_image_path = os.path.join("../images", f"{image_path}.png")
        # Use the UID directly as the filename since it's already a unique identifier
        new_image_name = f"xray_{xray_id}.png"
        new_image_path = images_dir / new_image_name
        
        # Copy actual image file if it exists
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, new_image_path)
            print(f"Processing {idx+1}/{len(records)}: {new_image_name}")
        else:
            print(f"Warning: Image file not found: {source_image_path}")
            # Skip this record if image doesn't exist
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
        
        # Split into train/val/test (80/10/10)
        # Use modulo for reproducible splitting
        split_val = xray_id % 10
        if split_val < 8:
            train_data.append(entry)
        elif split_val == 8:
            val_data.append(entry)
        else:
            test_data.append(entry)
    
    # Write JSONL files
    def write_jsonl(data, filename):
        filepath = output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry) + '\n')
        print(f"Wrote {len(data)} entries to {filename}")
    
    write_jsonl(train_data, "train.jsonl")
    write_jsonl(val_data, "val.jsonl")
    write_jsonl(test_data, "test.jsonl")
    
    # Write dataset statistics
    stats_data = {
        "total_records": len(records),
        "train_size": len(train_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
        "age_distribution": stats,
        "export_date": datetime.now().isoformat()
    }
    
    with open(output_path / "dataset_stats.json", 'w') as f:
        json.dumps(stats_data, f, indent=2)
    
    print("\n=== Export Summary ===")
    print(f"Total records: {len(records)}")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")
    print("\nAge distribution:")
    for age_group, count in stats.items():
        print(f"  {age_group}: {count}")
    print(f"\nData exported to: {output_path.absolute()}")
    
    # conn.close()
    
    return output_path

if __name__ == "__main__":
    # For testing, export a small sample first
    export_data(limit=100)
    
    # For full export:
    #export_data()
