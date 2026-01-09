import os
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

def calculate_age_group(age_days):
    """Calculate age group based on days."""
    if age_days <= 28:  # 0-4 weeks
        return "neonate"
    elif age_days <= 730:  # 0-2 years
        return "toddler"
    elif age_days <= 1825:  # 3-5 years
        return "preschool"
    elif age_days <= 3650:  # 6-10 years
        return "school_age"
    elif age_days <= 5475:  # 11-15 years
        return "adolescent"
    else:
        return "unknown"

def connect_to_database(db_path):
    """Connect to the SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")
    return sqlite3.connect(db_path)

def query_records(conn, limit=None, region=None, age_group=None):
    """Query records from the database with optional filters."""
    cursor = conn.cursor()
    
    # Base query
    query = """
    SELECT 
        p.cnp, p.name, p.birthdate, p.sex,
        e.uid, e.datetime, e.region, e.projection, e.modality, e.report_text, e.ai_report_text,
        e.radiologist_report_text, e.radiologist_name, e.status, e.image_path
    FROM patients p
    JOIN exams e ON p.cnp = e.patient_cnp
    WHERE e.image_path IS NOT NULL 
    AND e.status IN ('processed', 'reviewed')
    """
    
    params = []
    
    if region:
        query += " AND e.region = ?"
        params.append(region)
    
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query, params)
    return cursor.fetchall()

def process_record(record, images_source_dir, stats, processed_count, skipped_count):
    """Process a single record and prepare it for export."""
    (cnp, name, birthdate, sex, uid, datetime_str, region, projection, modality, 
     report_text, ai_report_text, rad_report_text, rad_name, status, image_path) = record
    
    # Skip if no image path
    if not image_path:
        skipped_count['value'] += 1
        return None
    
    # Calculate age
    try:
        birth_date = datetime.strptime(birthdate, '%Y-%m-%d')
        exam_date = datetime.strptime(datetime_str.split()[0], '%Y-%m-%d')
        age_days = (exam_date - birth_date).days
        age_group = calculate_age_group(age_days)
    except:
        age_group = "unknown"
    
    # Determine which report to use
    report = rad_report_text or ai_report_text or report_text
    if not report:
        skipped_count['value'] += 1
        return None
    
    # Update stats
    stats['total'] += 1
    if region not in stats['regions']:
        stats['regions'][region] = 0
    stats['regions'][region] += 1
    
    # Prepare metadata
    metadata = {
        "file_name": os.path.basename(image_path),  # Changed from 'image' to 'file_name'
        "cnp": cnp,
        "name": name,
        "birthdate": birthdate,
        "sex": sex,
        "age_days": age_days,
        "age_group": age_group,
        "exam_uid": uid,
        "datetime": datetime_str,
        "region": region,
        "projection": projection,
        "modality": modality,
        "report": report,
        "radiologist": rad_name,
        "status": status
    }
    
    processed_count['value'] += 1
    return metadata, image_path

def write_jsonl(data, filename, output_path):
    """Write data to a JSONL file."""
    filepath = os.path.join(output_path, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def print_summary(records, processed_count, skipped_count, train_data, val_data, test_data, stats, 
                  output_dir):
    """Print export summary."""
    print("\n" + "="*60)
    print("EXPORT SUMMARY")
    print("="*60)
    print(f"Total records found: {len(records)}")
    print(f"Successfully processed: {processed_count['value']}")
    print(f"Skipped: {skipped_count['value']}")
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"\nRegion distribution:")
    for region, count in stats['regions'].items():
        print(f"  {region}: {count} samples")
    print(f"\nOutput directory: {output_dir}")
    print("="*60 + "\n")

def export_data(output_dir="./export/pediatric_xray_dataset", limit=None, 
                db_path="./export/xrayvision.db", images_source_dir="./images",
                train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    """Export dataset in image_dataset format."""
    
    # Validate ratios
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.01:
        raise ValueError("Train, val, and test ratios must sum to 1.0")
    
    # Create output directories
    output_path = Path(output_dir)
    train_dir = output_path / "train"
    val_dir = output_path / "validate"
    test_dir = output_path / "test"
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
    conn = connect_to_database(db_path)
    
    # Query records
    records = query_records(conn, limit=limit)
    
    # Process records
    all_metadata = []
    stats = {'total': 0, 'regions': {}}
    processed_count = {'value': 0}
    skipped_count = {'value': 0}
    
    for record in records:
        result = process_record(record, images_source_dir, stats, 
                               processed_count, skipped_count)
        if result:
            metadata, image_path = result
            all_metadata.append((metadata, image_path))
    
    # Split data
    import random
    random.shuffle(all_metadata)
    
    n_total = len(all_metadata)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_data = all_metadata[:n_train]
    val_data = all_metadata[n_train:n_train + n_val]
    test_data = all_metadata[n_train + n_val:]
    
    # Copy images and write metadata for each split
    for split_name, split_data, split_dir in [
        ("train.jsonl", train_data, train_dir),
        ("val.jsonl", val_data, val_dir),
        ("test.jsonl", test_data, test_dir)
    ]:
        if split_data:
            # Extract just the metadata for the JSONL file
            metadata_only = [item[0] for item in split_data]
            write_jsonl(metadata_only, "metadata.jsonl", split_dir)
            
            # Copy images
            for metadata, image_path in split_data:
                src = os.path.join(images_source_dir, image_path)
                dst = split_dir / metadata["file_name"]
                if os.path.exists(src):
                    import shutil
                    shutil.copy2(src, dst)
    
    conn.close()
    print_summary(records, processed_count, skipped_count, 
                 train_data, val_data, test_data, stats, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dataset in image_dataset format")
    parser.add_argument("--output_dir", default="./export/pediatric_xray_dataset", 
                       help="Output directory for the dataset")
    parser.add_argument("--limit", type=int, help="Limit number of records to export")
    parser.add_argument("--db_path", default="./export/xrayvision.db", 
                       help="Path to the SQLite database")
    parser.add_argument("--images_source_dir", default="./images", 
                       help="Directory containing source images")
    parser.add_argument("--train_ratio", type=float, default=0.7, 
                       help="Training set ratio")
    parser.add_argument("--val_ratio", type=float, default=0.2, 
                       help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, 
                       help="Test set ratio")
    
    args = parser.parse_args()
    
    export_data(
        output_dir=args.output_dir,
        limit=args.limit,
        db_path=args.db_path,
        images_source_dir=args.images_source_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
