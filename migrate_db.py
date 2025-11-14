#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database migration script for XRayVision.

Converts old database format (single exams table) to new normalized format
with separate tables for patients, exams, AI reports, and radiologist reports.
"""

import sqlite3
import os
import sys
from datetime import datetime

def migrate_database(old_db_path, new_db_path):
    """
    Migrate database from old format to new format.
    
    Args:
        old_db_path: Path to the old database file
        new_db_path: Path to the new database file
    """
    print(f"Starting database migration from {old_db_path} to {new_db_path}")
    
    if not os.path.exists(old_db_path):
        print(f"Error: Old database file {old_db_path} not found.")
        return False
    
    # Create new database with updated schema
    print("Creating new database schema...")
    create_new_schema(new_db_path)
    
    # Migrate data
    print("Migrating data...")
    try:
        migrated_count = migrate_data(old_db_path, new_db_path)
        print(f"Migration completed successfully! Migrated {migrated_count} records.")
        return True
    except Exception as e:
        print(f"Error during migration: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_new_schema(db_path):
    """Create the new database schema."""
    print("Creating new database schema...")
    with sqlite3.connect(db_path) as conn:
        # Enable foreign key constraints
        conn.execute('PRAGMA foreign_keys = ON')
        
        # Patients table
        conn.execute('''
            CREATE TABLE patients (
                cnp TEXT PRIMARY KEY,
                id TEXT,
                name TEXT,
                age INTEGER,
                sex TEXT CHECK(sex IN ('M', 'F', 'O'))
            )
        ''')
        
        # Exams table
        conn.execute('''
            CREATE TABLE exams (
                uid TEXT PRIMARY KEY,
                cnp TEXT,
                id TEXT,
                created TIMESTAMP,
                protocol TEXT,
                region TEXT,
                type TEXT,
                status TEXT DEFAULT 'none',
                study TEXT,
                series TEXT,
                FOREIGN KEY (cnp) REFERENCES patients(cnp)
            )
        ''')
        
        # AI reports table
        conn.execute('''
            CREATE TABLE ai_reports (
                uid TEXT PRIMARY KEY,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                positive INTEGER DEFAULT -1 CHECK(positive IN (-1, 0, 1)),
                confidence INTEGER DEFAULT -1 CHECK(confidence BETWEEN -1 AND 100),
                is_correct INTEGER DEFAULT -1 CHECK(is_correct IN (-1, 0, 1)),
                model TEXT,
                latency INTEGER DEFAULT -1,
                FOREIGN KEY (uid) REFERENCES exams(uid)
            )
        ''')
        
        # Radiologist reports table
        conn.execute('''
            CREATE TABLE rad_reports (
                uid TEXT PRIMARY KEY,
                id TEXT,
                created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                text TEXT,
                positive INTEGER DEFAULT -1 CHECK(positive IN (-1, 0, 1)),
                severity INTEGER DEFAULT -1 CHECK(severity BETWEEN -1 AND 10),
                summary TEXT,
                type TEXT,
                radiologist TEXT,
                justification TEXT,
                model TEXT,
                latency INTEGER DEFAULT -1,
                FOREIGN KEY (uid) REFERENCES exams(uid)
            )
        ''')
        
        # Indexes for common query filters
        conn.execute('''
            CREATE INDEX idx_exams_status
            ON exams(status)
        ''')
        conn.execute('''
            CREATE INDEX idx_exams_region
            ON exams(region)
        ''')
        conn.execute('''
            CREATE INDEX idx_exams_cnp
            ON exams(cnp)
        ''')
        conn.execute('''
            CREATE INDEX idx_exams_created
            ON exams(created)
        ''')
        conn.execute('''
            CREATE INDEX idx_exams_study
            ON exams(study)
        ''')
        conn.execute('''
            CREATE INDEX idx_ai_reports_created
            ON ai_reports(created)
        ''')
        conn.execute('''
            CREATE INDEX idx_rad_reports_created
            ON rad_reports(created)
        ''')
        conn.execute('''
            CREATE INDEX idx_patients_name
            ON patients(name)
        ''')
    print("New database schema created successfully.")

def migrate_data(old_db_path, new_db_path):
    """Migrate data from old database to new database."""
    old_conn = sqlite3.connect(old_db_path)
    new_conn = sqlite3.connect(new_db_path)
    
    try:
        # Get column information from old database to handle schema variations
        old_cursor = old_conn.execute('PRAGMA table_info(exams)')
        columns = [column[1] for column in old_cursor.fetchall()]
        
        # Build SELECT query based on available columns
        select_columns = []
        # Map old column names to variables
        column_mapping = {
            'uid': 'uid',
            'name': 'patient_name', 
            'patient_name': 'patient_name',
            'id': 'patient_id',
            'patient_id': 'patient_id',
            'age': 'patient_age',
            'patient_age': 'patient_age',
            'sex': 'patient_sex',
            'patient_sex': 'patient_sex',
            'created': 'created',
            'protocol': 'protocol',
            'region': 'region',
            'reported': 'reported',
            'report': 'report',
            'positive': 'positive',
            'valid': 'valid',
            'reviewed': 'reviewed',
            'status': 'status'
        }
        
        # Check which columns exist in the old database
        available_columns = []
        select_list = []
        for old_col, var_name in column_mapping.items():
            if old_col in columns:
                available_columns.append(var_name)
                select_list.append(old_col)
        
        # Get all exams from old database
        select_query = f"SELECT {', '.join(select_list)} FROM exams"
        old_cursor = old_conn.execute(select_query)
        
        migrated_count = 0
        processed_count = 0
        
        for row in old_cursor:
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} records...")
                
            # Create dictionary mapping column names to values
            row_data = dict(zip(available_columns, row))
            
            # Extract values with defaults for missing columns
            uid = row_data.get('uid')
            patient_name = row_data.get('patient_name', '')
            patient_id = row_data.get('patient_id', '')
            patient_age = row_data.get('patient_age', -1)
            patient_sex = row_data.get('patient_sex', 'O')
            created = row_data.get('created')
            protocol = row_data.get('protocol', '')
            region = row_data.get('region', '')
            reported = row_data.get('reported')  # This will map to ai_reports.created
            report = row_data.get('report')
            positive = row_data.get('positive', 0)
            valid = row_data.get('valid', 1)  # valid -> is_correct
            reviewed = row_data.get('reviewed', 0)
            status = row_data.get('status', 'none')
            
            # Validate required fields
            if not uid:
                print(f"Warning: Skipping record with no UID")
                continue
                
            # Use patient_id as CNP for now (may need adjustment based on actual data)
            cnp = patient_id if patient_id else uid
            
            # Add patient record
            # Map old id to cnp, leave new id empty
            new_conn.execute('''
                INSERT OR IGNORE INTO patients (cnp, id, name, age, sex)
                VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, None, patient_name, patient_age, patient_sex))
            
            # Add exam record
            new_conn.execute('''
                INSERT OR REPLACE INTO exams 
                (uid, cnp, id, created, protocol, region, type, status, study, series)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (uid, cnp, None, created, protocol, region, '', status or 'none', None, None))
            
            # Add AI report if it exists
            if report is not None:
                # Map old 'valid' field to new 'is_correct' field
                # In old schema: valid=1 means correct, valid=0 means incorrect
                # In new schema: is_correct=1 means correct, is_correct=0 means incorrect, is_correct=-1 means not assessed
                is_correct = int(valid) if valid is not None else -1
                
                # Use 'reported' timestamp if available, otherwise use current timestamp
                report_created = reported if reported else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                new_conn.execute('''
                    INSERT OR REPLACE INTO ai_reports
                    (uid, created, updated, text, positive, confidence, is_correct, model, latency)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (uid, report_created, report_created, report, 
                      int(positive) if positive is not None else -1, 
                      -1,  # confidence
                      is_correct,  # is_correct (mapped from old 'valid' field)
                      None,  # model
                      -1))  # latency
                
                # Add radiologist report if valid is True (1)
                # Map: uid->uid, id->empty, created->reported, updated->empty, text->empty, 
                # positive->positive from old exams if valid is true, severity->-1, summary->empty,
                # type->empty, justification->empty, radiologist->'Dr. Stroie Costin', model->empty, latency->-1
                if valid == 1:
                    new_conn.execute('''
                        INSERT OR REPLACE INTO rad_reports
                        (uid, id, created, updated, text, positive, severity, summary, type, radiologist, justification, model, latency)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (uid, None, report_created, None, None, 
                          int(positive) if positive is not None else -1,
                          -1, None, None, 'Dr. Stroie Costin', None, None, -1))
            
            migrated_count += 1
        
        new_conn.commit()
        return migrated_count
        
    finally:
        old_conn.close()
        new_conn.close()

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python migrate_db.py <old_db_path> <new_db_path>")
        print("Example: python migrate_db.py xrayvision_old.db xrayvision_new.db")
        sys.exit(1)
    
    old_db_path = sys.argv[1]
    new_db_path = sys.argv[2]
    
    print(f"Migrating database from '{old_db_path}' to '{new_db_path}'")
    
    success = migrate_database(old_db_path, new_db_path)
    if success:
        print(f"Database migrated successfully to {new_db_path}")
    else:
        print("Database migration failed!")
        sys.exit(1)
