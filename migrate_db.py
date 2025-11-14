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
    if not os.path.exists(old_db_path):
        print(f"Error: Old database file {old_db_path} not found.")
        return False
    
    # Create new database with updated schema
    print("Creating new database schema...")
    create_new_schema(new_db_path)
    
    # Migrate data
    print("Migrating data...")
    try:
        migrate_data(old_db_path, new_db_path)
        print("Migration completed successfully!")
        return True
    except Exception as e:
        print(f"Error during migration: {e}")
        return False

def create_new_schema(db_path):
    """Create the new database schema."""
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

def migrate_data(old_db_path, new_db_path):
    """Migrate data from old database to new database."""
    old_conn = sqlite3.connect(old_db_path)
    new_conn = sqlite3.connect(new_db_path)
    
    try:
        # Get all exams from old database
        old_cursor = old_conn.execute('''
            SELECT uid, patient_name, patient_id, patient_age, patient_sex,
                   created, protocol, region, report, positive, status
            FROM exams
        ''')
        
        migrated_count = 0
        for row in old_cursor:
            uid, patient_name, patient_id, patient_age, patient_sex, created, protocol, region, report, positive, status = row
            
            # Use patient_id as CNP for now (may need adjustment based on actual data)
            cnp = patient_id if patient_id else uid
            
            # Add patient record
            new_conn.execute('''
                INSERT OR IGNORE INTO patients (cnp, id, name, age, sex)
                VALUES (?, ?, ?, ?, ?)
            ''', (cnp, patient_id, patient_name, patient_age, patient_sex))
            
            # Add exam record
            new_conn.execute('''
                INSERT OR REPLACE INTO exams 
                (uid, cnp, created, protocol, region, type, status, study, series)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (uid, cnp, created, protocol, region, '', status or 'none', None, None))
            
            # Add AI report if it exists
            if report is not None:
                new_conn.execute('''
                    INSERT OR REPLACE INTO ai_reports
                    (uid, text, positive, confidence, is_correct, model, latency)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (uid, report, int(positive) if positive is not None else -1, -1, -1, None, -1))
            
            migrated_count += 1
        
        new_conn.commit()
        print(f"Migrated {migrated_count} exams successfully.")
        
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
    
    success = migrate_database(old_db_path, new_db_path)
    if success:
        print(f"Database migrated successfully to {new_db_path}")
    else:
        print("Database migration failed!")
        sys.exit(1)
