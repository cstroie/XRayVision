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

# Add the current directory to Python path to import xrayvision
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import required functions directly to avoid DB_FILE conflicts
from xrayvision import db_init, db_create_insert_query, db_execute_query_retry

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
    # Temporarily override DB_FILE to use the new database path
    original_db_file = os.environ.get('XRAYVISION_DB_PATH', 'xrayvision.db')
    os.environ['XRAYVISION_DB_PATH'] = new_db_path
    try:
        db_init()
        print("New database schema created successfully.")
    finally:
        # Restore original DB_FILE
        os.environ['XRAYVISION_DB_PATH'] = original_db_file
    
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

        print("Available columns in old database:", select_query)
        
        for row in old_cursor:
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} records...")
                
            # Create dictionary mapping column names to values
            row_data = dict(zip(select_list, row))
            
            # Extract values with defaults for missing columns
            uid = row_data.get('uid')
            patient_name = row_data.get('name', '')
            patient_id = row_data.get('id', '')
            patient_age = row_data.get('age', -1)
            patient_sex = row_data.get('sex', 'O')
            created = row_data.get('created')
            protocol = row_data.get('protocol', '')
            region = row_data.get('region', '')
            reported = row_data.get('reported')  # This will map to ai_reports.created
            report = row_data.get('report')
            positive = row_data.get('positive', -1)
            valid = row_data.get('valid', -1)  # valid -> is_correct
            reviewed = row_data.get('reviewed', -1)
            status = row_data.get('status', 'none')
            
            # Validate required fields
            if not uid:
                print(f"Warning: Skipping record with no UID")
                continue
                
            # Validate required fields
            if not patient_id:
                print(f"Warning: Skipping record with no CNP")
                continue
            
            # Add patient record
            patient_query = db_create_insert_query('patients', 'cnp', 'id', 'name', 'age', 'sex')
            patient_params = (patient_id, "", patient_name, patient_age, patient_sex)
            new_conn.execute(patient_query, patient_params)
            
            # Add exam record
            exam_query = db_create_insert_query('exams', 'uid', 'cnp', 'id', 'created', 'protocol', 'region', 'type', 'status', 'study', 'series')
            exam_params = (uid, patient_id, "", created, protocol, region, "", status or 'none', "", "")
            new_conn.execute(exam_query, exam_params)
            
            # Add AI report if it exists
            if report:
                # Use 'reported' timestamp if available, otherwise use current timestamp
                report_created = reported if reported else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                ai_query = db_create_insert_query('ai_reports', 'uid', 'created', 'updated', 'text', 'positive', 'confidence', 'model', 'latency')
                ai_params = (uid, created, report_created, report, 
                            int(positive) if positive is not None else -1, 
                            -1,  # confidence
                            "medgemma-4b-it",  # model
                            -1)  # latency
                new_conn.execute(ai_query, ai_params)

                if reviewed == 1:
                    rad = 'rad'
                    if valid == 1:
                        rad_positive = positive
                    else:
                        if positive == 1:
                            rad_positive = 0
                        else:
                            rad_positive = 1
                else:
                    rad = ''
                    rad_positive = -1
                
                # Add radiologist report
                rad_query = db_create_insert_query('rad_reports', 'uid', 'id', 'created', 'updated', 'text', 'positive', 'severity', 'summary', 'type', 'radiologist', 'justification', 'model', 'latency')
                rad_params = (uid, "", report_created, report_created, "", 
                             rad_positive,
                             -1, "", "", rad, "", "", -1)
                new_conn.execute(rad_query, rad_params)
            
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
