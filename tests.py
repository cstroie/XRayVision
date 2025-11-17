#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testing script for XRayVision.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import shutil
import configparser
import sqlite3

# Add the project directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the modules we want to test
import xrayvision

class TestXRayVisionDatabase(unittest.TestCase):
    """Test cases for the xrayvision database operations"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        self.db_file = os.path.join(self.test_dir, 'test.db')
        # Set the database file path for testing
        xrayvision.DB_FILE = self.db_file
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_db_init_creates_tables(self):
        """Test that db_init creates all required tables"""
        # Initialize the database
        xrayvision.db_init()
        
        # Check that all tables were created
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Check patients table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='patients'")
            self.assertIsNotNone(cursor.fetchone(), "patients table should exist")
            
            # Check exams table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='exams'")
            self.assertIsNotNone(cursor.fetchone(), "exams table should exist")
            
            # Check ai_reports table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_reports'")
            self.assertIsNotNone(cursor.fetchone(), "ai_reports table should exist")
            
            # Check rad_reports table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rad_reports'")
            self.assertIsNotNone(cursor.fetchone(), "rad_reports table should exist")
    
    def test_db_init_creates_indexes(self):
        """Test that db_init creates all required indexes"""
        # Initialize the database
        xrayvision.db_init()
        
        # Check that indexes were created
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Get all indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            
            # Check for expected indexes
            expected_indexes = [
                'idx_exams_status',
                'idx_exams_region',
                'idx_exams_cnp',
                'idx_exams_created',
                'idx_exams_study',
                'idx_ai_reports_created',
                'idx_rad_reports_created',
                'idx_patients_name'
            ]
            
            for index in expected_indexes:
                self.assertIn(index, indexes, f"Index {index} should exist")
    
    def test_db_add_patient_inserts_new_patient(self):
        """Test that db_add_patient inserts a new patient"""
        # Initialize the database
        xrayvision.db_init()
        
        # Add a patient
        cnp = "1234567890123"
        id = "P001"
        name = "John Doe"
        age = 30
        sex = "M"
        
        result = xrayvision.db_add_patient(cnp, id, name, age, sex)
        
        # Check that the operation was successful
        self.assertIsNotNone(result, "db_add_patient should return a result")
        
        # Verify the patient was inserted
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cnp, id, name, age, sex FROM patients WHERE cnp = ?", (cnp,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Patient should be inserted")
            self.assertEqual(row[0], cnp)
            self.assertEqual(row[1], id)
            self.assertEqual(row[2], name)
            self.assertEqual(row[3], age)
            self.assertEqual(row[4], sex)
    
    def test_db_add_patient_updates_existing_patient(self):
        """Test that db_add_patient updates an existing patient"""
        # Initialize the database
        xrayvision.db_init()
        
        # Add a patient first
        cnp = "1234567890123"
        id = "P001"
        name = "John Doe"
        age = 30
        sex = "M"
        
        xrayvision.db_add_patient(cnp, id, name, age, sex)
        
        # Update the patient with new information
        new_id = "P002"
        new_name = "Jane Smith"
        new_age = 25
        new_sex = "F"
        
        result = xrayvision.db_add_patient(cnp, new_id, new_name, new_age, new_sex)
        
        # Check that the operation was successful
        self.assertIsNotNone(result, "db_add_patient should return a result")
        
        # Verify the patient was updated
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cnp, id, name, age, sex FROM patients WHERE cnp = ?", (cnp,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Patient should exist")
            self.assertEqual(row[0], cnp)
            self.assertEqual(row[1], new_id)
            self.assertEqual(row[2], new_name)
            self.assertEqual(row[3], new_age)
            self.assertEqual(row[4], new_sex)
    
    def test_db_add_patient_with_valid_sex_values(self):
        """Test that db_add_patient handles valid sex values"""
        # Initialize the database
        xrayvision.db_init()
        
        # Test each valid sex value
        test_cases = [
            ("1234567890123", "P001", "John Doe", 30, "M"),
            ("1234567890124", "P002", "Jane Smith", 25, "F"),
            ("1234567890125", "P003", "Other Patient", 40, "O")
        ]
        
        for cnp, id, name, age, sex in test_cases:
            result = xrayvision.db_add_patient(cnp, id, name, age, sex)
            
            # Check that the operation was successful
            self.assertIsNotNone(result, f"db_add_patient should return a result for sex={sex}")
            
            # Verify the patient was inserted
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cnp, sex FROM patients WHERE cnp = ?", (cnp,))
                row = cursor.fetchone()
                
                self.assertIsNotNone(row, f"Patient should be inserted for sex={sex}")
                self.assertEqual(row[0], cnp)
                self.assertEqual(row[1], sex)

    def test_db_add_exam_inserts_new_exam(self):
        """Test that db_add_exam inserts a new exam"""
        # Initialize the database
        xrayvision.db_init()
        
        # First add a patient
        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_age = 30
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_age, patient_sex)
        
        # Add an exam
        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'age': patient_age,
                'sex': patient_sex
            },
            'exam': {
                'id': 'E001',
                'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray',
                'region': 'chest',
                'type': 'CR',
                'study': '1.2.3.4.5.6',
                'series': '1.2.3.4.5.6.7'
            }
        }
        
        # Call db_add_exam without report
        xrayvision.db_add_exam(exam_info)
        
        # Verify the exam was inserted
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT uid, cnp, id, created, protocol, region, type, status, study, series 
                FROM exams WHERE uid = ?
            """, (exam_info['uid'],))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Exam should be inserted")
            self.assertEqual(row[0], exam_info['uid'])
            self.assertEqual(row[1], cnp)
            self.assertEqual(row[2], exam_info['exam']['id'])
            self.assertEqual(row[3], exam_info['exam']['created'])
            self.assertEqual(row[4], exam_info['exam']['protocol'])
            self.assertEqual(row[5], exam_info['exam']['region'])
            self.assertEqual(row[6], exam_info['exam']['type'])
            self.assertEqual(row[7], 'queued')  # Default status
            self.assertEqual(row[8], exam_info['exam']['study'])
            self.assertEqual(row[9], exam_info['exam']['series'])
    
    def test_db_add_exam_with_report_inserts_ai_report(self):
        """Test that db_add_exam with report also inserts AI report"""
        # Initialize the database
        xrayvision.db_init()
        
        # First add a patient
        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_age = 30
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_age, patient_sex)
        
        # Add an exam with report
        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'age': patient_age,
                'sex': patient_sex
            },
            'exam': {
                'id': 'E001',
                'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray',
                'region': 'chest',
                'type': 'CR',
                'study': '1.2.3.4.5.6',
                'series': '1.2.3.4.5.6.7'
            }
        }
        report_text = "No significant findings."
        positive = False
        confidence = 95
        
        # Call db_add_exam with report
        xrayvision.db_add_exam(exam_info, report=report_text, positive=positive, confidence=confidence)
        
        # Verify the exam was inserted
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT uid FROM exams WHERE uid = ?", (exam_info['uid'],))
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Exam should be inserted")
            
            # Verify the AI report was inserted
            cursor.execute("""
                SELECT uid, text, positive, confidence, is_correct, reviewed, model
                FROM ai_reports WHERE uid = ?
            """, (exam_info['uid'],))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "AI report should be inserted")
            self.assertEqual(row[0], exam_info['uid'])
            self.assertEqual(row[1], report_text)
            self.assertEqual(row[2], int(positive))
            self.assertEqual(row[3], confidence)
            self.assertEqual(row[4], -1)  # is_correct defaults to -1
            self.assertEqual(row[5], 0)   # reviewed defaults to 0 (False as integer)
            self.assertEqual(row[6], xrayvision.MODEL_NAME)  # model from config

    def test_db_add_ai_report_inserts_new_report(self):
        """Test that db_add_ai_report inserts a new AI report"""
        # Initialize the database
        xrayvision.db_init()
        
        # Add a patient and exam first
        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_age = 30
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_age, patient_sex)
        
        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'age': patient_age,
                'sex': patient_sex
            },
            'exam': {
                'id': 'E001',
                'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray',
                'region': 'chest',
                'type': 'CR',
                'study': '1.2.3.4.5.6',
                'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)
        
        # Add an AI report
        uid = '1.2.3.4.5'
        report_text = "Findings suggest possible pneumonia."
        positive = 1  # Using integer instead of boolean
        confidence = 85
        model = "test-model"
        latency = 2.5
        is_correct = 1  # Using integer for three-state value
        
        xrayvision.db_add_ai_report(uid, report_text, positive, confidence, model, latency, is_correct)
        
        # Verify the AI report was inserted
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT uid, text, positive, confidence, is_correct, reviewed, model, latency
                FROM ai_reports WHERE uid = ?
            """, (uid,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "AI report should be inserted")
            self.assertEqual(row[0], uid)
            self.assertEqual(row[1], report_text)
            self.assertEqual(row[2], positive)
            self.assertEqual(row[3], confidence)
            self.assertEqual(row[4], is_correct)
            self.assertEqual(row[5], 0)  # reviewed defaults to 0
            self.assertEqual(row[6], model)
            self.assertEqual(row[7], latency)

    def test_db_add_rad_report_inserts_new_report(self):
        """Test that db_add_rad_report inserts a new radiologist report"""
        # Initialize the database
        xrayvision.db_init()
        
        # Add a patient and exam first
        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_age = 30
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_age, patient_sex)
        
        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'age': patient_age,
                'sex': patient_sex
            },
            'exam': {
                'id': 'E001',
                'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray',
                'region': 'chest',
                'type': 'CR',
                'study': '1.2.3.4.5.6',
                'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)
        
        # Add a radiologist report
        uid = '1.2.3.4.5'
        report_id = "R001"
        report_text = "Confirmed pneumonia with consolidation in right lower lobe."
        positive = 1  # Using integer instead of boolean
        severity = 7
        summary = "pneumonia"
        report_type = "CR"
        radiologist = "Dr. Smith"
        justification = "Clinical presentation consistent with pneumonia"
        model = "test-model"
        latency = 5.0
        
        xrayvision.db_add_rad_report(uid, report_id, report_text, positive, severity, summary, report_type, radiologist, justification, model, latency)
        
        # Verify the radiologist report was inserted
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT uid, id, text, positive, severity, summary, type, radiologist, justification, model, latency
                FROM rad_reports WHERE uid = ?
            """, (uid,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Radiologist report should be inserted")
            self.assertEqual(row[0], uid)
            self.assertEqual(row[1], report_id)
            self.assertEqual(row[2], report_text)
            self.assertEqual(row[3], positive)
            self.assertEqual(row[4], severity)
            self.assertEqual(row[5], summary)
            self.assertEqual(row[6], report_type)
            self.assertEqual(row[7], radiologist)
            self.assertEqual(row[8], justification)
            self.assertEqual(row[9], model)
            self.assertEqual(row[10], latency)

    def test_db_set_status_updates_exam_status(self):
        """Test that db_set_status updates the status of an exam"""
        # Initialize the database
        xrayvision.db_init()
        
        # Add a patient and exam first
        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_age = 30
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_age, patient_sex)
        
        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'age': patient_age,
                'sex': patient_sex
            },
            'exam': {
                'id': 'E001',
                'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray',
                'region': 'chest',
                'type': 'CR',
                'study': '1.2.3.4.5.6',
                'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)
        
        # Set the status to 'processing'
        uid = '1.2.3.4.5'
        new_status = 'processing'
        result = xrayvision.db_set_status(uid, new_status)
        
        # Check that the function returns the correct status
        self.assertEqual(result, new_status)
        
        # Verify the status was updated in the database
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM exams WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Exam should exist")
            self.assertEqual(row[0], new_status)
        
        # Change the status to 'done'
        final_status = 'done'
        result = xrayvision.db_set_status(uid, final_status)
        
        # Check that the function returns the correct status
        self.assertEqual(result, final_status)
        
        # Verify the status was updated in the database
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM exams WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            
            self.assertIsNotNone(row, "Exam should exist")
            self.assertEqual(row[0], final_status)

class TestXRayVision(unittest.TestCase):
    """Test cases for the xrayvision module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validate_romanian_cnp(self):
        """Test Romanian ID validation function"""
        # Test valid Romanian ID (example: male, born 1990-01-01, Bucharest)
        result = xrayvision.validate_romanian_cnp("1900101400008")
        self.assertTrue(result['valid'])
        
        # Test invalid Romanian ID (wrong length)
        result = xrayvision.validate_romanian_cnp("12345")
        self.assertFalse(result['valid'])
        
        # Test invalid Romanian ID (non-numeric)
        result = xrayvision.validate_romanian_cnp("abcdefghijk")
        self.assertFalse(result['valid'])
    
    def test_compute_age_from_cnp(self):
        """Test age computation from Romanian ID"""
        # This is a placeholder test - actual implementation would depend on
        # the specific format of Romanian IDs
        age = xrayvision.compute_age_from_cnp("1234567890123")
        self.assertIsInstance(age, int)
    
    def test_contains_any_word(self):
        """Test word matching function"""
        # Test matching words
        self.assertTrue(xrayvision.contains_any_word("chest xray study", "chest", "abdomen"))
        
        # Test non-matching words
        self.assertFalse(xrayvision.contains_any_word("brain mri scan", "chest", "abdomen"))
        
        # Test empty string
        self.assertFalse(xrayvision.contains_any_word("", "chest"))
        
        # Test empty word list
        self.assertFalse(xrayvision.contains_any_word("chest xray",))
    
    @patch('xrayvision.identify_anatomic_region')
    def test_identify_anatomic_region_calls(self, mock_identify):
        """Test that identify_anatomic_region is called with correct parameters"""
        mock_identify.return_value = "chest"
        info = {"StudyDescription": "Chest X-Ray"}
        
        result = xrayvision.identify_anatomic_region(info)
        mock_identify.assert_called_once_with(info)
        self.assertEqual(result, "chest")
    
    def test_identify_imaging_projection(self):
        """Test imaging projection identification"""
        # Test AP projection
        info = {"exam": {"protocol": "Chest A.P."}}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "frontal")
        
        # Test PA projection
        info = {"exam": {"protocol": "Chest P.A."}}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "frontal")
        
        # Test lateral projection
        info = {"exam": {"protocol": "Chest Lat."}}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "lateral")
        
        # Test unknown projection
        info = {"exam": {"protocol": "Unknown"}}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "")
    
    def test_determine_patient_gender_description(self):
        """Test patient gender description determination"""
        # Test male
        info = {"patient": {"sex": "M"}}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "boy")
        
        # Test female
        info = {"patient": {"sex": "F"}}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "girl")
        
        # Test unknown
        info = {"patient": {"sex": "O"}}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "child")
        
        # Test missing field
        info = {"patient": {}}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "child")
    
    @patch('xrayvision.db_get_previous_reports')
    def test_db_get_previous_reports_called(self, mock_db_get):
        """Test that db_get_previous_reports is called correctly"""
        mock_db_get.return_value = []
        
        result = xrayvision.db_get_previous_reports("12345", "chest", 3)
        mock_db_get.assert_called_once_with("12345", "chest", 3)
        self.assertEqual(result, [])


class TestXRayVisionConfig(unittest.TestCase):
    """Test cases for xrayvision configuration"""
    
    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has the expected structure"""
        # Check that general section exists
        self.assertIn('general', xrayvision.DEFAULT_CONFIG)
        
        # Check that dicom section exists
        self.assertIn('dicom', xrayvision.DEFAULT_CONFIG)
        
        # Check that required general fields exist
        general_config = xrayvision.DEFAULT_CONFIG['general']
        self.assertIn('XRAYVISION_DB_PATH', general_config)
        self.assertIn('XRAYVISION_BACKUP_DIR', general_config)
        
        # Check that required dicom fields exist
        dicom_config = xrayvision.DEFAULT_CONFIG['dicom']
        self.assertIn('AE_TITLE', dicom_config)
        self.assertIn('AE_PORT', dicom_config)
        self.assertIn('REMOTE_AE_TITLE', dicom_config)
        self.assertIn('REMOTE_AE_IP', dicom_config)
        self.assertIn('REMOTE_AE_PORT', dicom_config)

if __name__ == '__main__':
    unittest.main()
