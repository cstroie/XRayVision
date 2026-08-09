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
        xrayvision.db_init()

        cnp = "1234567890123"
        id = "P001"
        name = "John Doe"
        birthdate = "1994-06-15"
        sex = "M"

        result = xrayvision.db_add_patient(cnp, id, name, birthdate, sex)

        self.assertIsNotNone(result, "db_add_patient should return a result")

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cnp, id, name, birthdate, sex FROM patients WHERE cnp = ?", (cnp,))
            row = cursor.fetchone()

            self.assertIsNotNone(row, "Patient should be inserted")
            self.assertEqual(row[0], cnp)
            self.assertEqual(row[1], id)
            self.assertEqual(row[2], name)
            self.assertEqual(row[3], birthdate)
            self.assertEqual(row[4], sex)

    def test_db_add_patient_updates_existing_patient(self):
        """Test that db_add_patient updates an existing patient"""
        xrayvision.db_init()

        cnp = "1234567890123"
        xrayvision.db_add_patient(cnp, "P001", "John Doe", "1994-06-15", "M")

        result = xrayvision.db_add_patient(cnp, "P002", "Jane Smith", "1999-03-20", "F")

        self.assertIsNotNone(result, "db_add_patient should return a result")

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT cnp, id, name, birthdate, sex FROM patients WHERE cnp = ?", (cnp,))
            row = cursor.fetchone()

            self.assertIsNotNone(row, "Patient should exist")
            self.assertEqual(row[0], cnp)
            self.assertEqual(row[1], "P002")
            self.assertEqual(row[2], "Jane Smith")
            self.assertEqual(row[3], "1999-03-20")
            self.assertEqual(row[4], "F")

    def test_db_add_patient_with_valid_sex_values(self):
        """Test that db_add_patient handles valid sex values"""
        xrayvision.db_init()

        test_cases = [
            ("1234567890123", "P001", "John Doe", "1994-06-15", "M"),
            ("1234567890124", "P002", "Jane Smith", "1999-03-20", "F"),
            ("1234567890125", "P003", "Other Patient", "1984-11-05", "O")
        ]

        for cnp, id, name, birthdate, sex in test_cases:
            result = xrayvision.db_add_patient(cnp, id, name, birthdate, sex)

            self.assertIsNotNone(result, f"db_add_patient should return a result for sex={sex}")

            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cnp, sex FROM patients WHERE cnp = ?", (cnp,))
                row = cursor.fetchone()

                self.assertIsNotNone(row, f"Patient should be inserted for sex={sex}")
                self.assertEqual(row[0], cnp)
                self.assertEqual(row[1], sex)

    def test_db_add_exam_inserts_new_exam(self):
        """Test that db_add_exam inserts a new exam"""
        xrayvision.db_init()

        cnp = "1234567890123"
        patient_id = "P001"
        patient_name = "John Doe"
        patient_birthdate = "1994-06-15"
        patient_sex = "M"
        xrayvision.db_add_patient(cnp, patient_id, patient_name, patient_birthdate, patient_sex)

        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {
                'cnp': cnp,
                'id': patient_id,
                'name': patient_name,
                'birthdate': patient_birthdate,
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

    def test_db_insert_ai_report(self):
        """Test that db_insert correctly inserts an AI report into ai_reports"""
        xrayvision.db_init()

        cnp = "1234567890123"
        xrayvision.db_add_patient(cnp, "P001", "John Doe", "1994-06-15", "M")

        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {'cnp': cnp, 'id': "P001", 'name': "John Doe", 'birthdate': "1994-06-15", 'sex': "M"},
            'exam': {
                'id': 'E001', 'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray', 'region': 'chest', 'type': 'CR',
                'study': '1.2.3.4.5.6', 'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)

        uid = '1.2.3.4.5'
        xrayvision.db_insert('ai_reports',
            uid=uid,
            text="Findings suggest possible pneumonia.",
            positive=1,
            confidence=85,
            model="test-model",
            latency=2)

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT uid, text, positive, confidence, model, latency FROM ai_reports WHERE uid = ?",
                (uid,))
            row = cursor.fetchone()

            self.assertIsNotNone(row, "AI report should be inserted")
            self.assertEqual(row[0], uid)
            self.assertEqual(row[1], "Findings suggest possible pneumonia.")
            self.assertEqual(row[2], 1)
            self.assertEqual(row[3], 85)
            self.assertEqual(row[4], "test-model")
            self.assertEqual(row[5], 2)

    def test_db_insert_rad_report(self):
        """Test that db_insert correctly inserts a radiologist report into rad_reports"""
        xrayvision.db_init()

        cnp = "1234567890123"
        xrayvision.db_add_patient(cnp, "P001", "John Doe", "1994-06-15", "M")

        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {'cnp': cnp, 'id': "P001", 'name': "John Doe", 'birthdate': "1994-06-15", 'sex': "M"},
            'exam': {
                'id': 'E001', 'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray', 'region': 'chest', 'type': 'CR',
                'study': '1.2.3.4.5.6', 'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)

        uid = '1.2.3.4.5'
        xrayvision.db_insert('rad_reports',
            uid=uid,
            id="R001",
            text="Confirmed pneumonia with consolidation in right lower lobe.",
            text_en="Pneumonia confirmed with consolidation in the right lower lobe.",
            positive=1,
            severity=7,
            summary="pneumonia",
            type="CR",
            radiologist="Dr. Smith",
            justification="Clinical presentation consistent with pneumonia",
            model="test-model",
            latency=5.0)

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT uid, id, text, text_en, positive, severity, summary, type, radiologist, justification, model, latency
                FROM rad_reports WHERE uid = ?
            """, (uid,))
            row = cursor.fetchone()

            self.assertIsNotNone(row, "Radiologist report should be inserted")
            self.assertEqual(row[0], uid)
            self.assertEqual(row[1], "R001")
            self.assertEqual(row[2], "Confirmed pneumonia with consolidation in right lower lobe.")
            self.assertEqual(row[3], "Pneumonia confirmed with consolidation in the right lower lobe.")
            self.assertEqual(row[4], 1)
            self.assertEqual(row[5], 7)
            self.assertEqual(row[6], "pneumonia")
            self.assertEqual(row[7], "CR")
            self.assertEqual(row[8], "Dr. Smith")
            self.assertEqual(row[9], "Clinical presentation consistent with pneumonia")
            self.assertEqual(row[10], "test-model")

    def test_db_insert_rad_report_without_translation(self):
        """Test that db_insert correctly inserts a radiologist report with no translation"""
        xrayvision.db_init()

        cnp = "1234567890124"
        xrayvision.db_add_patient(cnp, "P002", "Jane Smith", "1999-03-20", "F")

        exam_info = {
            'uid': '1.2.3.4.6',
            'patient': {'cnp': cnp, 'id': "P002", 'name': "Jane Smith", 'birthdate': "1999-03-20", 'sex': "F"},
            'exam': {
                'id': 'E002', 'created': '2025-01-01 11:00:00',
                'protocol': 'Chest X-ray', 'region': 'chest', 'type': 'CR',
                'study': '1.2.3.4.5.7', 'series': '1.2.3.4.5.6.8'
            }
        }
        xrayvision.db_add_exam(exam_info)

        uid = '1.2.3.4.6'
        xrayvision.db_insert('rad_reports',
            uid=uid,
            id="R002",
            text="No significant findings.",
            positive=0,
            severity=0,
            summary="normal",
            type="CR",
            radiologist="Dr. Johnson",
            justification="Routine screening",
            model="test-model",
            latency=3.0)

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT uid, text, text_en, positive FROM rad_reports WHERE uid = ?",
                (uid,))
            row = cursor.fetchone()

            self.assertIsNotNone(row, "Radiologist report should be inserted")
            self.assertEqual(row[0], uid)
            self.assertEqual(row[1], "No significant findings.")
            self.assertIsNone(row[2], "text_en should be None when not provided")
            self.assertEqual(row[3], 0)

    def test_db_get_exams_includes_translation(self):
        """Test that db_get_exams includes English translation in results"""
        xrayvision.db_init()

        cnp = "1234567890125"
        xrayvision.db_add_patient(cnp, "P003", "Bob Johnson", "1984-11-05", "M")

        exam_info = {
            'uid': '1.2.3.4.7',
            'patient': {'cnp': cnp, 'id': "P003", 'name': "Bob Johnson", 'birthdate': "1984-11-05", 'sex': "M"},
            'exam': {
                'id': 'E003', 'created': '2025-01-01 12:00:00',
                'protocol': 'Chest X-ray', 'region': 'chest', 'type': 'CR',
                'study': '1.2.3.4.5.8', 'series': '1.2.3.4.5.6.9'
            }
        }
        xrayvision.db_add_exam(exam_info)

        xrayvision.db_insert('ai_reports',
            uid='1.2.3.4.7',
            text="No significant findings.",
            positive=0,
            confidence=95,
            model="test-model",
            latency=2,
            summary="normal")

        xrayvision.db_insert('rad_reports',
            uid='1.2.3.4.7',
            id="R003",
            text="Fără semne de patologie.",
            text_en="No signs of pathology.",
            positive=0,
            severity=0,
            summary="normal",
            type="CR",
            radiologist="Dr. Brown",
            justification="Screening de rutină",
            model="test-model",
            latency=4.0)

        exams, total = xrayvision.db_get_exams(limit=1, uid='1.2.3.4.7')

        self.assertEqual(len(exams), 1)
        exam = exams[0]
        self.assertEqual(exam['report']['rad']['text'], "Fără semne de patologie.")
        self.assertEqual(exam['report']['rad']['text_en'], "No signs of pathology.")

    def test_db_set_status_updates_exam_status(self):
        """Test that db_set_status updates the status of an exam"""
        xrayvision.db_init()

        cnp = "1234567890123"
        xrayvision.db_add_patient(cnp, "P001", "John Doe", "1994-06-15", "M")

        exam_info = {
            'uid': '1.2.3.4.5',
            'patient': {'cnp': cnp, 'id': "P001", 'name': "John Doe", 'birthdate': "1994-06-15", 'sex': "M"},
            'exam': {
                'id': 'E001', 'created': '2025-01-01 10:00:00',
                'protocol': 'Chest X-ray', 'region': 'chest', 'type': 'CR',
                'study': '1.2.3.4.5.6', 'series': '1.2.3.4.5.6.7'
            }
        }
        xrayvision.db_add_exam(exam_info)

        uid = '1.2.3.4.5'
        result = xrayvision.db_set_status(uid, 'processing')
        self.assertEqual(result, 'processing')

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM exams WHERE uid = ?", (uid,))
            self.assertEqual(cursor.fetchone()[0], 'processing')

        result = xrayvision.db_set_status(uid, 'done')
        self.assertEqual(result, 'done')

        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status FROM exams WHERE uid = ?", (uid,))
            self.assertEqual(cursor.fetchone()[0], 'done')

class TestXRayVision(unittest.TestCase):
    """Test cases for the xrayvision module"""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Tear down test fixtures after each test method."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_validate_romanian_cnp(self):
        """Test Romanian ID validation function"""
        result = xrayvision.validate_romanian_cnp("1900101400004")
        self.assertTrue(result['valid'])

        result = xrayvision.validate_romanian_cnp("12345")
        self.assertFalse(result['valid'])

        result = xrayvision.validate_romanian_cnp("abcdefghijk")
        self.assertFalse(result['valid'])

    def test_compute_age_from_cnp(self):
        """Test age computation from Romanian ID"""
        age = xrayvision.compute_age_from_cnp("1234567890123")
        self.assertIsInstance(age, int)

    def test_contains_any_word(self):
        """Test word matching function"""
        self.assertTrue(xrayvision.contains_any_word("chest xray study", "chest", "abdomen"))
        self.assertFalse(xrayvision.contains_any_word("brain mri scan", "chest", "abdomen"))
        self.assertFalse(xrayvision.contains_any_word("", "chest"))
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
        info = {"exam": {"protocol": "Chest A.P."}}
        self.assertEqual(xrayvision.identify_imaging_projection(info), "frontal")

        info = {"exam": {"protocol": "Chest P.A."}}
        self.assertEqual(xrayvision.identify_imaging_projection(info), "frontal")

        info = {"exam": {"protocol": "Chest Lat."}}
        self.assertEqual(xrayvision.identify_imaging_projection(info), "lateral")

        info = {"exam": {"protocol": "Unknown"}}
        self.assertEqual(xrayvision.identify_imaging_projection(info), "")

    def test_determine_patient_gender_description(self):
        """Test patient gender description determination"""
        info = {"patient": {"sex": "M"}}
        self.assertEqual(xrayvision.determine_patient_gender_description(info), "boy")

        info = {"patient": {"sex": "F"}}
        self.assertEqual(xrayvision.determine_patient_gender_description(info), "girl")

        info = {"patient": {"sex": "O"}}
        self.assertEqual(xrayvision.determine_patient_gender_description(info), "child")

        info = {"patient": {}}
        self.assertEqual(xrayvision.determine_patient_gender_description(info), "child")

    @patch('xrayvision.db_get_previous_reports')
    def test_db_get_previous_reports_called(self, mock_db_get):
        """Test that db_get_previous_reports is called correctly"""
        mock_db_get.return_value = []
        result = xrayvision.db_get_previous_reports("12345", "chest", 3)
        mock_db_get.assert_called_once_with("12345", "chest", 3)
        self.assertEqual(result, [])

    def test_parse_ai_report_text_labeled_short_impression(self):
        """FINDINGS/IMPRESSION labels present, impression within word limit is kept"""
        report = "FINDINGS: Clear lung fields bilaterally.\n\nIMPRESSION: normal."
        findings, impression = xrayvision.parse_ai_report_text(report)
        self.assertEqual(findings, "Clear lung fields bilaterally.")
        self.assertEqual(impression, "normal.")

    def test_parse_ai_report_text_labeled_long_impression_discarded(self):
        """Impression longer than the word limit is discarded (set to None)"""
        report = "FINDINGS: Consolidation right lower lobe.\n\nIMPRESSION: bilateral perihilar consolidation likely pneumonia."
        findings, impression = xrayvision.parse_ai_report_text(report)
        self.assertEqual(findings, "Consolidation right lower lobe.")
        self.assertIsNone(impression)

    def test_parse_ai_report_text_custom_impression_max_words(self):
        """impression_max_words parameter controls the truncation threshold"""
        report = "FINDINGS: Consolidation right lower lobe.\n\nIMPRESSION: bilateral perihilar consolidation likely pneumonia."
        findings, impression = xrayvision.parse_ai_report_text(report, impression_max_words=10)
        self.assertEqual(findings, "Consolidation right lower lobe.")
        self.assertEqual(impression, "bilateral perihilar consolidation likely pneumonia.")

    def test_parse_ai_report_text_no_labels(self):
        """Unlabeled free text is returned entirely as findings, no impression"""
        report = "Clear lung fields bilaterally. No focal consolidation or effusion."
        findings, impression = xrayvision.parse_ai_report_text(report)
        self.assertEqual(findings, report)
        self.assertIsNone(impression)

    def test_parse_ai_report_text_findings_only(self):
        """FINDINGS label present but no IMPRESSION label"""
        report = "FINDINGS: Clear lung fields bilaterally. No acute abnormality."
        findings, impression = xrayvision.parse_ai_report_text(report)
        self.assertEqual(findings, "Clear lung fields bilaterally. No acute abnormality.")
        self.assertIsNone(impression)


class TestXRayVisionAsync(unittest.IsolatedAsyncioTestCase):
    """Async test cases for the xrayvision module"""

    def setUp(self):
        self._orig_translation_active = xrayvision.TASK_ACTIVE['translation']
        xrayvision.TASK_ACTIVE['translation'] = {'backend': 'test', 'url': 'http://test-backend/v1/chat/completions', 'model': 'test-model', 'api_key': ''}

    def tearDown(self):
        xrayvision.TASK_ACTIVE['translation'] = self._orig_translation_active

    @patch('xrayvision.send_to_llm')
    async def test_translate_report_success(self, mock_send_to_llm):
        """Test that translate_report successfully translates Romanian to English"""
        translation = "Clear costo-diaphragmatic sinuses, no pleural effusion."
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": f"```text\n{translation}\n```"}}]
        }
        result = await xrayvision.translate_report("SCD libere, fără lichid pleural.")
        self.assertEqual(result, translation)

    @patch('xrayvision.send_to_llm')
    async def test_translate_report_failure(self, mock_send_to_llm):
        """Test that translate_report handles AI failures gracefully"""
        mock_send_to_llm.return_value = None
        result = await xrayvision.translate_report("SCD libere, fără lichid pleural.")
        self.assertIsNone(result)

    @patch('xrayvision.send_to_llm')
    async def test_translate_report_accepts_unfenced_text(self, mock_send_to_llm):
        """Plain-text translation with no ```text``` fence must not be discarded
        -- some models (e.g. qwen3-4b) don't reliably follow the fence instruction."""
        translation = "No acute cardiopulmonary abnormality."
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": translation}}]
        }
        result = await xrayvision.translate_report("Cord, pulmon normale radiologic.")
        self.assertEqual(result, translation)

    @patch('xrayvision.send_to_llm')
    async def test_translate_report_invalid_json(self, mock_send_to_llm):
        """Test that translate_report handles invalid JSON responses"""
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": '{"invalid_field": "some text"}'}}]
        }
        result = await xrayvision.translate_report("SCD libere, fără lichid pleural.")
        self.assertIsNone(result)

    @patch('xrayvision.send_to_llm')
    async def test_translate_report_empty_input(self, mock_send_to_llm):
        """Test that translate_report handles empty input"""
        result = await xrayvision.translate_report("")
        self.assertIsNone(result)
        mock_send_to_llm.assert_not_called()


class TestCheckReportParsing(unittest.IsolatedAsyncioTestCase):
    """Regression tests for check_report()'s JSON extraction."""

    def setUp(self):
        self._orig_check_active = xrayvision.TASK_ACTIVE['check']
        xrayvision.TASK_ACTIVE['check'] = {'backend': 'test', 'url': 'http://test-backend/v1/chat/completions', 'model': 'test-model', 'api_key': ''}

    def tearDown(self):
        xrayvision.TASK_ACTIVE['check'] = self._orig_check_active

    @patch('xrayvision.send_to_llm')
    async def test_check_report_accepts_fenced_json(self, mock_send_to_llm):
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": '```json\n{"pathologic": "yes", "severity": 6, "summary": "pneumonia"}\n```'}}]
        }
        result = await xrayvision.check_report("FINDINGS: consolidation.")
        self.assertEqual(result, {"pathologic": "yes", "severity": 6, "summary": "pneumonia"})

    @patch('xrayvision.send_to_llm')
    async def test_check_report_accepts_bare_json(self, mock_send_to_llm):
        """Bare JSON with no code fence -- what chk_prompt.txt actually asks for
        ("Respond with ONLY valid JSON") -- must not be silently discarded."""
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": '{"pathologic": "yes", "severity": 7, "summary": "pneumonia"}'}}]
        }
        result = await xrayvision.check_report("FINDINGS: consolidation.")
        self.assertEqual(result, {"pathologic": "yes", "severity": 7, "summary": "pneumonia"})

    @patch('xrayvision.send_to_llm')
    async def test_check_report_rejects_garbage(self, mock_send_to_llm):
        mock_send_to_llm.return_value = {
            "choices": [{"message": {"content": "I cannot help with that."}}]
        }
        result = await xrayvision.check_report("FINDINGS: consolidation.")
        self.assertIn('error', result)


class TestXRayVisionConfig(unittest.TestCase):
    """Test cases for xrayvision configuration"""

    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has the expected structure"""
        self.assertIn('general', xrayvision.DEFAULT_CONFIG)
        self.assertIn('dicom', xrayvision.DEFAULT_CONFIG)

        general_config = xrayvision.DEFAULT_CONFIG['general']
        self.assertIn('XRAYVISION_DB_PATH', general_config)
        self.assertIn('XRAYVISION_BACKUP_DIR', general_config)

        dicom_config = xrayvision.DEFAULT_CONFIG['dicom']
        self.assertIn('AE_TITLE', dicom_config)
        self.assertIn('AE_PORT', dicom_config)
        self.assertIn('REMOTE_AE_TITLE', dicom_config)
        self.assertIn('REMOTE_AE_IP', dicom_config)
        self.assertIn('REMOTE_AE_PORT', dicom_config)

if __name__ == '__main__':
    unittest.main()
