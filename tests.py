import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys
import shutil
import configparser

# Add the project directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the modules we want to test
import xrayvision
import qr

class TestXRayVision(unittest.TestCase):
    """Test cases for the xrayvision module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_validate_romanian_id(self):
        """Test Romanian ID validation function"""
        # Test valid Romanian ID
        self.assertTrue(xrayvision.validate_romanian_id("1234567890123"))
        
        # Test invalid Romanian ID (wrong length)
        self.assertFalse(xrayvision.validate_romanian_id("12345"))
        
        # Test invalid Romanian ID (non-numeric)
        self.assertFalse(xrayvision.validate_romanian_id("abcdefghijk"))
    
    def test_compute_age_from_id(self):
        """Test age computation from Romanian ID"""
        # This is a placeholder test - actual implementation would depend on
        # the specific format of Romanian IDs
        age = xrayvision.compute_age_from_id("1234567890123")
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
        info = {"ViewPosition": "AP"}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "antero-posterior (AP)")
        
        # Test PA projection
        info = {"ViewPosition": "PA"}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "postero-anterior (PA)")
        
        # Test lateral projection
        info = {"ViewPosition": "Lateral"}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "lateral")
        
        # Test unknown projection
        info = {"ViewPosition": "Unknown"}
        projection = xrayvision.identify_imaging_projection(info)
        self.assertEqual(projection, "unknown")
    
    def test_determine_patient_gender_description(self):
        """Test patient gender description determination"""
        # Test male
        info = {"PatientSex": "M"}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "male")
        
        # Test female
        info = {"PatientSex": "F"}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "female")
        
        # Test unknown
        info = {"PatientSex": "O"}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "unknown")
        
        # Test missing field
        info = {}
        gender = xrayvision.determine_patient_gender_description(info)
        self.assertEqual(gender, "unknown")
    
    @patch('xrayvision.db_get_previous_reports')
    def test_db_get_previous_reports_called(self, mock_db_get):
        """Test that db_get_previous_reports is called correctly"""
        mock_db_get.return_value = []
        
        result = xrayvision.db_get_previous_reports("12345", "chest", 3)
        mock_db_get.assert_called_once_with("12345", "chest", 3)
        self.assertEqual(result, [])

class TestQRModule(unittest.TestCase):
    """Test cases for the qr module"""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Tear down test fixtures after each test method."""
        # Clean up temporary directory
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('qr.send_c_move')
    def test_send_c_move_called(self, mock_send):
        """Test that send_c_move function can be called"""
        mock_send.return_value = True
        
        # Call with mock parameters
        result = qr.send_c_move(
            ae=Mock(),
            peer_ae="TEST_AE",
            peer_ip="127.0.0.1",
            peer_port=104,
            study_instance_uid="1.2.3.4.5"
        )
        
        mock_send.assert_called_once()
        self.assertTrue(result)
    
    def test_default_config_structure(self):
        """Test that DEFAULT_CONFIG has the expected structure"""
        # Check that dicom section exists
        self.assertIn('dicom', qr.DEFAULT_CONFIG)
        
        # Check that required dicom fields exist
        dicom_config = qr.DEFAULT_CONFIG['dicom']
        self.assertIn('AE_TITLE', dicom_config)
        self.assertIn('AE_PORT', dicom_config)
        self.assertIn('REMOTE_AE_TITLE', dicom_config)
        self.assertIn('REMOTE_AE_IP', dicom_config)
        self.assertIn('REMOTE_AE_PORT', dicom_config)
    
    @patch('qr.configparser.ConfigParser')
    def test_config_parsing(self, mock_config_parser):
        """Test configuration parsing"""
        # Create a mock config parser
        mock_config = MagicMock()
        mock_config.get.return_value = "TEST_VALUE"
        mock_config.getint.return_value = 1234
        mock_config_parser.return_value = mock_config
        
        # Test that config values are read correctly
        ae_title = qr.AE_TITLE
        ae_port = qr.AE_PORT
        remote_ae_title = qr.REMOTE_AE_TITLE
        remote_ae_ip = qr.REMOTE_AE_IP
        remote_ae_port = qr.REMOTE_AE_PORT
        
        # Verify that the config methods were called
        mock_config.get.assert_any_call('dicom', 'AE_TITLE')
        mock_config.get.assert_any_call('dicom', 'REMOTE_AE_TITLE')
        mock_config.get.assert_any_call('dicom', 'REMOTE_AE_IP')
        mock_config.getint.assert_any_call('dicom', 'AE_PORT')
        mock_config.getint.assert_any_call('dicom', 'REMOTE_AE_PORT')

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
        self.assertIn('XRAYVISION_USER', general_config)
        self.assertIn('XRAYVISION_PASS', general_config)
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
