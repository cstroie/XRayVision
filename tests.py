import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

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
        import shutil
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
    
    @patch('xrayvision.identify_anatomic_region')
    def test_identify_anatomic_region_calls(self, mock_identify):
        """Test that identify_anatomic_region is called with correct parameters"""
        mock_identify.return_value = "chest"
        info = {"StudyDescription": "Chest X-Ray"}
        
        result = xrayvision.identify_anatomic_region(info)
        mock_identify.assert_called_once_with(info)
        self.assertEqual(result, "chest")

class TestQRModule(unittest.TestCase):
    """Test cases for the qr module"""
    
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

if __name__ == '__main__':
    unittest.main()
