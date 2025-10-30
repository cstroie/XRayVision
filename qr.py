#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# qr.py - DICOM Query/Retrieve utility for XRayVision
# 
# This script queries a remote DICOM PACS for CR (Computed Radiography) studies
# for a specified date range and requests them to be sent to the local AE.
# It's designed to work with the XRayVision system configuration.

import argparse
import logging
import time
from datetime import datetime, timedelta
import configparser
import os

from pynetdicom import AE, QueryRetrievePresentationContexts
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove
from pydicom.dataset import Dataset

# Logger config
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s | %(levelname)8s | %(message)s',
    handlers = [
        #logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)

# Default configuration values
DEFAULT_CONFIG = {
    'dicom': {
        'AE_TITLE': 'XRAYVISION',
        'AE_PORT': '4010',
        'REMOTE_AE_TITLE': 'DICOM_SERVER',
        'REMOTE_AE_IP': '192.168.1.1',
        'REMOTE_AE_PORT': '104'
    }
}

# Load configuration from file if it exists, otherwise use defaults
config = configparser.ConfigParser()
config.read_dict(DEFAULT_CONFIG)
try:
    config.read('xrayvision.cfg')
    logging.info("Configuration loaded from xrayvision.cfg")
    # Check for local configuration file to override settings
    local_config_files = config.read('local.cfg')
    if local_config_files:
        logging.info("Local configuration loaded from local.cfg")
except Exception as e:
    logging.info("Using default configuration values")

# Extract configuration values
AE_TITLE = config.get('dicom', 'AE_TITLE')
AE_PORT = config.getint('dicom', 'AE_PORT')
REMOTE_AE_TITLE = config.get('dicom', 'REMOTE_AE_TITLE')
REMOTE_AE_IP = config.get('dicom', 'REMOTE_AE_IP')
REMOTE_AE_PORT = config.getint('dicom', 'REMOTE_AE_PORT')

def send_c_move(ae, peer_ae, peer_ip, peer_port, study_instance_uid):
    """
    Send a C-MOVE request to retrieve a study from a remote DICOM server.
    
    This function establishes a DICOM association with a remote PACS and sends
    a C-MOVE request to have a specific study (identified by Study Instance UID)
    sent to our local AE.
    
    Args:
        ae (AE): Local Application Entity
        peer_ae (str): Remote AE title
        peer_ip (str): Remote AE IP address
        peer_port (int): Remote AE port
        study_instance_uid (str): Study Instance UID to retrieve
        
    Returns:
        None
    """
    # Create the association
    assoc = ae.associate(peer_ip, peer_port, ae_title = peer_ae)
    if assoc.is_established:
        # The retrieval dataset
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid
        # Get the response
        responses = [(None, None)]
        try:
            responses = assoc.send_c_move(ds, ae.ae_title, PatientRootQueryRetrieveInformationModelMove)
        except Exception as e:
            logging.error(f"Error in C-MOVE: {e}")
        for (move_status, _) in responses:
            if move_status:
                logging.info(f"C-MOVE for {study_instance_uid} returned status: 0x{move_status.Status:04X}")
        # Release the association
        assoc.release()
    else:
        logging.error("Could not establish C-MOVE association.")

def query_retrieve_monthly_cr_studies(local_ae, peer_ae, peer_ip, peer_port, year, month, day = None):
    """
    Query and retrieve CR studies for a specified date range.
    
    This function queries a remote DICOM PACS for CR studies for either a full month
    or a specific day, and requests each found study to be sent to the local AE.
    It processes one day at a time with appropriate delays between requests.
    
    Args:
        local_ae (str): Local AE title
        peer_ae (str): Remote AE title
        peer_ip (str): Remote AE IP address
        peer_port (int): Remote AE port
        year (int): Year to query
        month (int): Month to query (1-12)
        day (int, optional): Specific day to query (1-31). If None, queries entire month.
        
    Returns:
        None
    """
    ae = AE(ae_title = local_ae)
    ae.requested_contexts = QueryRetrievePresentationContexts
    ae.connection_timeout = 30
    # Date range
    if day is None:
        start_date = datetime(year, month, 1)
        end_date = (start_date + timedelta(days = 32)).replace(day = 1)
    else:
        start_date = datetime(year, month, day)
        end_date = (start_date + timedelta(days = 1))
    # Process each day separately
    for day in range((end_date - start_date).days):
        date = (start_date + timedelta(days=day)).strftime("%Y%m%d")
        logging.info(f"Query studies for {date}.")
        # The query dataset
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.Modality = "CR"
        ds.StudyDate = date
        # Create the association
        assoc = ae.associate(peer_ip, peer_port, ae_title = peer_ae)
        if assoc.is_established:
            # Get the responses list
            responses = [(None, None)]
            try:
                responses = assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind)
            except Exception as e:
                logging.error(f"Error in C-FIND: {e}")
            for (status, identifier) in responses:
                if status and status.Status in (0xFF00, 0xFF01):
                    study_instance_uid = identifier.StudyInstanceUID
                    logging.info(f"[{date}] Queued Study UID: {study_instance_uid}")
                    send_c_move(ae, peer_ae, peer_ip, peer_port, study_instance_uid)
                    time.sleep(1)
            # Sleep
            time.sleep(10)
            # Release the association
            assoc.release()
        else:
            logging.warning(f"Association failed for {date}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Run monthly CR Query/Retrieve")
    parser.add_argument("--day", type = int, help = "Day number (1-31)")
    parser.add_argument("--month", type = int, required = True, help = "Month number (1-12)")
    parser.add_argument("--year", type = int, required = True, help = "Year (e.g. 2025)")
    parser.add_argument("--ae", default = AE_TITLE, help = "Local AE Title")
    parser.add_argument("--peer-ae", default = REMOTE_AE_TITLE, help = "Peer AE Title")
    parser.add_argument("--peer-ip", default = REMOTE_AE_IP, help = "Peer IP address")
    parser.add_argument("--peer-port", type = int, default = REMOTE_AE_PORT, help = "Peer port")

    args = parser.parse_args()
    query_retrieve_monthly_cr_studies(
        local_ae = args.ae,
        peer_ae = args.peer_ae,
        peer_ip = args.peer_ip,
        peer_port = args.peer_port,
        year = args.year,
        month = args.month,
        day = args.day
    )
