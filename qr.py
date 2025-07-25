#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
import argparse
import logging
import time
from datetime import datetime, timedelta
from pynetdicom import AE, QueryRetrievePresentationContexts
from pynetdicom.sop_class import PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove
from pydicom.dataset import Dataset

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s | %(levelname)8s | %(message)s',
    handlers = [
        #logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)

def send_c_move(ae, peer_ae, peer_ip, peer_port, study_instance_uid):
    """ Ask for a study to be sent """
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
    """ Query and Retrieve studies """
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
    parser.add_argument("--ae", default = "XRAYVISION", help = "Local AE Title")
    parser.add_argument("--peer-ae", default = "3DNETCLOUD", help = "Peer AE Title")
    parser.add_argument("--peer-ip", default = "192.168.3.50", help = "Peer IP address")
    parser.add_argument("--peer-port", type = int, default = 104, help = "Peer port")

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
