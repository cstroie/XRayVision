#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CLI tool to import old history.json data into the SQLite database.
"""

import os
import json
import sqlite3
import argparse
import logging

IMAGES_DIR = "images"
DB_FILE = os.path.join(IMAGES_DIR, "history.db")
JSON_FILE = os.path.join(IMAGES_DIR, "history.json")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def ensure_database():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS history (
                file TEXT PRIMARY KEY,
                patient_name TEXT,
                patient_id TEXT,
                study_date TEXT,
                study_time TEXT,
                protocol TEXT,
                response TEXT,
                flagged INTEGER DEFAULT 0
            )
        ''')
    logging.info("Database and table verified.")


def import_history():
    if not os.path.exists(JSON_FILE):
        logging.error(f"No history.json found at {JSON_FILE}")
        return

    with open(JSON_FILE, 'r') as f:
        history_data = json.load(f)

    imported = 0
    with sqlite3.connect(DB_FILE) as conn:
        for item in history_data:
            try:
                conn.execute('''
                    INSERT OR REPLACE INTO history
                    (file, patient_name, patient_id, study_date, study_time, protocol, response, flagged)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    item['file'],
                    item['meta']['patient']['name'],
                    item['meta']['patient']['id'],
                    item['meta']['series']['date'],
                    item['meta']['series']['time'],
                    item['meta']['series']['desc'],
                    item['text'],
                    int(item.get('flagged', False))
                ))
                imported += 1
            except Exception as e:
                logging.warning(f"Failed to import item {item.get('file')}: {e}")

    logging.info(f"Import complete. {imported} entries imported.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import old history.json into the SQLite database.')
    parser.add_argument('--json', type=str, default=JSON_FILE, help='Path to history.json')
    parser.add_argument('--db', type=str, default=DB_FILE, help='Path to history.db')
    args = parser.parse_args()

    # Override paths if provided
    JSON_FILE = args.json
    DB_FILE = args.db

    ensure_database()
    import_history()
