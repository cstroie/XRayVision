#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Find Acronyms - Extract acronyms from radiologist reports
# Copyright (C) 2025 Costin Stroie <costinstroie@eridu.eu.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

import re
import sqlite3
import sys
import os
from datetime import datetime

# Add parent directory to path to import xrayvision module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration from xrayvision
from xrayvision import DB_FILE, config

def find_acronyms():
    """Find acronyms (words of 2 or more capital letters) in rad_reports text.

    Returns:
        list: List of distinct acronyms found in radiologist reports
    """
    # Get all non-empty reports
    query = "SELECT text FROM rad_reports WHERE text IS NOT NULL AND text != ''"

    acronyms = set()
    # Regex pattern for words with 2+ capital letters
    acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            for row in cursor.fetchall():
                text = row[0]
                if text:
                    # Find all matches in the text
                    matches = acronym_pattern.findall(text)
                    acronyms.update(matches)

        return sorted(list(acronyms))

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return []
    except Exception as e:
        print(f"Error finding acronyms: {e}")
        return []

def print_acronyms(acronyms):
    """Print acronyms in a formatted way.

    Args:
        acronyms: List of acronyms to print
    """
    if not acronyms:
        print("No acronyms found in radiologist reports.")
        return

    print(f"Found {len(acronyms)} unique acronyms in radiologist reports:")
    print("=" * 50)

    # Print acronyms in columns
    for i, acronym in enumerate(acronyms, 1):
        print(f"{acronym:10}", end="")
        if i % 5 == 0:
            print()

    # Print newline if we didn't end on a new line
    if len(acronyms) % 5 != 0:
        print()

def save_acronyms_to_file(acronyms, filename="acronyms.txt"):
    """Save acronyms to a text file.

    Args:
        acronyms: List of acronyms to save
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Acronyms found in radiologist reports ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write("=" * 60 + "\n")
            for acronym in acronyms:
                f.write(f"{acronym}\n")

        print(f"\nAcronyms saved to {filename}")

    except IOError as e:
        print(f"Error saving acronyms to file: {e}")

def main():
    """Main function to find and display acronyms."""
    print("Finding acronyms in radiologist reports...")
    print(f"Using database: {DB_FILE}")

    # Find acronyms
    acronyms = find_acronyms()

    # Print results
    print_acronyms(acronyms)

    # Save to file
    save_acronyms_to_file(acronyms)

if __name__ == '__main__':
    main()
