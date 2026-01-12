#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Find Untranslated Acronyms - Identify acronyms in English translations that weren't translated
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
import argparse

# Add parent directory to path to import xrayvision module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration from xrayvision
from xrayvision import DB_FILE, config

def find_untranslated_acronyms():
    """Find acronyms in English translations that appear to be untranslated.

    This function identifies acronyms (2+ capital letters) that appear in both
    the original Romanian reports and their English translations, suggesting they
    weren't translated.

    Returns:
        dict: Dictionary mapping acronyms to count of occurrences where they
              appear in both original and translated text
    """
    # Get reports with both original and translated text
    query = """
    SELECT text, text_en
    FROM rad_reports
    WHERE text IS NOT NULL AND text != ''
      AND text_en IS NOT NULL AND text_en != ''
    """

    acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
    untranslated_acronyms = {}

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            for row in cursor.fetchall():
                original_text, translated_text = row
                if original_text and translated_text:
                    # Find acronyms in original text
                    original_acronyms = set(acronym_pattern.findall(original_text))
                    # Find acronyms in translated text
                    translated_acronyms = set(acronym_pattern.findall(translated_text))

                    # Find intersection - acronyms that appear in both
                    common_acronyms = original_acronyms.intersection(translated_acronyms)

                    # Count occurrences
                    for acronym in common_acronyms:
                        if acronym in untranslated_acronyms:
                            untranslated_acronyms[acronym] += 1
                        else:
                            untranslated_acronyms[acronym] = 1

        return untranslated_acronyms

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}
    except Exception as e:
        print(f"Error finding untranslated acronyms: {e}")
        return {}

def print_untranslated_acronyms(untranslated_acronyms):
    """Print untranslated acronyms with their occurrence counts.

    Args:
        untranslated_acronyms: Dictionary mapping acronyms to counts
    """
    if not untranslated_acronyms:
        print("No untranslated acronyms found in English translations.")
        return

    print(f"Found {len(untranslated_acronyms)} unique acronyms that may not have been translated:")
    print("=" * 70)
    print(f"{'Acronym':<15} {'Occurrences':<15}")
    print("-" * 70)

    # Sort by occurrence count (descending) then by acronym
    sorted_acronyms = sorted(untranslated_acronyms.items(),
                           key=lambda x: (-x[1], x[0]))

    for acronym, count in sorted_acronyms:
        print(f"{acronym:<15} {count:<15}")

def mark_exams_for_check(untranslated_acronyms):
    """Mark exams with untranslated acronyms for review.

    Args:
        untranslated_acronyms: Dictionary of acronyms to check for

    Returns:
        int: Number of exams marked for check
    """
    # Find exams with untranslated acronyms in their English translations
    query = """
    SELECT uid, text, text_en
    FROM rad_reports
    WHERE text IS NOT NULL AND text != ''
      AND text_en IS NOT NULL AND text_en != ''
    """

    acronym_pattern = re.compile(r'\b[A-Z]{2,}\b')
    exams_to_check = []
    marked_count = 0

    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            cursor.execute(query)

            for row in cursor.fetchall():
                uid, original_text, translated_text = row
                if original_text and translated_text:
                    # Find acronyms in original text
                    original_acronyms = set(acronym_pattern.findall(original_text))
                    # Find acronyms in translated text
                    translated_acronyms = set(acronym_pattern.findall(translated_text))

                    # Find intersection - acronyms that appear in both
                    common_acronyms = original_acronyms.intersection(translated_acronyms)

                    # Check if any of these common acronyms are in our untranslated list
                    if common_acronyms.intersection(untranslated_acronyms.keys()):
                        exams_to_check.append(uid)

            # Mark exams for check
            if exams_to_check:
                update_query = "UPDATE exams SET status = 'check' WHERE uid = ?"
                cursor.executemany(update_query, [(uid,) for uid in exams_to_check])
                conn.commit()
                marked_count = len(exams_to_check)

        return marked_count

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0
    except Exception as e:
        print(f"Error marking exams for check: {e}")
        return 0

def main():
    """Main function to find and display untranslated acronyms."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find untranslated acronyms in English report translations.')
    parser.add_argument('--mark-for-check', action='store_true',
                       help='Mark exams with untranslated acronyms for review')
    args = parser.parse_args()

    print("Finding untranslated acronyms in English report translations...")
    print(f"Using database: {DB_FILE}")

    # Find untranslated acronyms
    untranslated_acronyms = find_untranslated_acronyms()

    # Print results
    print_untranslated_acronyms(untranslated_acronyms)

    # Optionally mark exams for check
    if args.mark_for_check and untranslated_acronyms:
        print("\nMarking exams with untranslated acronyms for review...")
        marked_count = mark_exams_for_check(untranslated_acronyms)
        print(f"Marked {marked_count} exams for review.")

if __name__ == '__main__':
    main()
