import sqlite3
import os
from datetime import datetime

OLD_DB = "images/history.db"
NEW_DB = "images/xrayvision.db"

def migrate_database():
    # Create backup of current database
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"images/history_backup_{timestamp}.db"
    os.rename(OLD_DB, backup_path)
    
    # Connect to old and new databases
    old_conn = sqlite3.connect(backup_path)
    new_conn = sqlite3.connect(NEW_DB)
    
    # Create new schema
    new_conn.execute('''
        CREATE TABLE exams (
            uid TEXT PRIMARY KEY,
            name TEXT,
            id TEXT,
            age INTEGER,
            sex TEXT CHECK(sex IN ('M', 'F', 'O')),
            created TIMESTAMP,
            protocol TEXT,
            region TEXT,
            reported TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            report TEXT,
            positive INTEGER DEFAULT 0 CHECK(positive IN (0, 1)),
            valid INTEGER DEFAULT 1 CHECK(valid IN (0, 1)),
            reviewed INTEGER DEFAULT 0 CHECK(reviewed IN (0, 1)),
            status TEXT DEFAULT 'none'
        )
    ''')
    new_conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_cleanup 
        ON exams(status, created, valid)
    ''')
    
    # Migrate data
    old_cursor = old_conn.cursor()
    old_cursor.execute("SELECT * FROM exams")
    
    for row in old_cursor.fetchall():
        # Convert iswrong to valid (invert value)
        valid = 0 if row[11] else 1  # iswrong was at index 11
        
        new_conn.execute('''
            INSERT INTO exams VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
        ''', (*row[:11], valid, *row[12:]))
    
    new_conn.commit()
    old_conn.close()
    new_conn.close()
    print(f"Database migrated successfully! Backup saved to {backup_path}")

if __name__ == "__main__":
    migrate_database()
