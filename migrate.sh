#!/bin/bash
# Create timestamped backup
BACKUP_FILE="images/xrayvision_backup_$(date +%Y%m%d%H%M%S).db"
sqlite3 images/xrayvision.db ".backup '$BACKUP_FILE'"

# Run migration script
sqlite3 images/xrayvision.db < migrate_valid_column.sql

echo "Migration complete. Backup saved to $BACKUP_FILE"
