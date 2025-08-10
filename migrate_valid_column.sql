-- Backup will be handled by shell command before running this script

-- Rename column and update values
ALTER TABLE exams RENAME COLUMN iswrong TO valid;
UPDATE exams SET valid = ABS(valid - 1);  -- Invert values (0 becomes 1, 1 becomes 0)

-- Update index
DROP INDEX IF EXISTS idx_cleanup;
CREATE INDEX IF NOT EXISTS idx_cleanup 
ON exams(status, created, valid);

-- Verify changes
PRAGMA main.foreign_key_check;
