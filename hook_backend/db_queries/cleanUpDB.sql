-- Delete all records from positions table first (because of foreign key constraint)
TRUNCATE TABLE positions CASCADE;

-- Delete all records from users table
TRUNCATE TABLE users CASCADE;

-- Verify tables are empty
SELECT COUNT(*) FROM positions;
SELECT COUNT(*) FROM users;