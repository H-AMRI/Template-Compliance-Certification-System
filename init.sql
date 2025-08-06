-- Create tables if they don't exist
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Templates table is created by SQLAlchemy, but we can add indexes
CREATE INDEX IF NOT EXISTS idx_templates_name ON templates(name);
CREATE INDEX IF NOT EXISTS idx_templates_created_at ON templates(created_at);

-- Add any additional setup here
GRANT ALL PRIVILEGES ON DATABASE compliance_db TO compliance_user;
