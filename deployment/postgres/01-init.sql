-- Postgres bootstrap. Runs once on first container start (when the data volume is empty).
-- SQLAlchemy's init_db() creates the actual tables on backend startup, so this file
-- only handles extensions and instance-level tuning.

-- Useful for case-insensitive text columns and trigram indexes if needed later
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
