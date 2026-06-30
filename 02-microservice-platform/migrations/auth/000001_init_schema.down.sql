-- Rollback auth database schema

DROP POLICY IF EXISTS tenant_isolation_audit ON auth_audit_log;
DROP FUNCTION IF EXISTS clean_expired_tokens();
DROP TABLE IF EXISTS auth_audit_log;
DROP TABLE IF EXISTS token_blacklist;
DROP EXTENSION IF EXISTS "uuid-ossp";
