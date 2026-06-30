-- Drop functions
DROP FUNCTION IF EXISTS cleanup_old_login_attempts();
DROP FUNCTION IF EXISTS cleanup_expired_tokens();

-- Drop tables
DROP TABLE IF EXISTS mfa_settings;
DROP TABLE IF EXISTS auth_audit_log;
DROP TABLE IF EXISTS login_attempts;
DROP TABLE IF EXISTS refresh_tokens;
