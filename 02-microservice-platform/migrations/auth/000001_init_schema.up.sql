-- Auth database schema (if using separate DB for auth)
-- Note: Sessions are stored in Redis, this is for audit logs and token blacklist

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Token blacklist (for logout/revocation)
CREATE TABLE token_blacklist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_id VARCHAR(255) NOT NULL UNIQUE,
    user_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    token_type VARCHAR(20) NOT NULL, -- access, refresh
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ DEFAULT NOW(),
    revoked_reason VARCHAR(100)
);

-- Auth audit log
CREATE TABLE auth_audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    user_id UUID,
    action VARCHAR(50) NOT NULL, -- login, logout, refresh, failed_login, password_change
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN DEFAULT TRUE,
    failure_reason VARCHAR(255),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_token_blacklist_token_id ON token_blacklist(token_id);
CREATE INDEX idx_token_blacklist_expires_at ON token_blacklist(expires_at);
CREATE INDEX idx_auth_audit_tenant_id ON auth_audit_log(tenant_id);
CREATE INDEX idx_auth_audit_user_id ON auth_audit_log(user_id);
CREATE INDEX idx_auth_audit_action ON auth_audit_log(action);
CREATE INDEX idx_auth_audit_created_at ON auth_audit_log(created_at);

-- Function to clean expired tokens from blacklist
CREATE OR REPLACE FUNCTION clean_expired_tokens()
RETURNS void AS $$
BEGIN
    DELETE FROM token_blacklist WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Row-Level Security
ALTER TABLE auth_audit_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_audit ON auth_audit_log
    USING (tenant_id = current_setting('app.current_tenant', true)::UUID);
