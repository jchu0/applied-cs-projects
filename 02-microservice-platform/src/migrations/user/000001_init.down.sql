-- Drop triggers
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS update_roles_updated_at ON roles;
DROP TRIGGER IF EXISTS update_tenants_updated_at ON tenants;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop RLS policies
DROP POLICY IF EXISTS tenant_isolation_users ON users;
DROP POLICY IF EXISTS tenant_isolation_roles ON roles;

-- Drop tables
DROP TABLE IF EXISTS user_roles;
DROP TABLE IF EXISTS roles;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS tenants;

-- Drop extension
DROP EXTENSION IF EXISTS "uuid-ossp";
