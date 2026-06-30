-- Drop policies
DROP POLICY IF EXISTS tenant_isolation_user_roles ON user_roles;
DROP POLICY IF EXISTS tenant_isolation_roles ON roles;
DROP POLICY IF EXISTS tenant_isolation_users ON users;

-- Drop tables in reverse order
DROP TABLE IF EXISTS user_roles;
DROP TABLE IF EXISTS roles;
DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS tenants;
