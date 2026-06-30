-- Drop triggers
DROP TRIGGER IF EXISTS update_customers_updated_at ON customers;
DROP TRIGGER IF EXISTS update_subscriptions_updated_at ON subscriptions;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop RLS policies
DROP POLICY IF EXISTS tenant_isolation_customers ON customers;
DROP POLICY IF EXISTS tenant_isolation_subscriptions ON subscriptions;
DROP POLICY IF EXISTS tenant_isolation_invoices ON invoices;
DROP POLICY IF EXISTS tenant_isolation_usage ON usage_records;

-- Drop tables
DROP TABLE IF EXISTS usage_records;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS payment_methods;
DROP TABLE IF EXISTS subscriptions;
DROP TABLE IF EXISTS customers;

-- Drop extension
DROP EXTENSION IF EXISTS "uuid-ossp";
