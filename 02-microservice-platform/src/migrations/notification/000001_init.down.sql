-- Drop triggers
DROP TRIGGER IF EXISTS update_templates_updated_at ON templates;
DROP TRIGGER IF EXISTS update_preferences_updated_at ON notification_preferences;

-- Drop function
DROP FUNCTION IF EXISTS update_updated_at_column();

-- Drop RLS policies
DROP POLICY IF EXISTS tenant_isolation_templates ON templates;
DROP POLICY IF EXISTS tenant_isolation_notifications ON notifications;
DROP POLICY IF EXISTS tenant_isolation_preferences ON notification_preferences;

-- Drop tables
DROP TABLE IF EXISTS delivery_events;
DROP TABLE IF EXISTS notification_preferences;
DROP TABLE IF EXISTS notifications;
DROP TABLE IF EXISTS templates;

-- Drop extension
DROP EXTENSION IF EXISTS "uuid-ossp";
