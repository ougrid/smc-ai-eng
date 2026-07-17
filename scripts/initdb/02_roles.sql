-- agent_ro: the ONLY role the LLM-generated-SQL path connects as.
-- LOCAL DEV ONLY password -- this stack never leaves localhost.
CREATE ROLE agent_ro LOGIN PASSWORD 'agent_ro_local_dev';

GRANT CONNECT ON DATABASE findata TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON TABLE financial_data TO agent_ro;
-- Deliberately NO other grants: the users table (created later by role "app")
-- is invisible to agent_ro, regardless of what SQL an LLM generates.

ALTER ROLE agent_ro SET default_transaction_read_only = on;
ALTER ROLE agent_ro SET statement_timeout = '5s';
