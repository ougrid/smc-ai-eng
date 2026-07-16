from sqlalchemy import Engine, create_engine


def build_engine(url: str) -> Engine:
    """Engine for the app role — full read/write access to its own tables."""
    return create_engine(url)


def build_agent_ro_engine(url: str) -> Engine:
    """Engine for the agent_ro role — the only role the LLM-generated-SQL path uses.

    Kept as a separate engine/connection (never the app engine) so the agent
    physically cannot see tables the app role can (e.g. users).
    """
    return create_engine(
        url,
        connect_args={"options": "-c default_transaction_read_only=on"},
    )
