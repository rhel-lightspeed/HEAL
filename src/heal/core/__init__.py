"""Core utilities for HEAL framework.

Note: All agents have been moved to heal.agents.
This module re-exports them for backward compatibility using lazy imports.
"""

__all__ = ["AnswerReviewAgent", "LinuxExpertAgent", "SolrExpertAgent", "URLValidationAgent"]


def __getattr__(name):
    """Lazy import agents to avoid circular import issues."""
    if name == "AnswerReviewAgent":
        from heal.agents.answer_review_agent import AnswerReviewAgent

        return AnswerReviewAgent
    elif name == "LinuxExpertAgent":
        from heal.agents.linux_expert import LinuxExpertAgent

        return LinuxExpertAgent
    elif name == "SolrExpertAgent":
        from heal.agents.solr_expert import SolrExpertAgent

        return SolrExpertAgent
    elif name == "URLValidationAgent":
        from heal.agents.url_validation_agent import URLValidationAgent

        return URLValidationAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
