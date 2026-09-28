from typing import Optional, Literal
from langgraph.graph import MessagesState


class SupportState(MessagesState):
    """Shared state for the customer support graph."""
    customer_id: str
    needs_escalation: bool
    escalation_reason: Optional[Literal["refund_request", "negative_sentiment", "none"]]
    rep_policy: Optional[str]
