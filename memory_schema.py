from typing import Optional
from pydantic import BaseModel


class CustomerProfile(BaseModel):
    """Long-term memory about a customer, persisted across conversations."""
    name: Optional[str] = None
    notes: Optional[str] = None
    past_escalations: int = 0
    preferences: Optional[str] = None
