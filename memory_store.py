from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.store.base import BaseStore

from memory_schema import CustomerProfile

profile_llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(CustomerProfile)


def load_profile(store: BaseStore, customer_id: str) -> CustomerProfile:
    """Read the customer's profile from the store, or return a blank one."""
    namespace = (customer_id, "profile")
    existing = store.get(namespace, "profile")
    if existing:
        return CustomerProfile(**existing.value)
    return CustomerProfile()


def write_profile(store: BaseStore, customer_id: str, messages: list, escalated: bool = False) -> CustomerProfile:
    """Reflect on the conversation and update the stored profile."""
    namespace = (customer_id, "profile")
    existing = store.get(namespace, "profile")
    existing_profile = existing.value if existing else {}
    existing_count = existing_profile.get("past_escalations", 0)

    convo_text = "\n".join(f"{m.type}: {m.content}" for m in messages if m.content)

    result = profile_llm.invoke(
        f"""Existing customer profile: {existing_profile}

Conversation:
{convo_text}

Update the profile with any new facts about this customer: their name if mentioned,
notes on what they asked about or issues they had, and any stated preferences (budget
range, product type, etc). Keep unrelated existing fields as they were.
Leave past_escalations unchanged, it is tracked separately."""
    )
    result.past_escalations = existing_count + (1 if escalated else 0)
    store.put(namespace, "profile", result.model_dump())
    return result
