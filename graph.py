from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore
from pydantic import BaseModel
from typing import Literal

from state import SupportState
from tools import tools
from prompts import SYSTEM_PROMPT
from memory_store import load_profile, write_profile
from policy_store import load_policy, add_policy_rule, format_policy

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_with_tools = llm.bind_tools(tools)


class TriageFlags(BaseModel):
    is_refund_request: bool
    is_negative_sentiment: bool


class SentimentDecision(BaseModel):
    escalate: bool


triage_llm = llm.with_structured_output(TriageFlags)
sentiment_llm = llm.with_structured_output(SentimentDecision)


def triage(state: SupportState, *, store: BaseStore):
    text = state["messages"][-1].content
    flags = triage_llm.invoke(
        f"Message: {text}\n\nIs this a request for a refund? Does it show clear negative or angry sentiment?"
    )

    # Refund requests always escalate. Standing policy never sees this decision.
    if flags.is_refund_request:
        return {"needs_escalation": True, "escalation_reason": "refund_request"}

    # Policy can only relax the negative-sentiment case.
    if flags.is_negative_sentiment:
        policy = format_policy(load_policy(store))
        decision = sentiment_llm.invoke(
            f"Message: {text}\n\n{policy}\n\n"
            "This customer message shows negative sentiment. Should a human rep step in? "
            "Follow the standing policy above if it applies, otherwise answer true."
        )
        if decision.escalate:
            return {"needs_escalation": True, "escalation_reason": "negative_sentiment"}

    return {"needs_escalation": False, "escalation_reason": "none"}


def assistant(state: SupportState, *, store: BaseStore):
    profile = load_profile(store, state.get("customer_id") or "guest")
    profile_context = (
        f"\n\nWhat you know about this customer: {profile.model_dump()}"
        if profile.notes or profile.name else ""
    )
    policy_text = format_policy(load_policy(store))
    policy_context = f"\n\n{policy_text}" if policy_text else ""
    messages = [SystemMessage(content=SYSTEM_PROMPT + profile_context + policy_context)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def route_after_triage(state: SupportState) -> Literal["human_feedback", "assistant"]:
    if state["needs_escalation"]:
        return "human_feedback"
    return "assistant"


def human_feedback(state: SupportState):
    """No-op node - interrupt_before pauses here for a human rep."""
    pass


def apply_policy(state: SupportState, *, store: BaseStore):
    """If the rep supplied a standing rule, save it. Then clear the field."""
    rule = state.get("rep_policy")
    if rule:
        add_policy_rule(store, rule)
    return {"rep_policy": None}


def route_after_assistant(state: SupportState) -> Literal["tools", "write_memory"]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return "write_memory"


def write_memory(state: SupportState, *, store: BaseStore):
    write_profile(
        store,
        state.get("customer_id") or "guest",
        state["messages"],
        escalated=bool(state.get("needs_escalation")),
    )
    return {}


builder = StateGraph(SupportState)
builder.add_node("triage", triage)
builder.add_node("human_feedback", human_feedback)
builder.add_node("apply_policy", apply_policy)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_node("write_memory", write_memory)

builder.add_edge(START, "triage")
builder.add_conditional_edges("triage", route_after_triage)
builder.add_edge("human_feedback", "apply_policy")
builder.add_edge("apply_policy", "assistant")
builder.add_conditional_edges("assistant", route_after_assistant)
builder.add_edge("tools", "assistant")
builder.add_edge("write_memory", END)

# For LangGraph API / Studio: no custom checkpointer/store - the platform provides both.
graph = builder.compile(interrupt_before=["human_feedback"])

