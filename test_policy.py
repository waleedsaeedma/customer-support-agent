from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from graph import builder
from policy_store import load_policy

store = InMemoryStore()
g = builder.compile(checkpointer=MemorySaver(), store=store, interrupt_before=["human_feedback"])


def send(thread, customer, text):
    config = {"configurable": {"thread_id": thread}}
    g.invoke({"messages": [HumanMessage(content=text)], "customer_id": customer}, config)
    return config, g.get_state(config).next == ("human_feedback",)


config, paused = send("t1", "cust-1", "My order is 2 days late and I am furious, this is unacceptable!")
print("Test 1 - angry, no policy yet. Paused:", paused, "(expect True)")

g.update_state(config, {"rep_policy": "Do not escalate shipping delays under 3 days. Answer from the shipping policy instead."}, as_node="human_feedback")
g.invoke(None, config)
print("Saved policy:", load_policy(store))

_, paused = send("t2", "cust-2", "My order is 1 day late and I am really annoyed.")
print("Test 2 - same kind of message after policy. Paused:", paused, "(expect False)")

_, paused = send("t3", "cust-3", "I want a refund for my laptop.")
print("Test 3 - refund request. Paused:", paused, "(expect True, policy can't change this)")

print("cust-1 past_escalations:", store.get(("cust-1", "profile"), "profile").value["past_escalations"], "(expect 1)")
