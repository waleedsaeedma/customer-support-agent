from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from graph import builder

memory = MemorySaver()
store = InMemoryStore()
local_graph = builder.compile(checkpointer=memory, store=store, interrupt_before=["human_feedback"])

customer_id = "cust-1"

# Conversation 1: thread "a"
config_a = {"configurable": {"thread_id": "thread-a"}}
result = local_graph.invoke(
    {"messages": [HumanMessage(content="My name is Ahmed, I want a laptop for gaming, budget 1600 euros")], "customer_id": customer_id},
    config_a,
)
print("=== Thread A reply ===")
print(result["messages"][-1].content)

# Check what got written to the store
print("\n=== Stored profile after thread A ===")
profile = store.get((customer_id, "profile"), "profile")
print(profile.value if profile else "NOTHING STORED")

# Conversation 2: a brand new thread, same customer
config_b = {"configurable": {"thread_id": "thread-b"}}
result = local_graph.invoke(
    {"messages": [HumanMessage(content="What do you know about me so far?")], "customer_id": customer_id},
    config_b,
)
print("\n=== Thread B reply (should reference Ahmed / gaming / budget) ===")
print(result["messages"][-1].content)
