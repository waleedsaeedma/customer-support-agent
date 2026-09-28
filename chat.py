from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from graph import builder
from policy_store import load_policy

memory = MemorySaver()
store = InMemoryStore()
local_graph = builder.compile(checkpointer=memory, store=store, interrupt_before=["human_feedback"])

customer_id = input("Customer ID (try 'cust-1'): ").strip() or "cust-1"
config = {"configurable": {"thread_id": "session-1"}}
print("Chat with the support bot (type 'quit' to exit)")

while True:
    user_input = input("\nYou: ")
    if user_input.lower() == "quit":
        break

    result = local_graph.invoke({"messages": [HumanMessage(content=user_input)], "customer_id": customer_id}, config)

    state = local_graph.get_state(config)
    if state.next == ("human_feedback",):
        reason = state.values.get("escalation_reason")
        rules = load_policy(store)
        print(f"\n[ESCALATED - reason: {reason}]")
        print("Current standing policy:", rules if rules else "none")
        print("  approve         - let the bot continue as-is")
        print("  <text>          - rewrite what the bot should say, then continue")
        print("  policy: <rule>  - save a standing rule for ALL customers, then continue")
        rep_input = input("\nRep: ").strip()

        if rep_input.lower().startswith("policy:"):
            rule = rep_input[len("policy:"):].strip()
            local_graph.update_state(config, {"rep_policy": rule}, as_node="human_feedback")
        elif rep_input.lower() != "approve":
            local_graph.update_state(config, {"messages": [HumanMessage(content=rep_input)]}, as_node="human_feedback")

        result = local_graph.invoke(None, config)

    print(f"\nBot: {result['messages'][-1].content}")
