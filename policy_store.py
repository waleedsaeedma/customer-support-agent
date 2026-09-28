from langgraph.store.base import BaseStore

NAMESPACE = ("global", "policy")
KEY = "rules"


def load_policy(store: BaseStore) -> list[str]:
    item = store.get(NAMESPACE, KEY)
    return list(item.value["rules"]) if item else []


def add_policy_rule(store: BaseStore, rule: str) -> list[str]:
    rules = load_policy(store)
    rules.append(rule.strip())
    store.put(NAMESPACE, KEY, {"rules": rules})
    return rules


def format_policy(rules: list[str]) -> str:
    if not rules:
        return ""
    return "Standing support policy set by human reps:\n" + "\n".join(f"- {r}" for r in rules)
