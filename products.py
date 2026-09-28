import json
from pathlib import Path

PRODUCTS_PATH = Path("knowledge_base/products.json")


def load_products() -> list[dict]:
    with open(PRODUCTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_products(
    category: str | None = None,
    max_price: float | None = None,
    min_price: float | None = None,
) -> list[dict]:
    """Filter products by category and/or price range."""
    products = load_products()

    if category:
        products = [p for p in products if p["category"] == category.lower()]
    if max_price is not None:
        products = [p for p in products if p["price_eur"] <= max_price]
    if min_price is not None:
        products = [p for p in products if p["price_eur"] >= min_price]

    return products


if __name__ == "__main__":
    # Quick sanity check
    laptops_under_1000 = get_products(category="laptop", max_price=1000)
    print(f"Laptops under 1000 EUR: {len(laptops_under_1000)}")
    for p in laptops_under_1000:
        print(f"  - {p['name']}: {p['price_eur']} EUR")
