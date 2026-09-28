from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool

from products import get_products

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)


@tool
def search_policies(query: str) -> str:
    """Search FAQ, shipping, and return policy documents."""
    results = vectorstore.similarity_search(query, k=2)
    return "\n\n---\n\n".join(r.page_content for r in results)


@tool
def find_products(category: str = None, max_price: float = None) -> str:
    """Find laptops or phones, optionally filtered by category and max price in EUR."""
    products = get_products(category=category, max_price=max_price)
    if not products:
        return "No products found matching those criteria."
    return "\n".join(
        f"{p['name']} - {p['price_eur']} EUR - {p['use_case']}" for p in products
    )


tools = [search_policies, find_products]
