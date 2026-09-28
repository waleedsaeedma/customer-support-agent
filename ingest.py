from dotenv import load_dotenv

load_dotenv()

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

KB_DIR = Path("knowledge_base")
PERSIST_DIR = "chroma_db"

# 1. Load the markdown policy files (not products.json - that's handled separately)
docs = []
for md_file in KB_DIR.glob("*.md"):
    loader = TextLoader(str(md_file), encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} documents: {[d.metadata['source'] for d in docs]}")

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n## ", "\n\n", "\n", " "],
)
chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")

# 3. Embed and store in Chroma
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
print(f"Saved to {PERSIST_DIR}")

# 4. Quick sanity check
results = vectorstore.similarity_search("what is your return policy", k=2)
print("\n--- Test query: 'what is your return policy' ---")
for r in results:
    print(f"[{r.metadata['source']}]\n{r.page_content}\n")
