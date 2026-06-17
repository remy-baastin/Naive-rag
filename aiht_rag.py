# ============================================================
# 1. Install Dependencies (run in terminal first)
# ============================================================
# pip install "numpy<2.0" "scipy>=1.11,<1.14" sentence-transformers langchain-groq pypdf langchain-huggingface langchain-pinecone langchain-community ragas datasets pandas python-dotenv torch --index-url https://download.pytorch.org/whl/cu121


# ============================================================
# 2. API Keys (from .env file)
# ============================================================
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")


# ============================================================
# 3. Check GPU availability
# ============================================================
import torch

if torch.cuda.is_available():
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f} GB)")
else:
    device = "cpu"
    print("⚠️  No GPU detected. Running on CPU.")

print(f"📍 Using device: {device}\n")


# ============================================================
# 4. Select files to load (supports multiple files)
# ============================================================
import sys
import glob

# Check if file paths are passed as arguments
if len(sys.argv) > 1:
    filepaths = sys.argv[1:]
else:
    # Look for PDF and CSV files in current directory
    available_files = sorted(glob.glob("*.pdf") + glob.glob("*.csv"))
    if not available_files:
        print("❌ No PDF or CSV files found in the current directory.")
        print("Usage: python aiht_rag.py <file1> <file2> ...")
        sys.exit(1)

    print("📂 Available files:")
    for i, f in enumerate(available_files, 1):
        print(f"   {i}. {f}")
    print(f"   a. All files")

    choice = input("\n📄 Enter file numbers separated by commas (e.g. 1,3), 'a' for all, or filenames: ").strip()

    if choice.lower() == "a":
        filepaths = available_files
    elif "," in choice:
        # Multiple selections: "1,2,3" or "file1.pdf,file2.csv"
        parts = [p.strip() for p in choice.split(",")]
        filepaths = []
        for p in parts:
            if p.isdigit() and 1 <= int(p) <= len(available_files):
                filepaths.append(available_files[int(p) - 1])
            elif os.path.isfile(p):
                filepaths.append(p)
            else:
                print(f"⚠️  Skipping invalid selection: {p}")
    elif choice.isdigit() and 1 <= int(choice) <= len(available_files):
        filepaths = [available_files[int(choice) - 1]]
    elif os.path.isfile(choice):
        filepaths = [choice]
    else:
        print(f"❌ Invalid selection: {choice}")
        sys.exit(1)

# Validate all files exist
for fp in filepaths:
    if not os.path.isfile(fp):
        print(f"❌ File not found: {fp}")
        sys.exit(1)

filenames = [os.path.basename(fp) for fp in filepaths]
print(f"\n✅ Selected {len(filepaths)} file(s):")
for f in filenames:
    print(f"   📄 {f}")
print()


# ============================================================
# 5. Embeddings & Document Loading (GPU accelerated, multi-file)
# ============================================================
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": device},
    encode_kwargs={
        "device": device,
        "batch_size": 64,
        "normalize_embeddings": True,
    },
)

print(f"✅ Embeddings model loaded on {device.upper()}")

# Load documents from ALL selected files
from langchain_community.document_loaders import PyPDFLoader, CSVLoader

all_documents = []

for filepath in filepaths:
    fname = os.path.basename(filepath)

    if fname.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
        print(f"📖 Loading PDF: {fname}")
    elif fname.lower().endswith(".csv"):
        loader = CSVLoader(filepath, encoding="utf-8-sig")
        print(f"📊 Loading CSV: {fname}")
    else:
        print(f"⚠️  Skipping unsupported file: {fname}")
        continue

    docs = loader.load()

    # Tag each document with its source filename
    for doc in docs:
        doc.metadata["source_file"] = fname

    all_documents.extend(docs)
    print(f"   ✅ {fname}: {len(docs)} pages/rows loaded")

# Split all documents into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
documents = text_splitter.split_documents(all_documents)

print(f"\n✅ Total: {len(documents)} chunks from {len(filepaths)} file(s)\n")


# ============================================================
# 6. Pinecone Vector Store
# ============================================================
import time
from pinecone import Pinecone as PineconeClient, ServerlessSpec

pc = PineconeClient(api_key=PINECONE_API_KEY)

index_name = "my-index"
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(index_name).status["ready"]:
        time.sleep(1)
    print("✅ Pinecone index created")
else:
    print("✅ Pinecone index already exists")

from langchain_pinecone import PineconeVectorStore

print("⏳ Embedding and uploading documents to Pinecone...")
vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name,
)

print("✅ Documents embedded and stored in Pinecone\n")


# ============================================================
# 7. Retriever
# ============================================================
retriever = vectorstore.as_retriever()


# ============================================================
# 8. RAG Chatbot with Memory (Groq LLM)
# ============================================================
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=GROQ_API_KEY,
)

# Chat prompt with conversation history
template = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant that answers questions based on the provided context. "
     "Use the provided context to answer the question. "
     "If the context doesn't contain relevant information, say so honestly. "
     "Keep your answers concise and conversational.\n\n"
     "Context: {context}"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Store conversation history
chat_history = []


def format_input(user_input):
    """Format the input with context and chat history."""
    docs = retriever.invoke(user_input)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {
        "context": context,
        "chat_history": chat_history,
        "input": user_input,
    }


# Build the chain
rag_chain = (
    RunnableLambda(format_input)
    | template
    | llm
    | StrOutputParser()
)


# ============================================================
# 9. Interactive Chat Loop
# ============================================================
files_str = ", ".join(filenames)
print("=" * 60)
print("🤖 RAG Chatbot Ready!")
print(f"📚 Loaded: {files_str}")
print(f"📦 Total chunks: {len(documents)}")
print(f"🖥️  Device: {device.upper()}" + (f" ({gpu_name})" if device == "cuda" else ""))
print("=" * 60)
print("Commands:")
print("  quit   → Exit the chatbot")
print("  clear  → Reset conversation history")
print("  eval   → Run RAGAS evaluation on chat history")
print("  files  → Show loaded files")
print("=" * 60 + "\n")

# Store Q&A for evaluation
all_questions = []
all_responses = []
all_contexts = []

while True:
    try:
        user_input = input("\033[94m🧑 You:\033[0m ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n\n👋 Goodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() == "quit":
        print("\n👋 Goodbye!")
        break

    if user_input.lower() == "clear":
        chat_history.clear()
        all_questions.clear()
        all_responses.clear()
        all_contexts.clear()
        print("🗑️  Conversation history cleared.\n")
        continue

    if user_input.lower() == "files":
        print(f"\n📚 Loaded {len(filepaths)} file(s):")
        for f in filenames:
            print(f"   📄 {f}")
        print(f"   📦 Total chunks: {len(documents)}\n")
        continue

    if user_input.lower() == "eval":
        if not all_questions:
            print("⚠️  No conversation to evaluate yet. Ask some questions first.\n")
            continue

        print("\n⏳ Running RAGAS evaluation on conversation...")

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        from ragas import EvaluationDataset, SingleTurnSample, evaluate
        from ragas.metrics import ResponseRelevancy, Faithfulness
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper

        ragas_llm = LangchainLLMWrapper(llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

        samples = []
        for i in range(len(all_questions)):
            samples.append(
                SingleTurnSample(
                    user_input=all_questions[i],
                    response=all_responses[i],
                    retrieved_contexts=all_contexts[i],
                )
            )

        eval_dataset = EvaluationDataset(samples=samples)

        metrics = [
            ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
            Faithfulness(llm=ragas_llm),
        ]

        results = evaluate(dataset=eval_dataset, metrics=metrics)

        import pandas as pd
        results_df = results.to_pandas()
        print("\n📈 RAGAS Evaluation Results:")
        print(results_df.to_string())
        print()
        continue

    # Get response from RAG chain
    try:
        # Get context for storing
        retrieved_docs = retriever.invoke(user_input)
        context_texts = [doc.page_content for doc in retrieved_docs]

        # Get response
        response = rag_chain.invoke(user_input)

        # Display response
        print(f"\n\033[92m🤖 Bot:\033[0m {response}\n")

        # Update chat history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        # Keep history manageable (last 10 exchanges = 20 messages)
        if len(chat_history) > 20:
            chat_history[:] = chat_history[-20:]

        # Store for evaluation
        all_questions.append(user_input)
        all_responses.append(response)
        all_contexts.append(context_texts)

    except Exception as e:
        print(f"\n❌ Error: {e}\n")