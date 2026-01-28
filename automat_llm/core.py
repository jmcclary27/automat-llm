# automat_llm/core.py
import os
import json
import logging

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_groq import ChatGroq

current_dir = os.getcwd()


def load_json_as_documents(client, directory):
    """
    Load all .json files in `directory` as LangChain Documents (pretty-printed JSON in page_content),
    then upload those documents to Weaviate collection "MyCollection" as objects with property 'entry'.

    Notes:
    - This function assumes your Weaviate client is already configured with auth/url.
    - It uses BM25 retrieval later, so no local embedding is required here.
    """
    documents = []

    client.connect()
    collection = client.collections.use("MyCollection")  # TBA: user-specific collection

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue

        path = os.path.join(directory, filename)
        with open(path, "r", encoding="utf-8") as f:
            try:
                raw_content = f.read()
                parsed = json.loads(raw_content)
                pretty_json = json.dumps(parsed, indent=2)
                documents.append(Document(page_content=pretty_json, metadata={"source": filename}))
            except Exception as e:
                print(f"Skipping {filename} due to error: {e}")

    # Upload each document's text as an 'entry' in Weaviate
    entries = [doc.page_content for doc in documents]

    with collection.batch.fixed_size(batch_size=200) as batch:
        for d in entries:
            batch.add_object({"entry": d})
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = collection.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")

    client.close()  # Free up resources
    return documents


def init_interactions():
    """
    Load or initialize user interactions stored in user_interactions.json in the repo cwd.
    """
    user_interactions_file = f"{current_dir}/user_interactions.json"
    try:
        with open(user_interactions_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        user_interactions = {"users": {}}
        with open(user_interactions_file, "w", encoding="utf-8") as f:
            json.dump(user_interactions, f, indent=4)
        return user_interactions


def load_personality_file():
    """
    Load the personality from robot_personality.json in the repo cwd.
    """
    personality_file = f"{current_dir}/robot_personality.json"
    try:
        with open(personality_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Personality file not found at {personality_file}. Please create robot_personality.json.")
        logging.error(f"Personality file not found at {personality_file}.")
        raise SystemExit(1)


def create_rag_chain(client, user_id, documents):
    """
    Create a RAG chain using:
    - Weaviate for storing/retrieving context (BM25, no local embeddings needed)
    - Groq as the LLM via langchain_groq

    This avoids:
    - langchain_huggingface
    - torch/transformers embedding pipeline on Windows
    - langchain.chains import drift
    """
    from weaviate.classes.config import Configure

    client.connect()

    # Ensure collection exists
    if client.collections.exists("Embeddings"):
        col = client.collections.get("Embeddings")
    else:
        col = client.collections.create(
            name="Embeddings",
            vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
        )

    # Upload docs as objects (Weaviate will vectorize if configured)
    with col.batch.fixed_size(batch_size=200) as batch:
        for doc in documents:
            batch.add_object(
                properties={
                    "user_id": user_id,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                }
            )
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a snarky but helpful assistant."),
            ("human", "{input}\n\nUse this context if helpful:\n{context}"),
        ]
    )

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set in your environment variables.")

    llm = ChatGroq(
        temperature=0.5,
        model="openai/gpt-oss-20b",
        max_tokens=5000,
        api_key=groq_key,
    )

    def retrieve_context(query: str, k: int = 5) -> str:
        # Simple BM25 text search, no local embeddings required
        res = col.query.bm25(
            query=query,
            limit=k,
            return_properties=["text", "source"],
        )

        parts = []
        for obj in getattr(res, "objects", []) or []:
            props = obj.properties or {}
            src = props.get("source", "unknown")
            txt = props.get("text", "")
            parts.append(f"[{src}]\n{txt}")
        return "\n\n".join(parts)

    rag_chain = (
        {
            "context": (lambda x: retrieve_context(x["input"])),
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


def update_user_interactions(user_id, user_interactions_file, user_interactions, is_rude=False, apologized=False):
    """
    Update the user interactions state (rudeness score and apology requirement).
    """
    if user_id not in user_interactions["users"]:
        user_interactions["users"][user_id] = {"rudeness_score": 0, "requires_apology": False}

    user_data = user_interactions["users"][user_id]
    if is_rude:
        user_data["rudeness_score"] += 1
        if user_data["rudeness_score"] >= 2:
            user_data["requires_apology"] = True
    elif apologized:
        user_data["rudeness_score"] = 0
        user_data["requires_apology"] = False

    with open(user_interactions_file, "w", encoding="utf-8") as f:
        json.dump(user_interactions, f, indent=4)


def generate_response(user_id, user_interactions, user_input, rude_keywords, personality_data, rag_chain):
    """
    Generate a response using the RAG chain.
    This version expects rag_chain to return a plain string (StrOutputParser).
    """
    input_lower = user_input.lower()

    user_data = user_interactions["users"].get(user_id, {"rudeness_score": 0, "requires_apology": False})
    if user_data.get("requires_apology", False):
        if "sorry" in input_lower or "apologize" in input_lower:
            user_interactions_file = f"{current_dir}/user_interactions.json"
            update_user_interactions(user_id, user_interactions_file, user_interactions, apologized=True)
            return next(
                item["response"]
                for item in personality_data["example_dialogue"]
                if item["user"].lower() == "i’m sorry for being rude."
            )
        return "I’m waiting for an apology, sweetie. I don’t respond to rudeness without respect."

    is_rude = any(keyword in input_lower for keyword in rude_keywords)
    if is_rude:
        user_interactions_file = f"{current_dir}/user_interactions.json"
        update_user_interactions(user_id, user_interactions_file, user_interactions, is_rude=True)
        return next(
            item["response"]
            for item in personality_data["example_dialogue"]
            if item["user"].lower() == "just do what i say, you stupid robot!"
        )

    try:
        response = rag_chain.invoke({"input": user_input})

        logging.info(f"User: {user_input}")
        logging.info(f"Bot: {response}")
        logging.info("")

        return response

    except Exception as e:
        print(f"Error generating response: {e}")
        logging.error(f"Error generating response: {e}", exc_info=True)
        return "I'm sorry, I couldn't process your request."
