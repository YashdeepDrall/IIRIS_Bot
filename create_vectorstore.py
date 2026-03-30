import os

from gemini_api import GeminiClient
from vector_store import GeminiVectorStore


def create_vector_store_from_directory(data_dir="data", store_path="vectorstore"):
    """
    Incrementally sync the local FAISS store with the text files in `data_dir`.
    Only new or changed chunks are embedded again.
    """
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' was not found.")
        print("Please create the data directory and place your .txt files inside it.")
        return

    store = GeminiVectorStore(
        client=GeminiClient(),
        data_dir=data_dir,
        store_path=store_path,
    )
    summary = store.sync()
    print(
        "Vector store synced successfully. "
        f"Total chunks: {summary['total_chunks']}, "
        f"Reused: {summary['reused_chunks']}, "
        f"New: {summary['new_chunks']}."
    )


if __name__ == "__main__":
    create_vector_store_from_directory()
