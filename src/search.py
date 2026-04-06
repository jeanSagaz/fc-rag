import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector


load_dotenv()

def search_prompt(query=None, k: int = 10):

    for e in ('OPENAI_API_KEY', 'PGVECTOR_URL','PGVECTOR_COLLECTION'):
        if not os.getenv(e):
            raise RuntimeError(f'Environment variable {e} is not set')

    embeddings = OpenAIEmbeddings(model=os.getenv('OPENAI_EMBEDDING_MODEL','text-embedding-3-small'))
    # embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv('GOOGLE_EMBEDDING_MODEL'))

    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv('PGVECTOR_COLLECTION'),
        connection=os.getenv('PGVECTOR_URL'),
        use_jsonb=True,
    )

    results = store.similarity_search_with_score(query, k=k)

    # for i, (doc, score) in enumerate(results, start=1):
    #     print("="*50)
    #     print(f"Resultado {i} (score: {score:.2f}):")
    #     print("="*50)

    #     print("\nTexto:\n")
    #     print(doc.page_content.strip())

    #     print("\nMetadados:\n")
    #     for k, v in doc.metadata.items():
    #         print(f"{k}: {v}")

    return results