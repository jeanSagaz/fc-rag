import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv

from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector


load_dotenv()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def chunk_id(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_existing_ids(engine, collection, source):
    """BUSCA CHUNKS EXISTENTES"""
    query = text("""
        SELECT id
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c
          ON e.collection_id = c.uuid
       WHERE c.name = :collection
        AND e.cmetadata->>'source' = :source
    """)
    with engine.connect() as conn:
        rows = conn.execute(query, {
            'collection': collection,
            'source': source
        }).fetchall()
    return {row[0] for row in rows}


def delete_removed_chunks(engine, collection, source, ids_to_keep):
    """REMOVE CHUNKS ANTIGOS"""
    query = text("""
        DELETE FROM langchain_pg_embedding e
         USING langchain_pg_collection c
         WHERE c.name = :collection
           AND e.cmetadata->>'source' = :source
           AND NOT (e.id = ANY(:ids))
    """)
    with engine.begin() as conn:
        conn.execute(query, {
            'collection': collection,
            'source': source,
            'ids': list(ids_to_keep)
        })


def main():

    for e in ('OPENAI_API_KEY', 'PGVECTOR_URL','PGVECTOR_COLLECTION'):
        if not os.getenv(e):
            raise RuntimeError(f'Environment variable {e} is not set')

    current_dir = Path(__file__).parent
    pdf_path = current_dir / 'document.pdf'

    if not pdf_path.exists():
        raise FileNotFoundError(f'{pdf_path} não encontrado')
    
    collection = os.getenv('PGVECTOR_COLLECTION')
    db_url = os.getenv('PGVECTOR_URL')
    
    engine = create_engine(db_url)

    file_hash_value  = file_hash(pdf_path)    

    docs = PyPDFLoader(str(pdf_path)).load()

    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150, add_start_index=False).split_documents(docs)
    if not splits:
        print('⚠️  Nenhum conteúdo encontrado.')
        raise SystemExit(0)
    
    enriched = []
    ids = []

    for d in splits:
        cid = chunk_id(d.page_content)

        enriched.append(
            Document(
                page_content=d.page_content,
                metadata={
                    **d.metadata,
                    'source': str(pdf_path),
                    'file_hash': file_hash_value,
                    'chunk_id': cid
                }
            )
        )
        ids.append(cid)

    try:
        # buscar existentes
        existing_ids = get_existing_ids(engine=engine, collection=collection, source=str(pdf_path))
    except ProgrammingError:
        print('⚠️  Tabelas ainda não existem. Primeira execução.')
        existing_ids = set()

    new_ids = set(ids)
    ids_to_insert = list(new_ids - existing_ids)

    if not ids_to_insert:
        print('⚠️  Nada novo para inserir.')
    else:
        docs_to_insert = [
            doc for doc, cid in zip(enriched, ids)
            if cid in ids_to_insert
        ]

        embeddings = OpenAIEmbeddings(model=os.getenv('OPENAI_EMBEDDING_MODEL','text-embedding-3-small'))
        # embeddings = GoogleGenerativeAIEmbeddings(model=os.getenv('GOOGLE_EMBEDDING_MODEL'))

        print('💾 Salvando no PostgreSQL (pgvector)...')
        store = PGVector(
            embeddings=embeddings,
            collection_name=os.getenv('PGVECTOR_COLLECTION'),
            connection=os.getenv('PGVECTOR_URL'),
            use_jsonb=True,
        )

        store.add_documents(documents=docs_to_insert, ids=ids_to_insert)
        print('💾 Inserindo embeddings...')

    # 🧹 remover chunks antigos
    delete_removed_chunks(engine=engine, collection=collection, source=str(pdf_path), ids_to_keep=new_ids)

    print('✅ Sincronização concluída!')

if __name__ == '__main__':
    main()
