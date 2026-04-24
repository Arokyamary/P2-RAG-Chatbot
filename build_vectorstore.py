import os
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

os.makedirs('chroma_db', exist_ok=True)
all_docs = []

for csv_file in ['data/sales_report.csv', 'data/product_data.csv']:
    print(f'Loading {csv_file}...')
    loader = CSVLoader(file_path=csv_file, encoding='utf-8')
    docs = loader.load()
    all_docs.extend(docs)
    print(f' Loaded {len(docs)} rows')

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
chunks = splitter.split_documents(all_docs)
print(f'Chunks: {len(chunks)}')

print('Loading embedding model (downloads 80MB on first run)...')
embeddings = HuggingFaceEmbeddings(
    model_name='all-MiniLM-L6-v2',
    model_kwargs={'device': 'cpu'}
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'
)
print(f'Done! Vectors: {vectorstore._collection.count()}')
print('chroma_db folder created.')