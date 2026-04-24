import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
vectorstore = Chroma(persist_directory='./chroma_db', embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

llm = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0,
    groq_api_key=os.getenv('GROQ_API_KEY'))

PROMPT = PromptTemplate(
    input_variables=['context', 'question'],
    template=(
        'You are a senior BI analyst. Use ONLY this data to answer accurately. '
        'Be specific with numbers. If not in data, say: Data not available. '
        'Never invent numbers.\n\nData:\n{context}\n\n'
        'Question: {question}\nAnswer:'
    )
)

def ask(question):
    docs = retriever.invoke(question)
    context = '\n'.join([d.page_content for d in docs])
    chain = PROMPT | llm | StrOutputParser()
    return chain.invoke({'context': context, 'question': question})

if __name__ == '__main__':
    for q in [
        'Which city had the lowest revenue in January 2024?',
        'What is the return rate for Electronics in Bengaluru March 2024?',
        'Which product has the most complaints?',
    ]:
        print(f'Q: {q}\nA: {ask(q)}\n')