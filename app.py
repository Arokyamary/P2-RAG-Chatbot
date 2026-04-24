import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Arokyamary's Assistant", layout='wide', page_icon='🤖')
st.title("Arokyamary's AI Assistant")
st.caption('Powered by LangChain + ChromaDB + Groq Llama 3 (Free LLM)')

@st.cache_resource
def load_chain():
    from langchain_groq import ChatGroq
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    emb = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    vs = Chroma(persist_directory='./chroma_db', embedding_function=emb)
    retriever = vs.as_retriever(search_kwargs={'k': 5})
    llm = ChatGroq(model_name='llama-3.3-70b-versatile', temperature=0,
        groq_api_key=st.secrets.get('GROQ_API_KEY', os.getenv('GROQ_API_KEY', '')))
    pr = PromptTemplate(
        input_variables=['context', 'question'],
        template=(
            "You are Arokyamary's Assistant. "
            "If the question is a greeting like hi/hello, just say: "
            "\"Hello! I'm Arokyamary's Assistant. How can I help you today?\" — nothing else. "
            "For business questions use this data:\n{context}\n\n"
            "For general questions answer helpfully without mentioning the business data. "
            "Question: {question}\nAnswer:"
        )
    )

    def ask(question):
        docs = retriever.invoke(question)
        context = '\n'.join([d.page_content for d in docs])
        chain = pr | llm | StrOutputParser()
        return chain.invoke({'context': context, 'question': question})

    return ask

ask = load_chain()

st.sidebar.header('Sample Questions')
sample_qs = [
    'Which city had the lowest revenue?',
    'Bengaluru Electronics return rate March 2024?',
    'Product with most complaints?',
    'Compare Mumbai vs Delhi Electronics revenue',
]
for s in sample_qs:
    if st.sidebar.button(s, use_container_width=True):
        st.session_state['auto_q'] = s

if 'messages' not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    avatar = '👩‍💼' if m['role'] == 'assistant' else None
    with st.chat_message(m['role'] if m['role'] == 'user' else 'Arokyamary', avatar=avatar):
        st.write(m['content'])

if 'auto_q' in st.session_state:
    q = st.session_state.pop('auto_q')
    st.session_state.messages.append({'role': 'user', 'content': q})
    with st.chat_message('user'):
        st.write(q)
    with st.chat_message('Arokyamary', avatar='👩‍💼'):
        with st.spinner('Searching your data...'):
            ans = ask(q)
        st.write(ans)
    st.session_state.messages.append({'role': 'assistant', 'content': ans})
    st.rerun()

if q := st.chat_input('Ask me anything...'):
    st.session_state.messages.append({'role': 'user', 'content': q})
    with st.chat_message('user'):
        st.write(q)
    with st.chat_message('Arokyamary', avatar='👩‍💼'):
        with st.spinner('Analysing...'):
            ans = ask(q)
        st.write(ans)
    st.session_state.messages.append({'role': 'assistant', 'content': ans})
    st.rerun()