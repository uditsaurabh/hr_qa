import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.document import Document

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from io import StringIO


def open_ai_api_key():
    st.title("Openai API key")
    password = st.text_input("Enter Openai api key:", type="password")
    return password


def create_llm(api_key):
    return OpenAI(openai_api_key=api_key)


def create_template_and_prompt():
    template = """
    You are giving reply to an HR of a big IT firm for 
    selection of a candidate.
    You need to be very specific with your answers.
    Your questions is :{question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    return prompt


def get_file():
    # Input
    st.markdown("## Upload the resume you want to ask questions from.")
    uploaded_file = st.file_uploader("Choose a file", type="txt")
    return uploaded_file


def get_document_from_file(file):
    bytes_data = file.getvalue()
    string_data = bytes_data.decode("utf-8")
    return string_data


def split_the_documents(text):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs


def create_embeddings(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings


def create_load_database(document_chunks, embeddings):
    stored_embeddings = FAISS.from_documents(document_chunks, embeddings)
    return stored_embeddings


def create_retrieval_chain(llm, stored_embeddings):
    QA_Chain = RetrievalQA.from_chain_type(
        chain_type="stuff", llm=llm, retriever=stored_embeddings.as_retriever()
    )
    return QA_Chain


def get_question():
    question = st.text_input("Enter your question:", type="default")
    return question


def answer_question(response):
    return st.write(response)


def main():

    api_key = open_ai_api_key()
    file = get_file()
    question = get_question()

    if api_key and file and question:
        document = get_document_from_file(file)
        doc_chunks = split_the_documents(document)
        embeddings = create_embeddings(api_key)
        if doc_chunks and embeddings:
            stored_embeddings = create_load_database(doc_chunks, embeddings)
            if (llm := create_llm(api_key)) and stored_embeddings:
                if chain := create_retrieval_chain(llm, stored_embeddings):
                    with st.spinner("Wait, please. I am working on it..."):
                        response = chain.run(question)
                        answer_question(response)
                        del api_key


if __name__ == "__main__":
    main()
