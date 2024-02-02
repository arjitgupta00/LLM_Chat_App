import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import faiss
import pickle

#Sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown ('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain] (https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM model
    ''')
    add_vertical_space (5)
    st.write('(https://youtube.com/@engineerprompt)')


def main():
    st.write('Hello')

    pdf_file = st.file_uploader("Upload your file", type='pdf')
    
    
    if pdf_file is not None:
        st.write(pdf_file.name)
        pdf_reader = PdfReader(pdf_file)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )

        chunks = text_splitter.split_text(text=text)


        embeddings_obj = OpenAIEmbeddings()

        vector_store = faiss.from_texts(chunks, embeddings_obj)

        store_name = pdf_file.name[:-4]
        with open (f"{store_name}.pkl", "wb") as file:
            pickle.dump(vector_store, f)
        # st.write(chunks)


if __name__ == '__main__':
    main()