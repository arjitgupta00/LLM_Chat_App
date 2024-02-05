import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.llms.openai import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
#from langchain.chat_models import C
import pickle
import os

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

load_dotenv()

def main():
    st.header('Chat with PDF!')

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

        store_name = pdf_file.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as file:
                vector_store = pickle.load(file)
            st.write("Embeddings loaded from the Disk")
        else:
            embeddings_obj = OpenAIEmbeddings()
            vector_store = FAISS.from_texts(chunks, embedding=embeddings_obj)
            with open (f"{store_name}.pkl", "wb") as file:
                pickle.dump(vector_store, file)
        # st.write(chunks)
                
        # Accept user questions/query
        query = st.text_input('Ask questions about your pdf file.')
        st.write('query')

        if query:
            docs = vector_store.similarity_search(query=query, k=3)

            #llm = OpenAI(temperature=0)
            llm = OpenAI(model_name="gpt-3.5-turbo")
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == '__main__':
    main()