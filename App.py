# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 15:36:36 2024

@author: F8089899
"""

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
import pickle
from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    ## Sobre
    Esse aplicativo é um chatbot fornecido pelo poder da LLM utilizando:
        
    - [Streamlit](https://streamlit.io)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) modelo LLM
    
    ''')
    add_vertical_space(5)
    st.write('Feito com Amor s2!')

def main():
    st.header("Chat with PDF 💬")
    
    # carrega .env
    load_dotenv()
    #upload pdf
    pdf = st.file_uploader("Carregue seu arquivo PDF", type = "pdf")
    
    #st.write(pdf)
    if pdf is not None:
        #instanciando o pdf_file'
        pdf_reader = PdfReader(pdf)
        st.write(pdf.name)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
            )
        chunks = text_splitter.split_text(text=text)
        
        # utilizando o embeddings do OPENAI para formação das palavras
        
        store_name = pdf.name[:-4]
        
        if os.path.exists(f'{store_name}.pkl'):
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
            #st.write('Embeddings carregado do disco')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f: 
                pickle.dump(VectorStore, f)
            #st.write('Embeddings carregado da base de dados!')
        
        # Criando o chat para perguntas
        st.header("Faça sua pergunta")
        query = st.text_input("Faça sua pergunta sobre o arquivo PDF:")
        #st.write(query)
        
        if query:    
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name = "gpt-3.5-turbo",temperature=0) # carrega modelo
            chain = load_qa_chain(llm = llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documenrs=docs, question=query)
                print(cb)
            st.write(response)
        
        #st.write('Embeddings computados')
        
        
        #st.write(chunks)
            
if __name__ == '__main__':
    main()
    
    