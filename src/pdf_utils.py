import os
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

def process_pdf(pdf):
    """
    Process a single PDF file: save to temp, load, split, and clean up.
    """
    temp_pdf = f'./temp_{pdf.name}'
    with open(temp_pdf, 'wb') as f:
        f.write(pdf.getvalue())
    
    try:
        pdf_loader = PyPDFLoader(temp_pdf)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        docs = pdf_loader.load_and_split(text_splitter=text_splitter)
    finally:
        if os.path.exists(temp_pdf):
            os.remove(temp_pdf)
            
    return docs

@st.cache_data
def load_and_process_pdfs(pdf_files):
    """
    Process multiple PDF files in parallel.
    """
    documents = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(process_pdf, pdf_files)
        for r in results:
            documents.extend(r)
    return documents
