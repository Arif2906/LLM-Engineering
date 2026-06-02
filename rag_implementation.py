import os
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import glob as gl
import numpy as np


full_knowledge_base_path='knowledge-base/**/*.md'

def create_full_knowledgebase(full_knowledge_base_path):
    full_knowledge_base=''
    files= gl.glob(full_knowledge_base_path)
    for file_path in files:
        with open(file_path,'r', encoding='utf-8') as f:
            full_knowledge_base += f.read()
            full_knowledge_base +='\n'
    return full_knowledge_base
    

def total_token(knowledge_base):
    encoding= tiktoken.encoding_for_model("gpt-4o")
    return len(encoding.encode(knowledge_base))

def load_in_langchain():
    knowledge=[]
    all_folder='knowledge-base/*'
    folders= gl.glob(all_folder)
    for folder in folders:
        doc_type=os.path.basename(folder)
        files= DirectoryLoader(path=folder, glob='**/*.md',loader_cls= TextLoader ,loader_kwargs={'encoding': 'utf-8'}).load()
        for f in files:
            f.metadata['doc-type']=doc_type
            knowledge.append(f)
    return knowledge

def chunk_document():
    
    kb=load_in_langchain()
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks=text_splitter.split_documents(kb)
    return chunks

#print(chunk_document())
def get_embeddings():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 
    db_name= 'chroma_db'
    if os.path.exists(db_name):
        db = Chroma(persist_directory=db_name,embedding_function=embeddings).delete_collection()
    chunk=chunk_document()
    vector_store= Chroma.from_documents(chunks=chunk, embedding= embeddings,persist_directory=db_name)
    collection=vector_store._collection
    return collection

def visualize_embedding():
    
    collection = get_embeddings()
    result = collection.get(include=['embeddings', 'documents', 'metadatas'])
    vector=np.array(result['embeddings'])
    tsne = TSNE(n_components=2, random_state=42)
    low_dim_embs = tsne.fit_transform(vector)
    fig= go.Figure()
    fig.add_trace(go.Scatter(x=low_dim_embs[:,0], y=low_dim_embs[:,1], mode='markers'))
    fig.update_layout(title='2D t-SNE Visualization', xaxis_title='Dimension 1', yaxis_title='Dimension 2')
    fig.show()

visualize_embedding()