from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
from chromadb import PersistentClient
from pydantic import BaseModel, Field
import glob
import numpy as np
import json
from litellm import completion
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def llm_client():

    router_api_key=os.getenv("OpenRouter")
    router_client= ChatOpenAI(api_key=router_api_key, base_url='https://openrouter.ai/api/v1', model='nvidia/nemotron-3-super-120b-a12b:free')
    return router_client

class Result(BaseModel):
    page_content: str
    metadata: dict
class Rerank(BaseModel):
    order: list[int] = Field(description="The order of the chunks")

class Chunk(BaseModel):
    headline: str = Field(description="The title of the chunk")
    original_text: str = Field(description="The whole content of the chunk")
    summary: str = Field(description="The summary of the chunk")

    def to_result(self, document):
        return Result(
            page_content= self.headline + "\n\n"  + self.summary + "\n\n" + self.original_text ,
            metadata={"source": document["source"], "type": document["type"]},
        )
class Chunks(BaseModel):
    chunks: list[Chunk]
def load_document():
    
    docs= []   
    knowledge_base = Path("knowledge-base")
    folder = knowledge_base.iterdir()

    for subfolder in folder:
        doc_type=subfolder.name
        for file in subfolder.glob("**/*.md"):
            with file.open("r", encoding="utf-8") as f:
                docs.append({"type": doc_type, "source": file.as_posix(), "text": f.read()})
    return docs


def chunk_documents_using_llm(document):
    how_many=len(document["text"])/500
    prompt=f"""
    You take a document and you split the document into overlapping chunks for a KnowledgeBase.

The document is from the shared drive of a company called Insurellm.
The document is of type: {document["type"]}
The document has been retrieved from: {document["source"]}

A chatbot will use these chunks to answer questions about the company.
You should divide up the document as you see fit, being sure that the entire document is returned in the chunks - don't leave anything out.
This document should probably be split into {how_many} chunks, but you can have more or less as appropriate.
There should be overlap between the chunks as appropriate; typically about 25% overlap or about 50 words, so you have the same text in multiple chunks for best retrieval results.

For each chunk, you should provide a headline, a summary, and the original text of the chunk.
Together your chunks should represent the entire document with overlap.

Here is the document:

{document["text"]}

Respond with the chunks.

"""
    result=llm_client().invoke([HumanMessage(content=prompt)], response_format=Chunks)
    result=Chunks.model_validate_json(result.content).chunks
    return [chunk.to_result(document) for chunk in result]

def create_chunks(docs):
    
    chunks = chunk_documents_using_llm(docs)
    #not doing for all chunk right now
    # chunks= []
    # for doc in docs:
    #     chunks.append(chunk_documents_using_llm(doc))
    # return chunks


def create_embedding(chunks):    
    
    chrome_db= PersistentClient(path="chroma_db")
    if "docs" in [name for name in chrome_db.list_collections()]:
        chrome_db.delete_collection("docs")
    
    texts=[chunk.page_content for chunk in chunks]
    hugging_face_embeddings= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings= hugging_face_embeddings.encode(texts).tolist()
    vextors = [e.embedding for e in embeddings]
    collection=chrome_db.get_or_create_collection("docs")
    collection.add(documents=texts, embeddings=vextors,metadatas=[chunk.metadata for chunk in chunks], 
                   ids=[str(i) for i in range(len(chunks))])
    return collection
    
def retrieve_context(question, collection):
    embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').encode(question).tolist()
    result= collection.query(
        query_embeddings=[embedding],
        n_results=3 ,
    )
    chunks=[]
    for chunk in zip(result["documents"][0], result["metadata"][0]):
        chunks.append(Result(page_content=chunk[0], metadata=chunk[1]))
    return chunks

def rerank_chunks(question, chunks):
    system_prompt="""You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt_with_chunks= "This is user question: " + question + "\n\n and reply with the order of the chunks" + "\n".join([f"Chunk {i}: {chunk.page_content}" for i, chunk in enumerate(chunks)])  

    result=llm_client().invoke(HumanMessage(content=user_prompt_with_chunks), system_message=SystemMessage(content=system_prompt)  , response_format=Rerank)
    ordereding_of_chunks= Rerank.model_validate_json(result.choices[0].message.content).order
    return [chunks[order-1] for order in ordereding_of_chunks]

    
    

docs= load_document()
print(docs[0])
chunks=create_chunks(docs[0])
print(chunks)
collection=create_embedding(chunks)
print(collection)
#chunks=retrieve_context("what is my name?",collection)
#print(chunks)
#chunks=rerank_chunks("what is my name?",chunks)
#print(chunks)
