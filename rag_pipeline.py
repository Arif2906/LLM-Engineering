from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
import os
from typing import Optional

load_dotenv()
router_api_key=os.getenv("OpenRouter")



#create llm invoker and vector store invoker
llm= ChatOpenAI(api_key=router_api_key,base_url='https://openrouter.ai/api/v1',model='nvidia/nemotron-3-super-120b-a12b:free'
,temperature=0.2,max_tokens=1000)
embeddings= HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vector_store= Chroma(persist_directory='chroma_db',embedding_function=embeddings)

knowledge_base=vector_store.as_retriever()
def get_whole_question_with_history (question: str,history:Optional[list[dict]]=None)-> str:
    """
    Get the whole question with history
    
    """
    history = history or []
    question_with_history= "\n ".join([h["content"] for h in history if h["role"] == "user"]) + " " + question
    return question_with_history


def answer_question(question, history:list[dict]=None):
    system_prompt="""you are a helpful assistant and you will answer questions about Insurellm and answer in short manner
    You are provided with additional context that might be relevant to the user's question.
    Give brief, accurate answers. If you don't know the answer, say so.
    context: {context}"""
    history= history or []
    combined_question= get_whole_question_with_history(question,history)
    relevent_context= knowledge_base.invoke(combined_question)
    relevent_content= ''.join(relevet_context.page_content for relevet_context in relevent_context)
    system_prompt = system_prompt.format(context=relevent_content)
    message= [SystemMessage(content=system_prompt)]
    message.extend(convert_to_messages(history))
    message.append(HumanMessage(content=question))  
    response=llm.invoke(message)
    return response.content, relevent_context



iface= gr.Interface(fn=answer_question,
                    inputs=[gr.Textbox()],
                    outputs=gr.Textbox())
iface.launch()