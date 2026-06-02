import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import glob

load_dotenv()
router_key=os.getenv("OpenRouter")
client = OpenAI(api_key=router_key, base_url='https://openrouter.ai/api/v1')

knowledge_dict = {}

def get_files(folder_path):
    files = glob.glob(f'knowledge-base/{folder_path}/*')
    return files

def get_knowledge(folder_path):
    
    files=get_files(folder_path)
    for filename in files:
        with open(filename, 'r', encoding='utf-8') as f:
            knowledge_dict[Path(filename).stem.split(" ")[0].lower()]=f.read()
    return knowledge_dict

def get_context_from_knowledge(question):
    question = ''.join(ch for ch in question if ch.isalnum() or ch == ' ').lower()
    print(question)
    whole_context = " ".join([knowledge_dict.get(words, "No Additional Context") for words in question.split(" ")])
    return whole_context

system_prompt="""
you are a helpful assistant and you will answer questions about Insurellm and answer in shport manner
You are provided with additional context that might be relevant to the user's question.
Give brief, accurate answers. If you don't know the answer, say so.

Relevant context:
"""

def chat(question,chat_history):
    print(get_context_from_knowledge(question))
    system_prefix = system_prompt + get_context_from_knowledge(question)
    history= [{"role": "user", "content" : h["content"]} for h in chat_history]
    messages = [{"role": "system", "content": system_prefix}] + history+ [{'role': 'user' , 'content' : question}]
    res= client.chat.completions.create(model='nvidia/nemotron-3-super-120b-a12b:free', messages=messages)
    return res.choices[0].message.content

print(chat("who is james wilson",[]))