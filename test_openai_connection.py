import dotenv
from openai import OpenAI
import os

#IF any issue in importing library make sure your interpreter is poiting to this venv where you just installed the lib

dotenv.load_dotenv()
api_key= os.getenv("OPENAI_API_KEY")
if api_key== '':
    print ('key is empty')

client= OpenAI(api_key=api_key)
message='this is my first message to gemini LLM'
#below is specific format which model take its list of dict
messages = [{"role":"user","content": message}]
chat_with_system_prompt = [{"role":"system","content": "You are a witty and playful assistant. Answer questions with humor, clever twists, and lighthearted jokes, while still being helpful and clear. Sprinkle in puns or funny metaphors when appropriate, but never confuse humor with facts"},
                           {"role":"user","content": message}]
response= client.chat.completions.create(model="gemini-3-flash-preview",messages=messages)
print(response.text)
curated_response= client.chat.completions.create(model="gemini-3-flash-preview",messages=chat_with_system_prompt)
print(response.text)
