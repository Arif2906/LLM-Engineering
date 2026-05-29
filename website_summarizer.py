from Helpers import UtilFunc
from openai import OpenAI
from dotenv import load_dotenv
import os

website_content= UtilFunc.extract_content("https://edwarddonner.com")
# creating client 
# client=UtilFunc.get_client("OpenAI")
# response =UtilFunc.chat_completion(client,website_content,"You are a witty and playful assistant. Answer questions with humor")
# print(website_content)
# print(response.choices[0].message.content)

# client=UtilFunc.get_client("OLLAMA")
# response =UtilFunc.chat_completion(client,website_content,"You are a witty and playful assistant. Answer questions with humor")
# print(website_content)
# print(response.choices[0].message.content)

response=UtilFunc.http_call_to_llm()
print(response.get("message",{}).get("content",{}))
