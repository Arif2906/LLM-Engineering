from openai import OpenAI
from Helpers import UtilFunc

client= OpenAI(base_url=UtilFunc.OLLAMA_BASE_URL, api_key='')
Techical_system_prompt="""you are a techinical expert of math of or programming language
answer the posted question in most simple way possible do not over explain things keep it simple """
user_input= input('This is your question :')

response= client.chat.completions.create(model='llama3', messages= [{"role":"system","content": Techical_system_prompt},
{"role": "user", "content": user_input}])
print(response.choices[0].message.content)
