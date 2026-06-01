from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
router_api_key=os.getenv("OpenRouter")
router_client= OpenAI(api_key=router_api_key, base_url='https://openrouter.ai/api/v1')

llm1_system="""You are a chatbot that reply everything very trickly consider youself as girlfriend who is angry at a moment
reply on everything in irritated manner"""

llm2_system="""You are a chatbot that reply everything very politely consider youself as boyfriend who is trying to makeup with
his girlfriend at a moment reply on everything in loving manner"""

llm1message=['hi there']
llm2meesage=['hi']

def chatllm1():
    message= [{"role" : "system", "content": llm1_system}]
    for mes1, mes2 in zip(llm1message,llm2meesage):
        message.append({"role": "user", "content": llm2meesage})
        message.append({"role": "assistant", "content": llm1message})

    response=router_client.chat.completions.create(model="nvidia/nemotron-3-super-120b-a12b:free",
    messages=message)
    return response.choices[0].message.content

def chatllm2():
    message= [{"role" : "system", "content": llm2_system}]
    for mes1, mes2 in zip(llm1message,llm2meesage):
        message.append({"role": "user", "content": llm1message})
        message.append({"role": "assistant", "content": llm2meesage})

    response=router_client.chat.completions.create(model="arcee-ai/trinity-large-preview:free",
    messages=message)
    return response.choices[0].message.content

def chatbetweenLLM ():
    for i in range(5):
        llm1resp=chatllm1();
        llm2resp=chatllm2();
        llm1message.append(llm1resp)
        llm2meesage.append(llm2resp)
        print(f'this is "llm1: " : {llm1resp}')
        print(f'\n')
        print(f'this is "llm2: " {llm2resp}')
if __name__=="__main__":
    chatbetweenLLM()