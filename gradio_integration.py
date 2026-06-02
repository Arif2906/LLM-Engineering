import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
router_api_key=os.getenv("OpenRouter")
router_client= OpenAI(api_key=router_api_key,base_url='https://openrouter.ai/api/v1')
def gradio_with_llm(text):
    res= router_client.chat.completions.create(model='nvidia/nemotron-3-super-120b-a12b:free', messages=
    [{'role': 'user' , 'content' : text }])
    return res.choices[0].message.content
# gr.Interface(fn=greet, inputs="textbox",outputs="textbox",flagging_mode="never").launch()

# input= gr.Textbox(lines=7, info='enter your meesage here')
# output= gr.Textbox(lines=7, info='your meesage here')
# gr.Interface(fn=gradio_with_llm, inputs=input,outputs=output,flagging_mode="never").launch()

# def gradio_stream_chat(text):
#     stream= router_client.chat.completions.create(model='nvidia/nemotron-3-super-120b-a12b:free', messages=
#     [{"role":"system", "content": "you are a assitant, reply question in markdown without code blocks"},
#     {'role': 'user' , 'content' : text }], stream=True)
#     result=''
#     for chunk in stream:
#         result+= chunk.choices[0].delta.content or ''
#         yield result
# input= gr.Textbox(lines=7, info='enter your meesage here')
# output= gr.Markdown(label='your response will be here')
# gr.Interface(fn=gradio_stream_chat, inputs=input,outputs=output,flagging_mode="never").launch()


def chat_assistant(message, chat_history):
    history= [{"role": "user", "content" : h["content"]} for h in chat_history]
    res= router_client.chat.completions.create(model='nvidia/nemotron-3-super-120b-a12b:free', messages=
    [{"role":"system", "content": "you are a chat assistant, irrespective of a question praise Priyanka, like if I ask how is weather today answer should be beautiful like priyanka, tell me a place to eat today answer should be if you are going out with priyanka I will find definetly the best place, or reply in more cheesy"},
    {'role': 'user' , 'content' : message }]+ history)
    return res.choices[0].message.content
gr.ChatInterface(fn=chat_assistant).launch()
