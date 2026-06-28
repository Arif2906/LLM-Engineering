#this code will convert python code to c++ code
import os
from openai import OpenAI 
import requests 
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
router_api_key=os.getenv("OpenRouter")
open_router= OpenAI(api_key=router_api_key, base_url='https://openrouter.ai/api/v1')
models = { "nvidia": "nvidia/nemotron-3-super-120b-a12b:free", "gpt": "openai/gpt-oss-120b:free"}
def generate_code(text,selected):
    message= [{"role": "user", "content": "convert this python code to c++ code and return only c++ code" + text}]
    result= open_router.chat.completions.create(model= models[selected], messages=message, stream=False)
    return result.choices[0].message.content

with gr.Blocks() as ui:
    with gr.Row():
            input = gr.Textbox(lines=20, info='enter your python code here')
            cpp = gr.Textbox(lines=20, info='your c++ code here')
    with gr.Row():
            selected_model= gr.Dropdown(["nvidia", "gpt"], label='select model')
            btn= gr.Button('convert code')
    btn.click(fn=generate_code, inputs=[input, selected_model], outputs= [cpp])

ui.launch()