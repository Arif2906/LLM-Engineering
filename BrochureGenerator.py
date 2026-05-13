from tarfile import LinkOutsideDestinationError
from Helpers import UtilFunc
import os
from openai import OpenAI
import json
from IPython.display import Markdown, display, update_display

def get_relevent_links(url):
    
    links=UtilFunc.extract_useful_links(url)
    system_prompt="""You are a provided a list of links thats are available on a website.
useful links are such as about us, contact us, services, products, etc. Do not extract links that are not useful.
you should be able to decide which are useful for a brochure.
you should be able to return the output in a json format with the following keys:
{
    "links" : [
        {"type":"about page","url":"https://edwarddonner.com/about"}
        {"type":"contact page","url":"https://edwarddonner.com/contact"}
        {"type":"services page","url":"https://edwarddonner.com/services"}
    ]
}
"""
    user_prompt= """ You are provided few links 
you have to decide which are useful for a brochure.
below are the links
"""
    user_prompt += "\n".join(links)

    op=OpenAI(base_url=UtilFunc.OLLAMA_BASE_URL, api_key='')
    res=op.chat.completions.create(model="llama3",
    messages=[{"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt}],
    response_format={"type": "json_object"})
    result = res.choices[0].message.content
    links= json.loads(result)
    print(links)
    return links

def createBrochure(campanyname,url):
    brochure_system_prompt=""" You are assistant that analyses the webpage content of company and their various webpage
    link and create a brocher for future customer prospect and investor and recruit.
    Respond in markdown without code blocks
    """
    brochure_user_prompt=f"""Here is the company name {campanyname} and below is webpage content and relevant links for the
    company, create a brocher
    """
    content =UtilFunc.extract_content(url)
    brochure_user_prompt += content
    links= get_relevent_links(url)
    print(links)
    for link in links["links"]:
        brochure_user_prompt += f'\n This is Page {link['type']} and content to this page is {UtilFunc.extract_content(link['url'])}'
    print(brochure_user_prompt)
    op=OpenAI(base_url=UtilFunc.OLLAMA_BASE_URL, api_key='')
    response=op.chat.completions.create(model="llama3",
    messages=[{"role": "system", "content": brochure_system_prompt},
    {"role": "user", "content": brochure_user_prompt}])
    result = response.choices[0].message.content
    print(result)
    display(Markdown(result))
createBrochure("HuggingFace","https://huggingface.co")

# response=UtilFunc.chat_completion(UtilFunc.get_client('OLLAMA'), user_prompt, system_prompt)
#print(response.choices[0].message.content)






