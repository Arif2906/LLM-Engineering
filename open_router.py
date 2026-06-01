from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
router_api_key=os.getenv("OpenRouter")
router_client= OpenAI(api_key=router_api_key, base_url='https://openrouter.ai/api/v1')
response = router_client.chat.completions.create(
    model='nvidia/nemotron-3-super-120b-a12b:free',
    messages=[{"role": "user", "content": "two coins flipped if one gets heads what is the probablity other coin will gets tails"}],
)
print(response)