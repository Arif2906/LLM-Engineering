from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from tokenizers import Tokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os

hf_token = os.getenv('HF_TOKEN')
login(hf_token, add_to_git_credential=True)
def load_all_split_dataset():
    train, val, test = load_dataset("frndarif055/product_prompts_raw_full", trust_remote_code=True, split=["train","val","test"])
    return train, val, test

bits_and_bytes_config = BitsAndBytesConfig(bnb_4bit_use_double_quant=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
model_path = "microsoft/Phi-4-mini-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=bits_and_bytes_config
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

def predict_price(prompt):

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=8)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

train, val, test = load_all_split_dataset()
predicted_price = predict_price(test[0]["prompt"])
print(predicted_price)