from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
from tokenizers import Tokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os

load_dotenv()
login(os.getenv("HF_Token"),add_to_git_credential=True)
def load_all_split_dataset():
    train, test, val = load_dataset("frndarif055/product_prices_raw_full", trust_remote_code=True, split=["train","validation","test"])
    return train, test, val

#prepare prompt and completion for finetuning
def make_prompt(item):
    max_token_length = 110
    prompt="what does this cost to nearest dollars: "+item["title"]
    prefix = 'the dollar amount is: '
    auto_tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
    token= auto_tokenizer.encode(item["title"])
    if len(token) > max_token_length:
        token = token[:max_token_length]
        summary = auto_tokenizer.decode(token)
    else:
        summary = auto_tokenizer.decode(token)
    prompt = prompt + "\n " + summary + "\n" + prefix
    completion = f"{item['price']}"
    return prompt, completion

def push_prompt_and_completion_tohub():
    prompts = []
    completions = []
    train, test, val = load_all_split_dataset()
    for train, val, test in zip(train, val, test):

        trainprompt, traincompletion = make_prompt(train)
        prompts.append({"train":trainprompt})
        completions.append({"train":traincompletion})
        valprompt, valcompletion = make_prompt(val)
        prompts.append({"val":valprompt})
        completions.append({"val":valcompletion})

        testprompt, testcompletion = make_prompt(test)
        prompts.append({"test":testprompt})
        completions.append({"test":testcompletion})

    dataset = DatasetDict({
        'train': Dataset.from_dict({'prompt': prompts['train'], 'completion': completions['train']}),
        'test': Dataset.from_dict({'prompt': prompts['test'], 'completion': completions['test']}),
        'val': Dataset.from_dict({'prompt': prompts['val'], 'completion': completions['val']})
    })
    dataset.push_to_hub("frndarif055/product_prompt_prices_raw_full")


push_prompt_and_completion_tohub()