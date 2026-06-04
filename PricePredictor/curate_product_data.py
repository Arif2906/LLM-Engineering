from datasets import load_dataset
from huggingface_hub import login
from dotenv import load_dotenv
import os
from item import Item
import matplotlib.pyplot as plt
from item import ItemLoader
import numpy as np
from collections import Counter

load_dotenv()

login(os.environ['HF_TOKEN'],add_to_git_credential=True)

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_Appliances", trust_remote_code=True, split="full")

items= [item for data in dataset if (item := Item.parse_data(data,"Appliances")) is not None]
print(len(items))
gifts_item=ItemLoader("Gift_Cards").load_multiple()
items.extend(gifts_item)
print(len(items))


#penalise data with low price
#get sample of 20k items
prices=np.array([item.price for item in items])
category=np.array([item.category for item in items])
p = (prices - prices.min()) / (prices.max() - prices.min() + 1e-9)
w = p**2
w[category == "Gift_Cards"] *= 0.5
w=w/np.sum(w)
sample=np.random.choice(items, size=20000, p=w)

prices_in_sample = [item.price for item in sample]

fig = plt.figure()
plt.title( f"Price Distribution avg price :{np.mean(prices_in_sample)} high price :{np.max(prices_in_sample)} low price :{np.min(prices_in_sample)}")
plt.xlabel("Price")
plt.ylabel("Count")
plt.hist(prices_in_sample, bins=20)
plt.show()


count_mapping = Counter([item.category for item in sample])
category_keys = count_mapping.keys()
count_per_category = count_mapping.values()
fig = plt.figure()
plt.title( f"Category Distribution avg price :{np.mean(prices_in_sample)} high price :{np.max(prices_in_sample)} low price :{np.min(prices_in_sample)}")
plt.xlabel("Category")
plt.ylabel("Count")
plt.bar(category_keys, count_per_category)
plt.show()

#push to huggingface

train_dataset = sample[:18000]
validation_dataset = sample[18000:19000]
test_dataset = sample[19000:]

username = "frndarif055"
full = f"{username}/product_prices_raw_full"

Item.push_to_hub(full,train_dataset,validation_dataset,test_dataset)   

