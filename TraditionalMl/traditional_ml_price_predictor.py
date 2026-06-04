import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
from huggingface_hub import login
from datasets import load_dataset
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt


load_dotenv()
login(os.getenv("HF_TOKEN"),add_to_git_credential=True)

def load_all_split_dataset():
    
    train, test, val = load_dataset("ed-donner/items_lite", trust_remote_code=True, split=["train","validation","test"])
    return train, test, val

def random_pricer(test):    
    result = []
    for i in range(len(test)):
        result.append({"actual_price":test["price"][i], "predicted_price":np.random.randint(1,100)})
    return result

def plot_charts(true_price,predicted_price):
    
    length = range(len(true_price))
    
    plt.scatter(length,true_price,color="red",label='actual_price')
    plt.scatter(length,predicted_price,color="blue",label='predicted_price')
    plt.legend()
    plt.xlabel("true price")
    plt.ylabel("predicted price")
    plt.show()

train, test, val = load_all_split_dataset()

result = random_pricer(test)
actual_price = [item["actual_price"] for item in result
                ]
predicted_price = [item["predicted_price"] for item in result
                ]
plot_charts(actual_price,predicted_price)    




    


