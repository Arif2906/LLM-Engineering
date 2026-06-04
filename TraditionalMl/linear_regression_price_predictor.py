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
from sklearn.feature_extraction.text import CountVectorizer

load_dotenv()
login(os.getenv("HF_TOKEN"),add_to_git_credential=True)

def load_all_split_dataset():
    train, test, val = load_dataset("ed-donner/items_lite", trust_remote_code=True, split=["train","validation","test"])
    return train, test, val

def get_feature(item):
    return {
        "weight": item["weight"],
        "unknown_weight": 0 if item["weight"] else 1,
        "summary_length": len(item["summary"]),
    }
def convert_dict_to_dataFrame(items):
    feature_item= [get_feature(item) for item in items]
    df = pd.DataFrame(feature_item)
    df["price"] = [item["price"] for item in items]
    return df
def predict_price_linear_regression():
    train, test, val = load_all_split_dataset()
    X_train = convert_dict_to_dataFrame(train)
    y_train = X_train["price"]

    X_test = convert_dict_to_dataFrame(test)
    y_test = X_test["price"]
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return X_test,y_pred


#bag of words
def bag_of_words():
    train, test, val = load_all_split_dataset()
    X_train = convert_dict_to_dataFrame(train)
    count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    count_vectorizer.fit_transform(X_train['summary'])
    frequencies = count_vectorizer.get_feature_names()
    return frequencies

print(bag_of_words())

