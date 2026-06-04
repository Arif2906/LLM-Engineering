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

rf_model= RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=4)
count_vectorizer = CountVectorizer(max_features=1000, stop_words='english')

train, test, val = load_all_split_dataset()
subset= 15000
x_train = count_vectorizer.fit_transform([item["summary"] for item in train])
y_train = np.array([item["price"] for item in train])
rf_model.fit(x_train[:subset], y_train[:subset])