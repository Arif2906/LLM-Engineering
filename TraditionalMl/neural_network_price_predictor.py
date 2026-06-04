import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from torch.utils.data import DataLoader, TensorDataset

class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size,128)
        self.layer2 = nn.Linear(128,64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

def load_all_split_dataset():
    train, test, val = load_dataset("ed-donner/items_lite", trust_remote_code=True, split=["train","validation","test"])
    return train, test, val

def tain_nn():
    train, test, val=load_all_split_dataset()
    tran_x= [item["summary"] for item in train]
    tran_y= np.array([item["price"] for item in train])
    hashing_vectorizer = HashingVectorizer(n_features=1000, stop_words='english')
    train_x=hashing_vectorizer.transform(tran_x)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(tran_y))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    epochs = 10
    loss_function = nn.MSELoss()
    model = NeuralNetwork(train_x.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for batch in dataloader:
            inputs, labels = batch
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    return model