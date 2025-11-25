import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError 

splits = {"train": "yelp_review_full/train-00000-of-00001.parquet"}
train_path = "hf://datasets/yelp/yelp_review_full/" + splits["train"]

df = pd.read_parquet(train_path)
print(df.head())

df["label"] = df["label"] + 1

texts = df["text"].values
labels = df["label"].values

tokenizer = Tokenizer(num_words = 10000, oov = "OOV>")
tokenizer.fit_on_texts(texts)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)