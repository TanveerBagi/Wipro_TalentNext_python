#Q1. Perform Text Preprocessing on SMSSpamCollection Dataset.
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
print(df.head())

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

df["cleaned_message"] = df["message"].apply(preprocess_text)
print(df[["message", "cleaned_message"]].head())

