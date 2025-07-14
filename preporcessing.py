
# Import regular expressions and Sastrawi libraries for stopword removal and stemming
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# Remove URLs, symbols, and numbers from text
def remove_symbols(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Remove symbols and numbers
    return text


# Convert text to lowercase
def to_lowercase(text):
    return text.lower()


# Split text into tokens (words)
def tokenize(text):
    return text.split()


# Initialize stopword remover
factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())


# Remove stopwords from token list
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

# Initialize stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()


# Stem each word in the token list
def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]


# Full preprocessing pipeline: clean, lowercase, tokenize, remove stopwords, and stem
def preprocess_text(text):
    text = remove_symbols(text)
    text = to_lowercase(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens)


import pandas as pd

# Load your data
df = pd.read_excel("filtered/taman bunga celosia_filtered.xlsx")
df["cleaned_caption"] = df["caption"].apply(preprocess_text)
# Show result
print(df[["caption", "cleaned_caption"]].head())
df.to_csv("preprocessed/taman bunga celosia.csv", index=False)

