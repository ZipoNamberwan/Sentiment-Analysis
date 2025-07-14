import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def remove_symbols(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Remove symbols and numbers
    return text

def to_lowercase(text):
    return text.lower()

def tokenize(text):
    return text.split()

factory = StopWordRemoverFactory()
stopwords = set(factory.get_stop_words())

def remove_stopwords(tokens):
    return [word for word in tokens if word not in stopwords]

stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

def stem_words(tokens):
    return [stemmer.stem(word) for word in tokens]

def preprocess_text(text):
    text = remove_symbols(text)
    text = to_lowercase(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = stem_words(tokens)
    return " ".join(tokens) 


