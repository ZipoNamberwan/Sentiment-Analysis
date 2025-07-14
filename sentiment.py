import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Step 1: Load preprocessed CSV ---
df = pd.read_csv("preprocessed/taman bunga celosia.csv")

# Make sure 'cleaned_caption' column exists
if "cleaned_caption" not in df.columns:
    raise Exception("❌ Column 'cleaned_caption' not found in the CSV file.")

# --- Step 2: Load tokenizer and model ---
model_name = "mdhugol/indonesia-bert-sentiment-classification"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Label mapping based on model documentation
label_index = {
    'LABEL_0': 'positive',
    'LABEL_1': 'neutral',
    'LABEL_2': 'negative'
}
    
# --- Step 3: Define safe sentiment prediction function ---
def predict_sentiment(text):
    try:
        text = str(text).strip()
        if not text:
            return "unknown"

        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs).item()

        # Convert predicted class (int) to label like 'LABEL_0'
        label_key = f'LABEL_{predicted_class}'
        return label_index.get(label_key, "unknown")

    except Exception as e:
        print(f"⚠️ Error processing: {text} → {e}")
        return "error"

# --- Step 4: Apply prediction safely ---
df["sentiment"] = df["cleaned_caption"].apply(predict_sentiment)

# --- Step 5: Save result ---
df.to_csv("result/preprocessed_reviews_with_sentiment.csv", index=False)
print("✅ Sentiment analysis complete. File saved as 'preprocessed_reviews_with_sentiment.csv'")
