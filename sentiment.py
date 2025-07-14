import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class SentimentAnalyzer:
    """
    A class to perform sentiment analysis on preprocessed text data.
    Purpose: Generate Excel files with sentiment predictions.
    """
    
    def __init__(self, model_name: str = "mdhugol/indonesia-bert-sentiment-classification", output_folder: str = "result"):
        """
        Initialize the SentimentAnalyzer.
        
        Args:
            model_name (str): Hugging Face model name for sentiment analysis
            output_folder (str): Folder to save results (default: "result")
        """
        self.model_name = model_name
        self.output_folder = output_folder
        
        # Load tokenizer and model
        print("🔄 Loading sentiment analysis model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        print("✅ Model loaded successfully!")
        
        # Label mapping based on model documentation
        self.label_index = {
            'LABEL_0': 'positive',
            'LABEL_1': 'neutral',
            'LABEL_2': 'negative'
        }
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            str: Predicted sentiment ('positive', 'neutral', 'negative', 'unknown', or 'error')
        """
        try:
            text = str(text).strip()
            if not text:
                return "unknown"

            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probs).item()

            # Convert predicted class (int) to label like 'LABEL_0'
            label_key = f'LABEL_{predicted_class}'
            return self.label_index.get(label_key, "unknown")

        except Exception as e:
            print(f"⚠️ Error processing: {text} → {e}")
            return "error"
    
    def analyze_csv(self, input_file: str, text_column: str = "cleaned_caption") -> pd.DataFrame:
        """
        Perform sentiment analysis on an input file.
        
        Args:
            input_file (str): Path to the preprocessed input file (CSV or Excel)
            text_column (str): Column name containing text to analyze (default: "cleaned_caption")
            
        Returns:
            pd.DataFrame: DataFrame with sentiment predictions
        """
        # Load preprocessed file based on extension
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)

        # Make sure the specified column exists
        if text_column not in df.columns:
            raise Exception(f"❌ Column '{text_column}' not found in the file.")

        print(f"🔄 Analyzing {len(df)} texts for sentiment...")
        
        # Apply sentiment prediction
        df["sentiment"] = df[text_column].apply(self.predict_sentiment)
        
        print("✅ Sentiment analysis completed!")
        return df
    
    def generate_sentiment_csv(self, input_file: str, text_column: str = "cleaned_caption") -> str:
        """
        Generate an Excel file with sentiment analysis results.
        
        Args:
            input_file (str): Path to the preprocessed input file (CSV or Excel)
            text_column (str): Column name containing text to analyze (default: "cleaned_caption")
            
        Returns:
            str: Path to the generated result file
        """
        # Perform sentiment analysis
        df_with_sentiment = self.analyze_csv(input_file, text_column)
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create output filename
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_path = os.path.join(self.output_folder, f"{base_filename}_with_sentiment.xlsx")
        
        # Save result to Excel
        df_with_sentiment.to_excel(output_path, index=False)
        
        # Show sentiment distribution
        sentiment_counts = df_with_sentiment["sentiment"].value_counts()
        print("\n📊 Sentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            print(f"  {sentiment}: {count}")
        
        print(f"✅ Results saved to: {output_path}")
        return output_path

# Example usage
if __name__ == "__main__":
    analyzer = SentimentAnalyzer(output_folder="result")
    output_file = analyzer.generate_sentiment_csv("preprocessed/taman bunga celosia_preprocessed.xlsx")
