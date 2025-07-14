
# Import regular expressions and Sastrawi libraries for stopword removal and stemming
import pandas as pd
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os

class TextPreprocessor:
    """
    A class to preprocess text data for sentiment analysis.
    Purpose: Generate cleaned Excel files from filtered data.
    """
    
    def __init__(self, output_folder: str = "preprocessed"):
        """
        Initialize the TextPreprocessor.
        
        Args:
            output_folder (str): Folder to save preprocessed results (default: "preprocessed")
        """
        self.output_folder = output_folder
        
        # Initialize stopword remover
        factory = StopWordRemoverFactory()
        self.stopwords = set(factory.get_stop_words())
        
        # Initialize stemmer
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()

    def remove_symbols(self, text):
        """Remove URLs, symbols, and numbers from text."""
        text = str(text)
        text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z\s]", " ", text)    # Remove symbols and numbers
        return text

    def to_lowercase(self, text):
        """Convert text to lowercase."""
        return text.lower()

    def tokenize(self, text):
        """Split text into tokens (words)."""
        return text.split()

    def remove_stopwords(self, tokens):
        """Remove stopwords from token list."""
        return [word for word in tokens if word not in self.stopwords]

    def stem_words(self, tokens):
        """Stem each word in the token list."""
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess_text(self, text):
        """
        Full preprocessing pipeline: clean, lowercase, tokenize, remove stopwords, and stem.
        
        Args:
            text (str): Input text to preprocess
            
        Returns:
            str: Preprocessed text
        """
        text = self.remove_symbols(text)
        text = self.to_lowercase(text)
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.stem_words(tokens)
        return " ".join(tokens)
    
    def generate_preprocessed_csv(self, input_file: str, text_column: str = "caption") -> str:
        """
        Generate a preprocessed Excel file from input data.
        
        Args:
            input_file (str): Input file path (Excel or CSV)
            text_column (str): Column name containing text to preprocess (default: "caption")
            
        Returns:
            str: Path to the generated preprocessed file
        """
        # Load data based on file extension
        if input_file.endswith('.xlsx'):
            df = pd.read_excel(input_file)
        else:
            df = pd.read_csv(input_file)
        
        # Apply preprocessing to the specified text column
        df["cleaned_caption"] = df[text_column].apply(self.preprocess_text)
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Create output Excel filename
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        # Remove any existing suffix and add _preprocessed
        if base_filename.endswith('_filtered'):
            base_filename = base_filename.replace('_filtered', '')
        output_path = os.path.join(self.output_folder, f"{base_filename}_preprocessed.xlsx")
        
        # Save to Excel
        df.to_excel(output_path, index=False)
        
        # Show result sample
        print("✅ Preprocessing completed!")
        print("Sample results:")
        print(df[[text_column, "cleaned_caption"]].head())
        print(f"✅ Preprocessed data saved to: {output_path}")
        
        return output_path

# Example usage
if __name__ == "__main__":
    preprocessor = TextPreprocessor(output_folder="preprocessed")
    output_file = preprocessor.generate_preprocessed_csv("filtered/taman bunga celosia_filtered.xlsx")

