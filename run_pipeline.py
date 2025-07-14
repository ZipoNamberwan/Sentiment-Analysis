"""
Simple main execution script for Sentiment Analysis Pipeline
Executes the three classes in sequence with minimal configuration.
All outputs are saved as Excel files (.xlsx).
"""

from filtering import DataFilter
from preporcessing import TextPreprocessor
from sentiment import SentimentAnalyzer

def run_sentiment_pipeline():
    """
    Run the complete sentiment analysis pipeline with default settings.
    """
    print("🚀 Starting Sentiment Analysis Pipeline...")
    
    # Step 1: Filter data
    print("\n📋 Step 1: Filtering data...")
    filter_processor = DataFilter(max_years=2, output_folder="filtered")
    filtered_file = filter_processor.generate_filtered_csv("source/taman bunga celosia.csv")
    
    # Step 2: Preprocess text
    print("\n🧹 Step 2: Preprocessing text...")
    preprocessor = TextPreprocessor(output_folder="preprocessed")
    preprocessed_file = preprocessor.generate_preprocessed_csv(filtered_file, "caption")
    
    # Step 3: Analyze sentiment
    print("\n🎭 Step 3: Analyzing sentiment...")
    analyzer = SentimentAnalyzer(output_folder="result")
    final_result = analyzer.generate_sentiment_csv(preprocessed_file, "cleaned_caption")
    
    print(f"\n🎉 Pipeline completed! Final result: {final_result}")
    return final_result

if __name__ == "__main__":
    run_sentiment_pipeline()
