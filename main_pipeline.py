"""
Main execution script for Sentiment Analysis Pipeline
This script runs the three classes in sequence:
1. DataFilter - Filter source data by timestamp and caption criteria
2. TextPreprocessor - Clean and preprocess text data  
3. SentimentAnalyzer - Perform sentiment analysis on cleaned text

Usage: python main_pipeline.py
"""

from filtering import DataFilter
from preporcessing import TextPreprocessor
from sentiment import SentimentAnalyzer
import os
import sys

def main():
    """
    Execute the complete sentiment analysis pipeline in sequence.
    """
    print("🚀 Starting Sentiment Analysis Pipeline")
    print("=" * 60)
    
    # Configuration
    source_file = "source/taman bunga celosia.csv"
    max_years = 2
    text_column = "caption"
    
    # Check if source file exists
    if not os.path.exists(source_file):
        print(f"❌ Error: Source file not found: {source_file}")
        print("Please make sure the source CSV file exists in the 'source' folder.")
        sys.exit(1)
    
    try:
        # Step 1: Initialize DataFilter and filter data
        print("📋 STEP 1: FILTERING DATA")
        print("-" * 30)
        data_filter = DataFilter(max_years=max_years, output_folder="filtered")
        filtered_file = data_filter.generate_filtered_csv(source_file)
        print(f"✅ Step 1 completed: {filtered_file}")
        print()
        
        # Step 2: Initialize TextPreprocessor and preprocess text
        print("🧹 STEP 2: PREPROCESSING TEXT")
        print("-" * 30)
        text_preprocessor = TextPreprocessor(output_folder="preprocessed")
        preprocessed_file = text_preprocessor.generate_preprocessed_csv(filtered_file, text_column)
        print(f"✅ Step 2 completed: {preprocessed_file}")
        print()
        
        # Step 3: Initialize SentimentAnalyzer and analyze sentiment
        print("🎭 STEP 3: SENTIMENT ANALYSIS")
        print("-" * 30)
        sentiment_analyzer = SentimentAnalyzer(output_folder="result")
        final_result = sentiment_analyzer.generate_sentiment_csv(preprocessed_file)
        print(f"✅ Step 3 completed: {final_result}")
        print()
        
        # Summary
        print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Generated Files:")
        print(f"  1. Filtered data: {filtered_file}")
        print(f"  2. Preprocessed data: {preprocessed_file}")
        print(f"  3. Final results: {final_result}")
        print()
        print("🎯 You can now find the sentiment analysis results in Excel format in the 'result' folder.")
        
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please check that all required files exist.")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        print("Pipeline execution failed. Please check the error message above.")
        sys.exit(1)

def run_with_custom_file(source_csv_path: str, text_col: str = "caption", years: int = 2):
    """
    Run the pipeline with custom parameters.
    
    Args:
        source_csv_path (str): Path to the source CSV file
        text_col (str): Column name containing text data (default: "caption")
        years (int): Maximum age of reviews to keep (default: 2)
    """
    print(f"🚀 Running pipeline with custom file: {source_csv_path}")
    print("=" * 60)
    
    if not os.path.exists(source_csv_path):
        print(f"❌ Error: Source file not found: {source_csv_path}")
        return None
    
    try:
        # Step 1: Filter
        print("📋 Filtering data...")
        data_filter = DataFilter(max_years=years, output_folder="filtered")
        filtered_file = data_filter.generate_filtered_csv(source_csv_path)
        
        # Step 2: Preprocess
        print("🧹 Preprocessing text...")
        text_preprocessor = TextPreprocessor(output_folder="preprocessed")
        preprocessed_file = text_preprocessor.generate_preprocessed_csv(filtered_file, text_col)
        
        # Step 3: Sentiment Analysis
        print("🎭 Analyzing sentiment...")
        sentiment_analyzer = SentimentAnalyzer(output_folder="result")
        final_result = sentiment_analyzer.generate_sentiment_csv(preprocessed_file)
        
        print("🎉 Custom pipeline completed!")
        return final_result
        
    except Exception as e:
        print(f"❌ Error in custom pipeline: {e}")
        return None

if __name__ == "__main__":
    # Check if custom file is provided as command line argument
    if len(sys.argv) > 1:
        custom_file = sys.argv[1]
        text_column = sys.argv[2] if len(sys.argv) > 2 else "caption"
        max_years = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        
        print(f"Running with custom parameters:")
        print(f"  File: {custom_file}")
        print(f"  Text column: {text_column}")
        print(f"  Max years: {max_years}")
        print()
        
        run_with_custom_file(custom_file, text_column, max_years)
    else:
        # Run with default settings
        main()
