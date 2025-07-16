"""
Main script to run the complete sentiment analysis pipeline.
This script demonstrates how to use the four classes together:
1. DataFilter - Filter source data
2. TextPreprocessor - Clean and preprocess text
3. SentimentAnalyzer - Perform sentiment analysis
4. SummaryGenerator - Generate comprehensive summary reports
"""

from filtering import DataFilter
from preporcessing import TextPreprocessor  # Note: keeping original filename
from sentiment import SentimentAnalyzer
from summary_generator import SummaryGenerator
import os
import glob
import pandas as pd

class SentimentAnalysisPipeline:
    """
    A complete pipeline for sentiment analysis from raw data to final results.
    """
    
    def __init__(self, 
                 filter_max_years: int = 2,
                 filter_output_folder: str = "filtered",
                 preprocess_output_folder: str = "preprocessed", 
                 sentiment_output_folder: str = "result/all",
                 sentiment_model: str = "mdhugol/indonesia-bert-sentiment-classification"):
        """
        Initialize the complete pipeline.
        
        Args:
            filter_max_years (int): Maximum age of reviews to keep (default: 2 years)
            filter_output_folder (str): Folder for filtered data
            preprocess_output_folder (str): Folder for preprocessed data
            sentiment_output_folder (str): Folder for final results
            sentiment_model (str): Hugging Face model for sentiment analysis
        """
        self.data_filter = DataFilter(max_years=filter_max_years, output_folder=filter_output_folder)
        self.text_preprocessor = TextPreprocessor(output_folder=preprocess_output_folder)
        self.sentiment_analyzer = SentimentAnalyzer(model_name=sentiment_model, output_folder=sentiment_output_folder)
    
    def run_complete_pipeline(self, source_csv_path: str, text_column: str = "caption") -> str:
        """
        Run the complete sentiment analysis pipeline.
        
        Args:
            source_csv_path (str): Path to the source CSV file
            text_column (str): Column name containing text data (default: "caption")
            
        Returns:
            str: Path to the final result file
        """
        print("🚀 Starting Sentiment Analysis Pipeline...")
        print("=" * 50)
        
        # Step 1: Filter data
        print("📋 Step 1: Filtering data...")
        filtered_file = self.data_filter.generate_filtered_csv(source_csv_path)
        print(f"✅ Filtering completed: {filtered_file}")
        print()
        
        # Step 2: Preprocess text
        print("🧹 Step 2: Preprocessing text...")
        preprocessed_file = self.text_preprocessor.generate_preprocessed_csv(filtered_file, text_column)
        print(f"✅ Preprocessing completed: {preprocessed_file}")
        print()
        
        # Step 3: Analyze sentiment
        print("🎭 Step 3: Analyzing sentiment...")
        final_result = self.sentiment_analyzer.generate_sentiment_csv(preprocessed_file)
        print(f"✅ Sentiment analysis completed: {final_result}")
        print()
        
        print("🎉 Pipeline completed successfully!")
        print("=" * 50)
        return final_result

def generate_summary_report(results: list) -> str:
    """
    Generate a summary Excel file with sentiment analysis results for all processed files.
    (Legacy function - now uses SummaryGenerator class)
    
    Args:
        results (list): List of processing results with file paths and status
        
    Returns:
        str: Path to the generated summary file
    """
    summary_generator = SummaryGenerator(output_folder="result/all")
    return summary_generator.generate_summary_report(results)

def main():
    """
    Main function to run the pipeline on all CSV files in the source folder.
    """
    # Initialize pipeline
    pipeline = SentimentAnalysisPipeline(
        filter_max_years=2,
        filter_output_folder="filtered",
        preprocess_output_folder="preprocessed",
        sentiment_output_folder="result/all"
    )
    
    # Find all CSV files in the source folder
    source_folder = "source"
    csv_files = glob.glob(os.path.join(source_folder, "*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in the '{source_folder}' folder.")
        print("Please make sure there are CSV files in the source folder.")
        return
    
    print(f"📁 Found {len(csv_files)} CSV file(s) in the source folder:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    print()
    
    # Process each CSV file
    results = []
    for i, source_file in enumerate(csv_files, 1):
        print(f"🔄 Processing file {i}/{len(csv_files)}: {os.path.basename(source_file)}")
        print("=" * 60)
        
        try:
            final_result = pipeline.run_complete_pipeline(source_file)
            results.append({
                'source_file': source_file,
                'result_file': final_result,
                'status': 'Success'
            })
            print(f"✅ Successfully processed: {os.path.basename(source_file)}")
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(source_file)}: {str(e)}")
            results.append({
                'source_file': source_file,
                'result_file': None,
                'status': f'Error: {str(e)}'
            })
        
        print()
    
    # Summary report
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    successful = len([r for r in results if r['status'] == 'Success'])
    failed = len(results) - successful
    
    print(f"✅ Successfully processed: {successful} file(s)")
    print(f"❌ Failed to process: {failed} file(s)")
    print()
    
    if successful > 0:
        print("🎯 Successfully processed files:")
        for result in results:
            if result['status'] == 'Success':
                print(f"  📄 {os.path.basename(result['source_file'])} → {result['result_file']}")
    
    if failed > 0:
        print("\n⚠️ Failed files:")
        for result in results:
            if result['status'] != 'Success':
                print(f"  📄 {os.path.basename(result['source_file'])}: {result['status']}")
    
    # Generate summary report
    print("\n" + "=" * 60)
    summary_file = generate_summary_report(results)
    
    print("\n🎊 BATCH PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"📊 Summary report available at: {summary_file}")
    print("🎯 Check the summary file for detailed sentiment analysis statistics!")
    
    return summary_file

def process_single_file(filename: str):
    """
    Process a single specific file.
    
    Args:
        filename (str): Name of the CSV file in the source folder (e.g., "taman bunga celosia.csv")
    """
    pipeline = SentimentAnalysisPipeline()
    source_file = os.path.join("source", filename)
    
    if os.path.exists(source_file):
        print(f"🎯 Processing single file: {filename}")
        final_result = pipeline.run_complete_pipeline(source_file)
        print(f"🎯 Final result saved to: {final_result}")
        return final_result
    else:
        print(f"❌ Source file not found: {source_file}")
        print("Please make sure the CSV file exists in the source folder.")
        return None

if __name__ == "__main__":
    import sys
    
    # Check if specific filename is provided as command line argument
    if len(sys.argv) > 1:
        filename = sys.argv[1]
        print(f"🎯 Processing specific file: {filename}")
        process_single_file(filename)
    else:
        # Process all CSV files in the source folder
        print("🚀 Processing all CSV files in the source folder...")
        main()
