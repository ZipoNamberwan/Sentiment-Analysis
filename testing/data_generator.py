"""
TestingDataGenerator class for generating balanced sentiment testing data.
"""

import pandas as pd
import os
import glob
import numpy as np
from collections import Counter

class TestingDataGenerator:
    """
    Class for generating testing data from sentiment analysis results.
    Takes 20% of rows from each file with balanced sentiment distribution.
    """
    
    def __init__(self, source_folder="result/all", output_folder="result/testing", sample_percentage=0.20):
        """
        Initialize the TestingDataGenerator.
        
        Args:
            source_folder (str): Folder containing sentiment analysis results
            output_folder (str): Folder to save testing data
            sample_percentage (float): Percentage of data to sample (default: 0.20 for 20%)
        """
        self.source_folder = source_folder
        self.output_folder = output_folder
        self.sample_percentage = sample_percentage
        self.results = []
        self.total_original_rows = 0
        self.total_testing_rows = 0
    
    def sample_balanced_sentiments(self, df, sample_size, sentiment_column='sentiment'):
        """
        Sample data with balanced sentiment distribution (1/3 each for positive, negative, neutral).
        
        Args:
            df (pd.DataFrame): Input dataframe
            sample_size (int): Total number of samples to take
            sentiment_column (str): Column name containing sentiment labels
            
        Returns:
            pd.DataFrame: Sampled dataframe with balanced sentiments
        """
        # Calculate target counts for each sentiment (1/3 each)
        target_per_sentiment = sample_size // 3
        remainder = sample_size % 3
        
        # Group by sentiment
        sentiment_groups = df.groupby(sentiment_column)
        sampled_dfs = []
        
        sentiments = ['positive', 'negative', 'neutral']
        
        for i, sentiment in enumerate(sentiments):
            if sentiment in sentiment_groups.groups:
                group_df = sentiment_groups.get_group(sentiment)
                
                # Add remainder to first sentiment groups if needed
                current_target = target_per_sentiment + (1 if i < remainder else 0)
                
                # Sample from this sentiment group
                if len(group_df) >= current_target:
                    sampled = group_df.sample(n=current_target, random_state=42)
                else:
                    # If not enough samples, take all available
                    sampled = group_df.copy()
                    print(f"  ⚠️ Only {len(group_df)} {sentiment} samples available (needed {current_target})")
                
                sampled_dfs.append(sampled)
            else:
                print(f"  ⚠️ No {sentiment} samples found in the data")
        
        # Combine all sampled data and shuffle
        if sampled_dfs:
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
            return result_df
        else:
            return pd.DataFrame()
    
    def process_single_file(self, excel_file):
        """
        Process a single Excel file to generate testing data.
        
        Args:
            excel_file (str): Path to the Excel file
            
        Returns:
            dict: Result of processing
        """
        filename = os.path.basename(excel_file)
        print(f"📖 Processing file: {filename}")
        
        try:
            # Read the Excel file
            df = pd.read_excel(excel_file)
            original_count = len(df)
            self.total_original_rows += original_count
            
            # Check if sentiment column exists
            if 'sentiment' not in df.columns:
                print(f"  ❌ No 'sentiment' column found in {filename}")
                return {
                    'file': filename,
                    'status': 'Error: No sentiment column',
                    'original_rows': original_count,
                    'testing_rows': 0
                }
            
            # Calculate sample size (20% of total rows)
            sample_size = max(3, int(original_count * self.sample_percentage))  # Minimum 3 for balanced sampling
            
            # Show original sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            print(f"  📊 Original data: {original_count} rows")
            print(f"     Sentiment distribution: {dict(sentiment_counts)}")
            
            # Sample balanced data
            testing_df = self.sample_balanced_sentiments(df, sample_size)
            testing_count = len(testing_df)
            self.total_testing_rows += testing_count
            
            if testing_count == 0:
                print(f"  ❌ No testing data generated for {filename}")
                return {
                    'file': filename,
                    'status': 'Error: No data sampled',
                    'original_rows': original_count,
                    'testing_rows': 0
                }
            
            # Show testing sentiment distribution
            testing_sentiment_counts = testing_df['sentiment'].value_counts()
            print(f"  🧪 Testing data: {testing_count} rows ({testing_count/original_count*100:.1f}%)")
            print(f"     Sentiment distribution: {dict(testing_sentiment_counts)}")
            
            # Create output filename
            base_name = filename.replace('_with_sentiment.xlsx', '')
            output_filename = f"{base_name}_testing.xlsx"
            output_path = os.path.join(self.output_folder, output_filename)
            
            # Save testing data
            testing_df.to_excel(output_path, index=False)
            print(f"  ✅ Testing data saved: {output_filename}")
            
            return {
                'file': filename,
                'status': 'Success',
                'original_rows': original_count,
                'testing_rows': testing_count,
                'output_file': output_filename
            }
            
        except Exception as e:
            print(f"  ❌ Error processing {filename}: {str(e)}")
            return {
                'file': filename,
                'status': f'Error: {str(e)}',
                'original_rows': 0,
                'testing_rows': 0
            }
    
    def generate_testing_data(self):
        """
        Generate testing data from all sentiment analysis files.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("🧪 TESTING DATA GENERATOR")
        print("=" * 50)
        print(f"📂 Source folder: {self.source_folder}")
        print(f"📂 Output folder: {self.output_folder}")
        print(f"📊 Sample percentage: {self.sample_percentage*100}%")
        print(f"⚖️ Sentiment distribution: 1/3 each (positive, negative, neutral)")
        print()
        
        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Find all Excel files in the source folder
        excel_pattern = os.path.join(self.source_folder, "*_with_sentiment.xlsx")
        excel_files = glob.glob(excel_pattern)
        
        if not excel_files:
            print(f"❌ No sentiment analysis files found in '{self.source_folder}'")
            print("   Looking for files ending with '_with_sentiment.xlsx'")
            return False
        
        print(f"📁 Found {len(excel_files)} sentiment analysis file(s):")
        for file in excel_files:
            print(f"  - {os.path.basename(file)}")
        print()
        
        # Process each file
        self.results = []
        self.total_original_rows = 0
        self.total_testing_rows = 0
        
        for i, excel_file in enumerate(excel_files, 1):
            print(f"📖 Processing file {i}/{len(excel_files)}")
            result = self.process_single_file(excel_file)
            self.results.append(result)
            print()
        
        # Generate summary
        self._generate_summary()
        
        return len([r for r in self.results if r['status'] == 'Success']) > 0
    
    def _generate_summary(self):
        """
        Generate and display summary of testing data generation.
        """
        print("📊 TESTING DATA GENERATION SUMMARY")
        print("=" * 50)
        
        successful = [r for r in self.results if r['status'] == 'Success']
        failed = [r for r in self.results if r['status'] != 'Success']
        
        print(f"✅ Successfully processed: {len(successful)} file(s)")
        print(f"❌ Failed to process: {len(failed)} file(s)")
        print(f"📄 Total original rows: {self.total_original_rows:,}")
        print(f"🧪 Total testing rows: {self.total_testing_rows:,}")
        
        if self.total_original_rows > 0:
            print(f"📊 Overall sampling rate: {self.total_testing_rows/self.total_original_rows*100:.1f}%")
        
        print()
        
        if successful:
            print("🎯 Successfully generated testing data:")
            for result in successful:
                print(f"  📄 {result['file']}")
                print(f"     {result['original_rows']} → {result['testing_rows']} rows ({result['testing_rows']/result['original_rows']*100:.1f}%)")
        
        if failed:
            print("\n⚠️ Failed files:")
            for result in failed:
                print(f"  📄 {result['file']}: {result['status']}")
        
        # Create summary Excel file
        summary_df = pd.DataFrame(self.results)
        summary_path = os.path.join(self.output_folder, "testing_data_summary.xlsx")
        summary_df.to_excel(summary_path, index=False)
        
        print(f"\n📋 Summary report saved: {summary_path}")
        print("\n🎊 TESTING DATA GENERATION COMPLETED!")
        print("=" * 50)
    
    def get_results(self):
        """
        Get the results of testing data generation.
        
        Returns:
            dict: Summary of results
        """
        return {
            'results': self.results,
            'total_original_rows': self.total_original_rows,
            'total_testing_rows': self.total_testing_rows,
            'successful_files': len([r for r in self.results if r['status'] == 'Success']),
            'failed_files': len([r for r in self.results if r['status'] != 'Success'])
        }

def main():
    """
    Main function for standalone execution of data generation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate testing data from sentiment analysis results')
    parser.add_argument('--source', default='result/all', 
                      help='Source folder containing sentiment analysis results (default: result/all)')
    parser.add_argument('--output', default='result/testing',
                      help='Output folder for testing data (default: result/testing)')
    parser.add_argument('--percentage', type=float, default=0.20,
                      help='Percentage of data to sample (default: 0.20)')
    
    args = parser.parse_args()
    
    print("🧪 STANDALONE TESTING DATA GENERATOR")
    print("=" * 50)
    print(f"📂 Source: {args.source}")
    print(f"📂 Output: {args.output}")
    print(f"📊 Sample percentage: {args.percentage*100}%")
    print("=" * 50)
    print()
    
    # Create generator instance
    generator = TestingDataGenerator(
        source_folder=args.source,
        output_folder=args.output,
        sample_percentage=args.percentage
    )
    
    # Generate testing data
    success = generator.generate_testing_data()
    
    if success:
        results = generator.get_results()
        print("\n🎉 DATA GENERATION COMPLETED!")
        print("=" * 50)
        print(f"✅ Successful files: {results['successful_files']}")
        print(f"❌ Failed files: {results['failed_files']}")
        print(f"📄 Total testing rows: {results['total_testing_rows']:,}")
        print(f"📄 Total original rows: {results['total_original_rows']:,}")
        if results['total_original_rows'] > 0:
            print(f"📊 Sampling rate: {results['total_testing_rows']/results['total_original_rows']*100:.1f}%")
    else:
        print("\n❌ DATA GENERATION FAILED!")
        print("Check the output above for details.")

if __name__ == "__main__":
    main()
