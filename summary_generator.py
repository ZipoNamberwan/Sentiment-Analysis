"""
Summary Generator for Sentiment Analysis Pipeline
This module contains the SummaryGenerator class that creates comprehensive 
Excel reports with sentiment statistics from processing results.
"""

import os
import pandas as pd

class SummaryGenerator:
    """
    A class to generate summary reports for sentiment analysis results.
    Purpose: Create comprehensive Excel reports with sentiment statistics.
    """
    
    def __init__(self, output_folder: str = "result"):
        """
        Initialize the SummaryGenerator.
        
        Args:
            output_folder (str): Folder to save summary reports (default: "result")
        """
        self.output_folder = output_folder
    
    def calculate_sentiment_statistics(self, result_file_path: str) -> dict:
        """
        Calculate sentiment statistics from a result file.
        
        Args:
            result_file_path (str): Path to the sentiment analysis result file
            
        Returns:
            dict: Dictionary containing sentiment statistics
        """
        try:
            df = pd.read_excel(result_file_path)
            
            # Calculate sentiment statistics
            total_reviews = len(df)
            sentiment_counts = df['sentiment'].value_counts()
            
            # Calculate counts and percentages
            positive_count = sentiment_counts.get('positive', 0)
            neutral_count = sentiment_counts.get('neutral', 0)
            negative_count = sentiment_counts.get('negative', 0)
            error_count = sentiment_counts.get('error', 0)
            unknown_count = sentiment_counts.get('unknown', 0)
            
            # Calculate percentages (avoid division by zero)
            if total_reviews > 0:
                positive_percent = round((positive_count / total_reviews) * 100, 2)
                neutral_percent = round((neutral_count / total_reviews) * 100, 2)
                negative_percent = round((negative_count / total_reviews) * 100, 2)
                error_percent = round((error_count / total_reviews) * 100, 2)
                unknown_percent = round((unknown_count / total_reviews) * 100, 2)
            else:
                positive_percent = neutral_percent = negative_percent = error_percent = unknown_percent = 0
            
            return {
                'total_reviews': total_reviews,
                'positive_count': positive_count,
                'positive_percent': positive_percent,
                'neutral_count': neutral_count,
                'neutral_percent': neutral_percent,
                'negative_count': negative_count,
                'negative_percent': negative_percent,
                'error_count': error_count,
                'error_percent': error_percent,
                'unknown_count': unknown_count,
                'unknown_percent': unknown_percent
            }
            
        except Exception as e:
            print(f"⚠️ Error calculating statistics for {result_file_path}: {e}")
            return {
                'total_reviews': 0,
                'positive_count': 0, 'positive_percent': 0,
                'neutral_count': 0, 'neutral_percent': 0,
                'negative_count': 0, 'negative_percent': 0,
                'error_count': 0, 'error_percent': 0,
                'unknown_count': 0, 'unknown_percent': 0
            }
    
    def create_summary_data(self, results: list) -> list:
        """
        Create summary data from processing results.
        
        Args:
            results (list): List of processing results with file paths and status
            
        Returns:
            list: List of dictionaries containing summary data
        """
        summary_data = []
        
        for result in results:
            base_data = {
                'Filename': os.path.basename(result['source_file']),
                'Status': result['status'],
                'Result_File': result.get('result_file', 'N/A')
            }
            
            if result['status'] == 'Success' and result.get('result_file'):
                # Get sentiment statistics
                stats = self.calculate_sentiment_statistics(result['result_file'])
                
                # Combine base data with statistics
                summary_data.append({
                    **base_data,
                    'Total_Reviews': stats['total_reviews'],
                    'Positive_Count': stats['positive_count'],
                    'Positive_Percent': stats['positive_percent'],
                    'Neutral_Count': stats['neutral_count'],
                    'Neutral_Percent': stats['neutral_percent'],
                    'Negative_Count': stats['negative_count'],
                    'Negative_Percent': stats['negative_percent'],
                    'Error_Count': stats['error_count'],
                    'Error_Percent': stats['error_percent'],
                    'Unknown_Count': stats['unknown_count'],
                    'Unknown_Percent': stats['unknown_percent']
                })
            else:
                # File processing failed - add zero statistics
                summary_data.append({
                    **base_data,
                    'Total_Reviews': 0,
                    'Positive_Count': 0, 'Positive_Percent': 0,
                    'Neutral_Count': 0, 'Neutral_Percent': 0,
                    'Negative_Count': 0, 'Negative_Percent': 0,
                    'Error_Count': 0, 'Error_Percent': 0,
                    'Unknown_Count': 0, 'Unknown_Percent': 0
                })
        
        return summary_data
    
    def create_statistics_data(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create overall statistics from summary data.
        
        Args:
            summary_df (pd.DataFrame): Summary DataFrame
            
        Returns:
            pd.DataFrame: Statistics DataFrame
        """
        successful_files = summary_df[summary_df['Status'] == 'Success']
        
        if len(successful_files) > 0:
            stats_data = {
                'Metric': [
                    'Total Files Processed',
                    'Successfully Processed',
                    'Failed Processing',
                    'Total Reviews Analyzed',
                    'Average Positive %',
                    'Average Neutral %',
                    'Average Negative %',
                    'Total Positive Reviews',
                    'Total Neutral Reviews',
                    'Total Negative Reviews'
                ],
                'Value': [
                    len(summary_df),
                    len(successful_files),
                    len(summary_df) - len(successful_files),
                    successful_files['Total_Reviews'].sum(),
                    round(successful_files['Positive_Percent'].mean(), 2),
                    round(successful_files['Neutral_Percent'].mean(), 2),
                    round(successful_files['Negative_Percent'].mean(), 2),
                    successful_files['Positive_Count'].sum(),
                    successful_files['Neutral_Count'].sum(),
                    successful_files['Negative_Count'].sum()
                ]
            }
        else:
            stats_data = {
                'Metric': ['No successful files to analyze'],
                'Value': ['N/A']
            }
        
        return pd.DataFrame(stats_data)
    
    def generate_summary_report(self, results: list, summary_filename: str = "sentiment_analysis_summary.xlsx") -> str:
        """
        Generate a comprehensive summary Excel file with sentiment analysis results.
        
        Args:
            results (list): List of processing results with file paths and status
            summary_filename (str): Name of the summary file (default: "sentiment_analysis_summary.xlsx")
            
        Returns:
            str: Path to the generated summary file
        """
        print("📊 Generating comprehensive summary report...")
        
        # Create summary data
        summary_data = self.create_summary_data(results)
        summary_df = pd.DataFrame(summary_data)
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)
        
        # Generate summary file path
        summary_file = os.path.join(self.output_folder, summary_filename)
        
        # Save summary to Excel with multiple sheets
        with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
            # Main summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Statistics sheet
            stats_df = self.create_statistics_data(summary_df)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Individual file details (if any successful files)
            successful_files = summary_df[summary_df['Status'] == 'Success']
            if len(successful_files) > 0:
                details_data = []
                for _, row in successful_files.iterrows():
                    details_data.append({
                        'Filename': row['Filename'],
                        'Sentiment': 'Positive',
                        'Count': row['Positive_Count'],
                        'Percentage': row['Positive_Percent']
                    })
                    details_data.append({
                        'Filename': row['Filename'],
                        'Sentiment': 'Neutral', 
                        'Count': row['Neutral_Count'],
                        'Percentage': row['Neutral_Percent']
                    })
                    details_data.append({
                        'Filename': row['Filename'],
                        'Sentiment': 'Negative',
                        'Count': row['Negative_Count'],
                        'Percentage': row['Negative_Percent']
                    })
                
                details_df = pd.DataFrame(details_data)
                details_df.to_excel(writer, sheet_name='Detailed_Breakdown', index=False)
        
        print(f"✅ Summary report generated: {summary_file}")
        print(f"📋 Report contains {len(summary_df)} file(s) with detailed sentiment analysis")
        
        return summary_file

# Example usage
if __name__ == "__main__":
    # Example of how to use SummaryGenerator independently
    summary_gen = SummaryGenerator(output_folder="result")
    
    # Example results data structure
    example_results = [
        {
            'source_file': 'source/example.csv',
            'result_file': 'result/example_with_sentiment.xlsx',
            'status': 'Success'
        }
    ]
    
    # Generate summary report
    # summary_file = summary_gen.generate_summary_report(example_results)
    print("SummaryGenerator class is ready to use!")
