"""
SentimentEvaluator class for evaluating sentiment analysis accuracy.
"""

import pandas as pd
import os
import glob
import numpy as np
from collections import Counter

# Try to import sklearn, if not available, use basic metrics
try:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

class SentimentEvaluator:
    """
    Class for evaluating sentiment analysis accuracy by comparing predicted sentiment 
    with Google review ratings.
    
    Rating mapping:
    - Positive: 4-5 star ratings
    - Neutral: 3 star ratings  
    - Negative: 1-2 star ratings
    """
    
    def __init__(self, testing_folder="result/testing", output_folder="result/testing"):
        """
        Initialize the SentimentEvaluator.
        
        Args:
            testing_folder (str): Folder containing testing data
            output_folder (str): Folder to save evaluation results
        """
        self.testing_folder = testing_folder
        self.output_folder = output_folder
        self.all_results = []
        self.all_predictions = []
        self.all_ground_truth = []
        self.overall_cm = np.zeros((3, 3), dtype=int)
        self.sklearn_available = SKLEARN_AVAILABLE
        self.plotting_available = PLOTTING_AVAILABLE
        
        if not self.sklearn_available:
            print("⚠️ scikit-learn not available, using basic accuracy calculation")
        if not self.plotting_available:
            print("⚠️ Plotting libraries not available, skipping visualization")
    
    def map_rating_to_sentiment(self, rating):
        """
        Map Google review rating to sentiment category.
        
        Args:
            rating (int/float): Rating value (1-5)
            
        Returns:
            str: Sentiment category ('positive', 'neutral', 'negative')
        """
        try:
            rating = float(rating)
            if rating >= 4:
                return 'positive'
            elif rating == 3:
                return 'neutral'
            elif rating <= 2:
                return 'negative'
            else:
                return 'unknown'
        except (ValueError, TypeError):
            return 'unknown'
    
    def evaluate_single_file(self, file_path):
        """
        Evaluate sentiment predictions for a single file.
        
        Args:
            file_path (str): Path to the testing data file
            
        Returns:
            dict: Evaluation metrics and data
        """
        filename = os.path.basename(file_path)
        print(f"📊 Evaluating: {filename}")
        
        try:
            # Read the testing data
            df = pd.read_excel(file_path)
            
            # Check required columns
            required_columns = ['sentiment', 'rating']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ❌ Missing columns: {missing_columns}")
                return {
                    'filename': filename,
                    'status': f'Missing columns: {missing_columns}',
                    'total_samples': len(df),
                    'accuracy': 0,
                    'precision': {},
                    'recall': {},
                    'f1_score': {}
                }
            
            # Map ratings to ground truth sentiments
            df['ground_truth'] = df['rating'].apply(self.map_rating_to_sentiment)
            
            # Filter out unknown ratings
            valid_df = df[df['ground_truth'] != 'unknown'].copy()
            
            if len(valid_df) == 0:
                print(f"  ❌ No valid ratings found")
                return {
                    'filename': filename,
                    'status': 'No valid ratings',
                    'total_samples': len(df),
                    'accuracy': 0,
                    'precision': {},
                    'recall': {},
                    'f1_score': {}
                }
            
            # Get predictions and ground truth
            y_pred = valid_df['sentiment'].tolist()
            y_true = valid_df['ground_truth'].tolist()
            
            # Calculate metrics
            if self.sklearn_available:
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average=None, labels=['positive', 'neutral', 'negative'], zero_division=0
                )
                
                # Create metrics dictionary
                labels = ['positive', 'neutral', 'negative']
                precision_dict = {labels[i]: precision[i] if i < len(precision) else 0 for i in range(3)}
                recall_dict = {labels[i]: recall[i] if i < len(recall) else 0 for i in range(3)}
                f1_dict = {labels[i]: f1[i] if i < len(f1) else 0 for i in range(3)}
                support_dict = {labels[i]: support[i] if i < len(support) else 0 for i in range(3)}
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])
            else:
                # Basic accuracy calculation
                correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
                accuracy = correct / len(y_true) if len(y_true) > 0 else 0
                precision_dict = {}
                recall_dict = {}
                f1_dict = {}
                support_dict = {}
                cm = np.zeros((3, 3))
            
            # Show distribution
            pred_dist = Counter(y_pred)
            true_dist = Counter(y_true)
            
            print(f"  📈 Valid samples: {len(valid_df)}/{len(df)}")
            print(f"  🎯 Accuracy: {accuracy:.3f}")
            print(f"  📊 Predicted: {dict(pred_dist)}")
            print(f"  📊 Ground truth: {dict(true_dist)}")
            
            return {
                'filename': filename,
                'status': 'Success',
                'total_samples': len(df),
                'valid_samples': len(valid_df),
                'accuracy': accuracy,
                'precision': precision_dict,
                'recall': recall_dict,
                'f1_score': f1_dict,
                'support': support_dict,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'ground_truth': y_true,
                'predicted_distribution': dict(pred_dist),
                'true_distribution': dict(true_dist)
            }
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return {
                'filename': filename,
                'status': f'Error: {str(e)}',
                'total_samples': 0,
                'accuracy': 0,
                'precision': {},
                'recall': {},
                'f1_score': {}
            }
    
    def create_confusion_matrix_plot(self, cm, labels, title, save_path=None):
        """
        Create and save confusion matrix plot.
        
        Args:
            cm (np.array): Confusion matrix
            labels (list): Class labels
            title (str): Plot title
            save_path (str): Path to save the plot
        """
        if not self.plotting_available:
            print(f"  ⚠️ Plotting not available, skipping confusion matrix visualization")
            return
            
        try:
            plt.figure(figsize=(10, 8))
            
            # Calculate percentages for better interpretation
            cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Create heatmap with both counts and percentages
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels,
                        cbar_kws={'label': 'Count'})
            
            # Add percentage annotations
            for i in range(len(labels)):
                for j in range(len(labels)):
                    if cm.sum(axis=1)[i] > 0:  # Avoid division by zero
                        plt.text(j + 0.5, i + 0.7, f'({cm_percentage[i, j]:.1f}%)', 
                                ha='center', va='center', fontsize=10, color='red')
            
            plt.title(f'{title}\n(Numbers = Count, Red = Percentage)', fontsize=14)
            plt.ylabel('Actual Sentiment (from Rating)', fontsize=12)
            plt.xlabel('Predicted Sentiment (from Model)', fontsize=12)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  📊 Confusion matrix saved: {os.path.basename(save_path)}")
            
            plt.close()
        except Exception as e:
            print(f"  ⚠️ Error creating plot: {str(e)}")
    
    def evaluate_testing_data(self):
        """
        Evaluate all testing data files and generate comprehensive report.
        
        Returns:
            bool: True if successful, False otherwise
        """
        print("🧪 SENTIMENT ANALYSIS EVALUATION")
        print("=" * 50)
        print(f"📂 Testing folder: {self.testing_folder}")
        print(f"📂 Output folder: {self.output_folder}")
        print()
        
        # Find all testing Excel files
        excel_pattern = os.path.join(self.testing_folder, "*_testing.xlsx")
        excel_files = glob.glob(excel_pattern)
        
        if not excel_files:
            print(f"❌ No testing files found in '{self.testing_folder}'")
            print("   Looking for files ending with '_testing.xlsx'")
            return False
        
        print(f"📁 Found {len(excel_files)} testing file(s):")
        for file in excel_files:
            print(f"  - {os.path.basename(file)}")
        print()
        
        # Reset evaluation data
        self.all_results = []
        self.all_predictions = []
        self.all_ground_truth = []
        self.overall_cm = np.zeros((3, 3), dtype=int)
        
        # Evaluate each file
        for i, excel_file in enumerate(excel_files, 1):
            print(f"📖 Processing file {i}/{len(excel_files)}")
            result = self.evaluate_single_file(excel_file)
            self.all_results.append(result)
            
            if result['status'] == 'Success':
                # Aggregate data for overall metrics
                self.all_predictions.extend(result['predictions'])
                self.all_ground_truth.extend(result['ground_truth'])
                if self.sklearn_available:
                    self.overall_cm += result['confusion_matrix']
                
                # Create individual confusion matrix plot
                place_name = result['filename'].replace('_preprocessed_testing.xlsx', '')
                cm_path = os.path.join(self.output_folder, f"{place_name}_confusion_matrix.png")
                self.create_confusion_matrix_plot(
                    result['confusion_matrix'], 
                    ['Positive', 'Neutral', 'Negative'],
                    f'Confusion Matrix - {place_name.title()}',
                    cm_path
                )
            
            print()
        
        # Calculate and display overall metrics
        self._calculate_overall_metrics()
        
        # Save detailed results
        self._save_detailed_results()
        
        # Create summary report
        self._create_summary_report()
        
        successful_results = [r for r in self.all_results if r['status'] == 'Success']
        failed_results = [r for r in self.all_results if r['status'] != 'Success']
        
        print(f"\n🎊 EVALUATION COMPLETED!")
        print("=" * 50)
        
        if failed_results:
            print(f"\n⚠️ Failed evaluations:")
            for result in failed_results:
                print(f"  📄 {result['filename']}: {result['status']}")
        
        return len(successful_results) > 0
    
    def _calculate_overall_metrics(self):
        """
        Calculate and display overall evaluation metrics.
        """
        print("📊 OVERALL EVALUATION RESULTS")
        print("=" * 50)
        
        successful_results = [r for r in self.all_results if r['status'] == 'Success']
        failed_results = [r for r in self.all_results if r['status'] != 'Success']
        
        print(f"✅ Successfully evaluated: {len(successful_results)} file(s)")
        print(f"❌ Failed to evaluate: {len(failed_results)} file(s)")
        
        # Initialize variables for summary
        self.overall_accuracy = 0
        self.macro_precision = self.macro_recall = self.macro_f1 = 0
        self.weighted_precision = self.weighted_recall = self.weighted_f1 = 0
        
        if self.all_predictions and self.all_ground_truth:
            # Overall accuracy
            if self.sklearn_available:
                self.overall_accuracy = accuracy_score(self.all_ground_truth, self.all_predictions)
                
                # Overall precision, recall, F1-score
                overall_precision, overall_recall, overall_f1, overall_support = precision_recall_fscore_support(
                    self.all_ground_truth, self.all_predictions, average=None, 
                    labels=['positive', 'neutral', 'negative'], zero_division=0
                )
                
                labels = ['positive', 'neutral', 'negative']
                print(f"\n📈 Overall Metrics by Class:")
                for i, label in enumerate(labels):
                    if i < len(overall_precision):
                        print(f"  {label.capitalize()}:")
                        print(f"    Precision: {overall_precision[i]:.3f}")
                        print(f"    Recall: {overall_recall[i]:.3f}")
                        print(f"    F1-Score: {overall_f1[i]:.3f}")
                        print(f"    Support: {overall_support[i] if i < len(overall_support) else 0}")
                
                # Macro and weighted averages
                self.macro_precision = np.mean(overall_precision)
                self.macro_recall = np.mean(overall_recall)
                self.macro_f1 = np.mean(overall_f1)
                
                self.weighted_precision, self.weighted_recall, self.weighted_f1, _ = precision_recall_fscore_support(
                    self.all_ground_truth, self.all_predictions, average='weighted', zero_division=0
                )
                
                print(f"\n📊 Overall Averages:")
                print(f"  Macro Precision: {self.macro_precision:.3f}")
                print(f"  Macro Recall: {self.macro_recall:.3f}")
                print(f"  Macro F1-Score: {self.macro_f1:.3f}")
                print(f"  Weighted Precision: {self.weighted_precision:.3f}")
                print(f"  Weighted Recall: {self.weighted_recall:.3f}")
                print(f"  Weighted F1-Score: {self.weighted_f1:.3f}")
            else:
                # Basic accuracy calculation
                correct = sum(1 for true, pred in zip(self.all_ground_truth, self.all_predictions) if true == pred)
                self.overall_accuracy = correct / len(self.all_ground_truth) if len(self.all_ground_truth) > 0 else 0
                print(f"📈 Basic Metrics:")
                print(f"  Overall Accuracy: {self.overall_accuracy:.3f}")
                
            print(f"🎯 Overall Accuracy: {self.overall_accuracy:.3f}")
            
            # Overall confusion matrix
            overall_cm_path = os.path.join(self.output_folder, "overall_confusion_matrix.png")
            self.create_confusion_matrix_plot(
                self.overall_cm,
                ['Positive', 'Neutral', 'Negative'],
                'Overall Confusion Matrix - All Places',
                overall_cm_path
            )
            
            # Overall distribution
            pred_dist = Counter(self.all_predictions)
            true_dist = Counter(self.all_ground_truth)
            print(f"\n📊 Overall Distribution:")
            print(f"  Predicted: {dict(pred_dist)}")
            print(f"  Ground Truth: {dict(true_dist)}")
    
    def _save_detailed_results(self):
        """
        Save detailed evaluation results to Excel file.
        """
        detailed_results = []
        for result in self.all_results:
            if result['status'] == 'Success':
                row = {
                    'filename': result['filename'],
                    'place_name': result['filename'].replace('_preprocessed_testing.xlsx', ''),
                    'total_samples': result['total_samples'],
                    'valid_samples': result['valid_samples'],
                    'accuracy': result['accuracy'],
                    'precision_positive': result['precision'].get('positive', 0),
                    'precision_neutral': result['precision'].get('neutral', 0),
                    'precision_negative': result['precision'].get('negative', 0),
                    'recall_positive': result['recall'].get('positive', 0),
                    'recall_neutral': result['recall'].get('neutral', 0),
                    'recall_negative': result['recall'].get('negative', 0),
                    'f1_positive': result['f1_score'].get('positive', 0),
                    'f1_neutral': result['f1_score'].get('neutral', 0),
                    'f1_negative': result['f1_score'].get('negative', 0),
                    'support_positive': result['support'].get('positive', 0),
                    'support_neutral': result['support'].get('neutral', 0),
                    'support_negative': result['support'].get('negative', 0),
                    'predicted_positive': result['predicted_distribution'].get('positive', 0),
                    'predicted_neutral': result['predicted_distribution'].get('neutral', 0),
                    'predicted_negative': result['predicted_distribution'].get('negative', 0),
                    'true_positive': result['true_distribution'].get('positive', 0),
                    'true_neutral': result['true_distribution'].get('neutral', 0),
                    'true_negative': result['true_distribution'].get('negative', 0),
                    'status': 'Success'
                }
            else:
                row = {
                    'filename': result['filename'],
                    'place_name': result['filename'].replace('_preprocessed_testing.xlsx', ''),
                    'total_samples': result.get('total_samples', 0),
                    'valid_samples': 0,
                    'accuracy': 0,
                    'status': result['status']
                }
            detailed_results.append(row)
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_results)
        detailed_path = os.path.join(self.output_folder, "evaluation_detailed_results.xlsx")
        detailed_df.to_excel(detailed_path, index=False)
        print(f"\n📋 Detailed results saved: {detailed_path}")
    
    def _create_summary_report(self):
        """
        Create and save summary report.
        """
        successful_results = [r for r in self.all_results if r['status'] == 'Success']
        
        if successful_results:
            summary_data = {
                'Metric': [],
                'Value': []
            }
            
            # Add overall metrics
            if self.all_predictions and self.all_ground_truth:
                summary_data['Metric'].extend([
                    'Total Files Evaluated',
                    'Successful Evaluations',
                    'Failed Evaluations',
                    'Total Samples',
                    'Overall Accuracy'
                ])
                
                summary_data['Value'].extend([
                    len(self.all_results),
                    len(successful_results),
                    len(self.all_results) - len(successful_results),
                    len(self.all_predictions),
                    f"{self.overall_accuracy:.3f}"
                ])
                
                # Add sklearn metrics if available
                if self.sklearn_available:
                    summary_data['Metric'].extend([
                        'Macro Precision',
                        'Macro Recall', 
                        'Macro F1-Score',
                        'Weighted Precision',
                        'Weighted Recall',
                        'Weighted F1-Score'
                    ])
                    
                    summary_data['Value'].extend([
                        f"{self.macro_precision:.3f}",
                        f"{self.macro_recall:.3f}",
                        f"{self.macro_f1:.3f}",
                        f"{self.weighted_precision:.3f}",
                        f"{self.weighted_recall:.3f}",
                        f"{self.weighted_f1:.3f}"
                    ])
            
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.output_folder, "evaluation_summary.xlsx")
            summary_df.to_excel(summary_path, index=False)
            print(f"📋 Summary report saved: {summary_path}")
    
    def get_results(self):
        """
        Get the results of evaluation.
        
        Returns:
            dict: Summary of results
        """
        return {
            'all_results': self.all_results,
            'overall_accuracy': getattr(self, 'overall_accuracy', 0),
            'total_samples': len(self.all_predictions),
            'successful_files': len([r for r in self.all_results if r['status'] == 'Success']),
            'failed_files': len([r for r in self.all_results if r['status'] != 'Success']),
            'predictions': self.all_predictions,
            'ground_truth': self.all_ground_truth
        }

def main():
    """
    Main function for standalone execution of sentiment evaluation.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate sentiment analysis accuracy against Google ratings')
    parser.add_argument('--testing', default='result/testing',
                      help='Testing folder containing testing data (default: result/testing)')
    parser.add_argument('--output', default='result/testing',
                      help='Output folder for evaluation results (default: result/testing)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Skip generating confusion matrix plots')
    
    args = parser.parse_args()
    
    print("🧪 STANDALONE SENTIMENT EVALUATOR")
    print("=" * 50)
    print(f"📂 Testing folder: {args.testing}")
    print(f"📂 Output folder: {args.output}")
    print(f"📊 Generate plots: {not args.no_plots}")
    print("=" * 50)
    print()
    
    # Create evaluator instance
    evaluator = SentimentEvaluator(
        testing_folder=args.testing,
        output_folder=args.output,
        generate_plots=not args.no_plots
    )
    
    # Evaluate testing data
    success = evaluator.evaluate_testing_data()
    
    if success:
        results = evaluator.get_results()
        print("\n🎉 EVALUATION COMPLETED!")
        print("=" * 50)
        print(f"🎯 Overall Accuracy: {results['overall_accuracy']:.3f}")
        print(f"📄 Total samples: {results['total_samples']:,}")
        print(f"✅ Successful files: {results['successful_files']}")
        print(f"❌ Failed files: {results['failed_files']}")
        
        if SKLEARN_AVAILABLE and results['total_samples'] > 0:
            print(f"📊 Macro Precision: {results.get('macro_precision', 0):.3f}")
            print(f"📊 Macro Recall: {results.get('macro_recall', 0):.3f}")
            print(f"📊 Macro F1-Score: {results.get('macro_f1', 0):.3f}")
    else:
        print("\n❌ EVALUATION FAILED!")
        print("Check the output above for details.")

if __name__ == "__main__":
    main()
