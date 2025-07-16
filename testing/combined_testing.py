"""
Combined script for sentiment analysis testing and evaluation.

This script performs two main functions:
1. Generate testing data: Takes 20% of rows from each file in result/all folder 
   with balanced sentiment distribution (1/3 each for positive, negative, neutral)
2. Evaluate testing data: Compare predicted sentiment with Google review ratings
   - Positive: 4-5 star ratings
   - Neutral: 3 star ratings  
   - Negative: 1-2 star ratings

Usage examples:
- Run both steps: python combined_testing.py
- Generate data only: python combined_testing.py --generate-only
- Evaluate only: python combined_testing.py --evaluate-only
- Custom folders: python combined_testing.py --source result/all --output result/testing
"""

import argparse
from data_generator import TestingDataGenerator
from sentiment_evaluator import SentimentEvaluator

def main():
    """
    Main function to run both testing data generation and evaluation.
    """
    parser = argparse.ArgumentParser(description='Combined sentiment analysis testing and evaluation')
    parser.add_argument('--generate-only', action='store_true',
                      help='Only generate testing data, skip evaluation')
    parser.add_argument('--evaluate-only', action='store_true',
                      help='Only evaluate existing testing data, skip generation')
    parser.add_argument('--source', default='result/all',
                      help='Source folder containing sentiment analysis results (default: result/all)')
    parser.add_argument('--output', default='result/testing',
                      help='Output folder for testing data and results (default: result/testing)')
    parser.add_argument('--percentage', type=float, default=0.20,
                      help='Percentage of data to sample for testing (default: 0.20)')
    parser.add_argument('--no-plots', action='store_true',
                      help='Skip generating confusion matrix plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.generate_only and args.evaluate_only:
        print("❌ Error: Cannot use both --generate-only and --evaluate-only")
        return
    
    print("🚀 SENTIMENT ANALYSIS TESTING & EVALUATION PIPELINE")
    print("=" * 60)
    
    if args.generate_only:
        print("Mode: Generate testing data only")
    elif args.evaluate_only:
        print("Mode: Evaluate existing testing data only")
    else:
        print("Mode: Full pipeline (generate + evaluate)")
    
    print("=" * 60)
    print()
    
    generation_success = True
    evaluation_success = True
    
    # Step 1: Generate testing data (unless evaluate-only)
    if not args.evaluate_only:
        print("🚀 STEP 1: GENERATE TESTING DATA")
        print("=" * 50)
        
        data_generator = TestingDataGenerator(
            source_folder=args.source,
            output_folder=args.output,
            sample_percentage=args.percentage
        )
        
        generation_success = data_generator.generate_testing_data()
        
        if not generation_success:
            print("❌ Testing data generation failed.")
            if not args.generate_only:
                print("Stopping pipeline.")
                return
        else:
            # Get generation results
            gen_results = data_generator.get_results()
            print(f"\n📊 Generation Summary:")
            print(f"  ✅ Successful files: {gen_results['successful_files']}")
            print(f"  ❌ Failed files: {gen_results['failed_files']}")
            print(f"  📄 Total rows generated: {gen_results['total_testing_rows']:,}")
            print()
    
    # Step 2: Evaluate testing data (unless generate-only)
    if not args.generate_only and generation_success:
        print("🚀 STEP 2: EVALUATE TESTING DATA")
        print("=" * 50)
        
        evaluator = SentimentEvaluator(
            testing_folder=args.output,
            output_folder=args.output
        )
        
        evaluation_success = evaluator.evaluate_testing_data()
        
        if evaluation_success:
            # Get evaluation results
            eval_results = evaluator.get_results()
            
            print(f"\n📊 Evaluation Summary:")
            print(f"  🎯 Overall Accuracy: {eval_results['overall_accuracy']:.3f}")
            print(f"  📄 Total samples evaluated: {eval_results['total_samples']:,}")
            print(f"  ✅ Successfully evaluated: {eval_results['successful_files']} file(s)")
            print(f"  ❌ Failed evaluations: {eval_results['failed_files']} file(s)")
    
    # Final summary
    print("\n" + "=" * 60)
    if args.generate_only:
        if generation_success:
            print("🎉 DATA GENERATION COMPLETED SUCCESSFULLY!")
        else:
            print("❌ DATA GENERATION FAILED!")
    elif args.evaluate_only:
        if evaluation_success:
            print("🎉 EVALUATION COMPLETED SUCCESSFULLY!")
        else:
            print("❌ EVALUATION FAILED!")
    else:
        if generation_success and evaluation_success:
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        elif generation_success:
            print("⚠️ Pipeline partially completed - Generation successful, Evaluation failed")
        else:
            print("❌ Pipeline failed at data generation stage")
    
    print("=" * 60)
    
    if generation_success or evaluation_success:
        print(f"📁 Check the '{args.output}' folder for:")
        if not args.evaluate_only:
            print("  - Testing data files (*_testing.xlsx)")
        if not args.generate_only:
            print("  - Evaluation results (evaluation_*.xlsx)")
            if not args.no_plots:
                print("  - Confusion matrices (*.png)")
            print("  - Summary reports")

if __name__ == "__main__":
    main()
