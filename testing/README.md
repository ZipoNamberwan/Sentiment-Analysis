# Sentiment Analysis Testing & Evaluation

This folder contains scripts for generating testing data and evaluating sentiment analysis accuracy.

## 📁 Files Overview

- `data_generator.py` - Standalone script for generating testing data
- `sentiment_evaluator.py` - Standalone script for evaluating sentiment accuracy
- `combined_testing.py` - Combined script that runs both steps
- `generate data.py` - Original generation script (legacy)
- `testing.py` - Original evaluation script (legacy)

## 🚀 Usage Options

### Option 1: Combined Pipeline (Recommended)

Run both data generation and evaluation in sequence:

```bash
# Full pipeline with default settings
python combined_testing.py

# Generate only
python combined_testing.py --generate-only

# Evaluate only (requires existing testing data)
python combined_testing.py --evaluate-only

# Custom settings
python combined_testing.py --source result/all --output result/testing --percentage 0.15

# Skip confusion matrix plots
python combined_testing.py --no-plots
```

### Option 2: Standalone Data Generation

Generate testing data only:

```bash
# Default settings (20% sample from result/all to result/testing)
python data_generator.py

# Custom settings
python data_generator.py --source result/all --output result/testing --percentage 0.25
```

### Option 3: Standalone Evaluation

Evaluate existing testing data only:

```bash
# Default settings (evaluate files in result/testing)
python sentiment_evaluator.py

# Custom settings
python sentiment_evaluator.py --testing result/testing --output result/testing

# Skip plots
python sentiment_evaluator.py --no-plots
```

## 📊 Command Line Arguments

### Combined Testing (`combined_testing.py`)

- `--generate-only` - Only generate testing data, skip evaluation
- `--evaluate-only` - Only evaluate existing testing data, skip generation  
- `--source` - Source folder with sentiment analysis results (default: result/all)
- `--output` - Output folder for testing data and results (default: result/testing)
- `--percentage` - Percentage of data to sample (default: 0.20)
- `--no-plots` - Skip generating confusion matrix plots

### Data Generator (`data_generator.py`)

- `--source` - Source folder with sentiment analysis results (default: result/all)
- `--output` - Output folder for testing data (default: result/testing)
- `--percentage` - Percentage of data to sample (default: 0.20)

### Sentiment Evaluator (`sentiment_evaluator.py`)

- `--testing` - Testing folder with testing data (default: result/testing)
- `--output` - Output folder for evaluation results (default: result/testing)
- `--no-plots` - Skip generating confusion matrix plots

## 📈 What Each Script Does

### Data Generation
1. Reads sentiment analysis results from `result/all` folder
2. Samples 20% of rows from each file
3. Ensures balanced distribution (1/3 positive, 1/3 negative, 1/3 neutral)
4. Saves testing files with `_testing.xlsx` suffix

### Sentiment Evaluation
1. Reads testing data files from `result/testing` folder
2. Maps Google ratings to sentiment categories:
   - **Positive**: 4-5 star ratings
   - **Neutral**: 3 star ratings
   - **Negative**: 1-2 star ratings
3. Compares predicted vs. actual sentiments
4. Generates accuracy metrics, confusion matrices, and detailed reports

## 📁 Output Files

The scripts generate several output files in the `result/testing` folder:

### Testing Data Files
- `*_testing.xlsx` - Testing data files (20% sample of original data)
- `testing_data_summary.xlsx` - Summary of data generation process

### Evaluation Results
- `evaluation_detailed_results.xlsx` - Detailed metrics per file
- `evaluation_summary.xlsx` - Overall evaluation summary
- `*_confusion_matrix.png` - Individual confusion matrices per place
- `overall_confusion_matrix.png` - Combined confusion matrix

## 🎯 Understanding Results

### Accuracy Metrics
- **Overall Accuracy**: Percentage of correct predictions
- **Precision**: Of all positive predictions, how many were correct
- **Recall**: Of all actual positive cases, how many were found
- **F1-Score**: Balanced measure of precision and recall

### Good Performance Indicators
- **Accuracy > 0.7**: Model performs well
- **Balanced metrics**: Similar performance across sentiment categories
- **High diagonal values in confusion matrix**: Most predictions are correct

### Common Challenges
- **Positive bias**: Most Google reviews are 4-5 stars
- **Neutral ambiguity**: Rating 3 can be interpreted differently
- **Text vs. rating mismatch**: High rating with negative text content

## 🔧 Requirements

- pandas
- numpy
- scikit-learn (optional, for advanced metrics)
- matplotlib (optional, for plots)
- seaborn (optional, for plots)

## 💡 Tips

1. **Start with combined script**: Use `python combined_testing.py` for most cases
2. **Adjust sample size**: Use `--percentage 0.15` for smaller samples if needed
3. **Skip plots for speed**: Use `--no-plots` to run faster without visualizations
4. **Evaluate only**: Use `--evaluate-only` to re-run evaluation with different settings
