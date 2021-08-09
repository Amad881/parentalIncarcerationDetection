# Victims of Parental Incarceration Cohort Identification
An implementation of BERT for identification of youths exposed to parental incarceration via free-text clinical notes

## Components
The execution of this script relies on the following files:  
- `environment.yml`
  - The environment file to be activated through conda in order to import all necessary libraries with their proper versions
- `config.json`
  - Used to adjust various execution parameters for each script, including file input/output paths
- `processData.py`
  - Trims clinical notes within the raw data file (tsv, csv) for use in model training and evaluation
- `kFoldTrainEval.py`
  - Tokenizes, trains, and evaluates a model given processed training data. Evaluation is done over the entire dataset one slice at a time. This is the workflow used to generate our findings
- `trainEval.py`
  - Tokenizes, trains, and evaluates (batch and interactive) a model given processed training data. This can be used to train a model on train/test split data or to evaluate notes given an already trained model
- `models/NCHTrainedModel`
  - This is an example of a model trained and evaluated on our data, provided for quick evaluation of outside data sources. Please note, this model has only been fine-tuned on our dataset and therefore we **cannot** guarantee performance on outside datasets.

## General Setup
This project makes use of [conda](https://docs.conda.io/en/latest/miniconda.html) in order to maintain the necessary packages within a virtual environment. Once conda is installed, you can create the virtual environment `conda env create -f environment.yml` and load it in `conda activate torch-nlp`.

For each of the following sections, you will directly modify `config.json` to change most parameters. This file has partitioned parameters into 4 sections:
1. raw data processing
2. overall training and evaluation (used for both K-fold and non-K-fold)
3. k-fold training and evaluation
4. non-k-fold training and evaluation

## Raw Data Processing `processData.py`
This script converts an existing .csv/.tsv data file into a format the BERT model will accept. It shortens the raw text input into a trimmed chunk <=512 tokens long. This shortening is done while maximizing the number of relevant term matches (i.e. prison terminology).

### Execution
- Script can be called using `python3 processData.py`
- Raw data input file must be .csv or .tsv
- Input file must have `text`, and `label` columns. 
- Output will be a .tsv file that has a new `trimmedText` column

### Configuration
Parameters can be found in the first section of `config.json`
- `{prisonTerms}`
  - List of incarceration related terminology which will be used to center trimming such that these terms are maximally included
- `{rawData}`
  - The input raw data file which must be a .csv or .tsv
- `{processedDataPath}`
  - The output data file which now includes the trimmedText column. Must be a .tsv. This file path will be sourced for tokenization in subsequent scripts.

## General Model Training and Evaluation Configuration
The following parameters within the second section of `config.json` are used for both k-fold and non-k-fold training and evaluation
- `{logDirectory}`
  -  During training, checkpoints and other log information is stored at this path
- `{baselinePositiveRate}`
  - This marks the distribution of positive occurences of parental incarceration within the data sample. This will be marked on evaluation plots to show predictive performance of the model against a naive model which always marks a case as positive
- `{evalOnDataFrame}`
  - Appends results from model evaluation (i.e. predicted label and confidence) as new columns on the `{processedDataPath}` .tsv. This will ouput to `{kFoldResultPath}` or `{resultsPath}` depending on which evaluation script is used.
- `{earlyStoppingPatience}`
  - We have implemented early stopping for training where the model will train up to 100 epochs unless it notices no improvement for X epochs. The value for X is defined by `{earlyStoppingPatience}`. Upon stopping, the best performing epoch is selected.

## K-Fold Training and Evaluation: `kFoldTrainEval.py`
This script can be used to replicate the training and evaluation used by our work. Namely, it will conduct a k-fold training an evaluation such that k models are trained/validated on ((k-1)/k)-size slices and evaluated over the complementary slice of data. This evaluation is therefore over the entire dataset as each model evaluates over the leftover data. 

### Execution
This script has 3 main operations:
- `tokenize`
  - This will import the processedData from `{processedDataPath}` and output a tokenized dataset with k-folds. This dataset will also be saved as a binary pickle to `{kFoldTokenizedDataPath}`
- `train`
  - Takes in the tokenized fold data and trains k models, evaluating at each fold. Each model is saved to `{kFoldModelPath}` with a number from 0 to k-1 appended to the end of the path. This script will output a k-long list of dictionaries which contain evaluation metrics for each fold. This list of dictionaries is saved to `{kFoldResultsPath}/kFoldResultsDict.pkl`. If `{evalOnDataFrame}` is `true`, this script will output `{kFoldResultsPath}/kFoldResults.tsv`.
- `plot`
  - Takes in `{kFoldResultsPath}/kFoldResultsDict.pkl` and outputs a precision-recall curve to `{kFoldResultsPath}/kFoldResultCurve.png`. If `{plotEachFold}` is `true`, the output plot will contain a precision-recall curve for each fold alongside the curve for the combined folds.

Execution takes the form of `python3 kFoldTrainEval.py {tokenize/train/plot}` where `tokenize`, `train`, `plot` can be independently included or omitted to acheive the desired function. If multiple flags are used, the pipeline directly passes the output of one operation to the other, otherwise the data inputs and outputs are automatically sourced according to `config.json`. For example, `python3 kFoldTrainEval tokenize train eval` will execute the full pipleine while `python3 kFoldTrainEval train eval` will assume tokenization has already occured and gather the appropriate input data from `{kFoldTokenizedPath}`. 

### Configuration
The following parameters within the third section of `config.json` are used for k-fold training and evaluation:
- `{kFoldTokenizedDataPath}`
  -  Where to output the tokenized data. Saved as a .pkl binary
- `{kFoldModelPath}`
  - Where to store the models once trained. The fold number (from 0 to k-1) will automatically be appended to the end of this path
- `{kFoldResultsPath}`
  - Where to store evaluation pickles, .tsv files, and plot images
- `{numK}`
  - The number of folds desired
- `{kFoldValSplitSize}`
  - The size of the validation data split as a percentage of the original data size. Training and test sizes are automatically determined as a function of K
- `{plotEachFold}`
  - Whether or not to plot the precision-recall curve from each fold of evaluation
  
## General Training and Evaluation `trainEval.py`
This script allows for the training and evaluation of a model using a general train-validation-test split. Additionally, this script is able to load in a model and evaluate any given dataset (.tsv), or interactively evaluate clinical notes as they are entered.

### Execution
This script has 6 main operations:
- `tokenizeSplit`
  - Creates a train-test-validation split of `{processedDataPath}` and outputs tokenized data. Split sizes are determined by `{testSplitSize}` / `{valSplitSize}`. Tokenized data is output to `{tokenizedSplitDataPath}`
- `tokenizeEval`
  - Tokenizes `{processedDataPath}` as one uniform dataset without splits. Tokenized data is output to `{tokenizedEvalDataPath}`
- `train`
  - Fine-tunes the model using `{tokenizedSplitDataPath}`, outputting the trained model. The trained model is saved to `{modelPathOutput}`
- `eval`
  - Uses `{modelPathInput}` to evaluate over a tokenized dataset. If `{evalSplitData}` is `true`, `{tokenizedSplitDataPath}` is used. Otherwise evaluation is done on `{tokenizedEvalDataPath}`. Outputs a pickled dictionary of results to `{resultsPath}/resultDict.pkl`. If `{evalOnDataFrame}` is `true`, it will also output to `{resultsPath}/evalOutput.tsv`
- `plot`
  - Uses `{resultsPath}/resultDict.pkl` to plot the precision-recall curve over the test split. The curve is output to `{resultsPath}/resultCurve.png`
- `interactive`
  - Loads in `{modelPathInput}` and allows the user to enter in a clinical note for evaluation. The note will automatically be trimmed according to `{prisonTerms}` before returning the predicted label and confidence. The interactive session can be escaped by entering 'q' or 'quit'

Execution takes the form of `python3 trainEval.py {tokenizeSplit/tokenizeEval/train/eval/plot/interactive}` where `tokenizeSplit`, `tokenizeEval`, `train`, `eval`, `plot`, `interactive` can be independently included or omitted to acheive the desired function. If multiple flags are used, the pipeline directly passes the output of one operation to the other, otherwise the data inputs and outputs are automatically sourced according to `config.json`. For example, `python3 trainEval.py tokenizeSplit train eval plot` will execute the full pipleine while `python3 trainEval.py train interactive` will assume tokenization has already occured and gather the appropriate input data from `{tokenizedSplitDataPath}` before loading up an interactive session for live note evaluation. Likewise `python3 trainEval.py tokenizeEval eval plot` will source in whichever model is in `{modelPathInput}`, tokenize whichever dataset is in `{tokenizedEvalDataPath}` and output the evaluation results accordingly.

### Configuration
The following parameters within the fourth (last) section of `config.json` are used for general training and evaluation:
- `{tokenizedSplitDataPath}`
  - Where to output the train-validation-test split tokenized data. Saved as a .pkl binary
- `{tokenizedEvalDataPath}`
  - Where to output the evaluation tokenized data. Saved as a .pkl binary
- `{evalSplitData}`
  - If true, model evaluated on `{tokenizedSplitDataPath}`
- `{modelPathOutput}`
  - Where to output the trained model
- `{modelPathInput}`
  - Which model to use for evaluation
- `{resultsPath}`
  - Where to store evaluation pickles, .tsv, and plots
- `{testSplitSize}`
  - The size of the test data split as a percentage of the original data size.
- `{valSplitSize}`
  - The size of the validation data split as a percentage of the original data size.
