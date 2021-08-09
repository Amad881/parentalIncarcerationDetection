import pandas as pd
from pdb import set_trace as bp
import numpy as np
import pickle
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import torch.nn.functional as nnf
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, RocCurveDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split
import sys
import json

def tokenize(batch):
    return tokenizer(batch['trimmedText'], padding=True, truncation=True, max_length=500)

def dfToDataset(df):
    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize, batched=True, batch_size=len(ds))
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    return ds

def createSplitDataset(configDict):

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    
    dataDf = pd.read_csv(configDict['processedDataPath'], sep="\t")
    X = dataDf.trimmedText
    y = dataDf.label
    indices = range(len(y))
    XTrain, XTest, yTrain, yTest, idxTrain, idxTest= train_test_split(X, y, indices, test_size=configDict['testSplitSize'])
    valSize = (configDict['valSplitSize']*len(y))/len(yTrain)
    _, _, _, _, idxTrain, idxVal = train_test_split(XTrain, yTrain, idxTrain, test_size=valSize)

    trainDf = dataDf.iloc[idxTrain]
    valDf = dataDf.iloc[idxVal]
    testDf = dataDf.iloc[idxTest]

    trainDs = dfToDataset(trainDf)
    valDs = dfToDataset(valDf)
    testDs = dfToDataset(testDf)

    splitData = (trainDs, valDs, (testDs, testDf))

    if configDict['tokenizedSplitDataPath'] != False:
        with open(configDict['tokenizedSplitDataPath'], 'wb') as handle:
            pickle.dump(splitData, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("tokenized datasets")
    return splitData

def createEvalDataset(configDict):
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    dataDf = pd.read_csv(configDict['processedDataPath'])
    evalDs = dfToDataset(dataDf)
    outInfo = (evalDs, dataDf)
    if configDict['tokenizedEvalDataPath'] != False:
        with open(configDict['tokenizedEvalDataPath'], 'wb') as handle:
            pickle.dump(outInfo, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Created eval dataset")
    return outInfo

def train(configDict, tokenizedData=None):

    # This evaluation callback is used to evaluate over the validation-slice within each fold. Used to determine early stopping
    def computeMetricsSparse(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    if tokenizedData == None:
        with open(configDict['tokenizedSplitDataPath'], 'rb') as handle:
            tokenizedData = pickle.load(handle)

    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=configDict['earlyStoppingPatience']
    )

    model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    trainData = tokenizedData[0]
    valData = tokenizedData[1]

    training_args = TrainingArguments(
        output_dir=configDict['logDirectory'],
        num_train_epochs=100,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=configDict['logDirectory'],
        save_total_limit=10,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1'
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=computeMetricsSparse,
        train_dataset=trainData,
        eval_dataset=valData,
        callbacks=[early_stop_callback]
    )

    trainer.train()
    torch.save(model, configDict['modelPathOutput'])
    print("Saved model")
    return model

def evalModel(configDict, model=None, testData=None):
    # Modification to trainer object which evaluates over the test-slice and using the more detailed evaluation callback function
    def heldOutEval(model, testDataset):
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=computeMetricsDetailed,
            train_dataset=testDataset,
            eval_dataset=testDataset,
        )

        resultDict = trainer.evaluate()
        return resultDict

    # This evaluation callback is used to evaluate over the test-slice within each fold
    def computeMetricsDetailed(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        class1 = pred.predictions[:, 1]
        odds = np.exp(class1)
        probs = odds / (1+odds)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        fpr, tpr, _ = roc_curve(labels, probs)
        precRate, recRate, _ = precision_recall_curve(labels, probs)
        rocAuc = auc(fpr, tpr)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'rocAuc': rocAuc,
            'fpr': fpr,
            'tpr': tpr,
            'precRate': precRate,
            'recRate': recRate,
            'labels': labels,
            'probs': probs,
            'preds': preds
        }
    
    if model == None:
        model = torch.load(configDict['modelPathInput'])
        model.eval()

    if testData == None:
        if configDict['evalSplitData']:
            with open(configDict['tokenizedSplitDataPath'], 'rb') as handle:
                testData = pickle.load(handle)[-1]
        else:            
            with open(configDict['tokenizedEvalDataPath'], 'rb') as handle:
                testData = pickle.load(handle)
    testDf = testData[1]
    testData = testData[0]

    training_args = TrainingArguments(
        output_dir=configDict['logDirectory'],
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        logging_dir=configDict['logDirectory'],
        save_total_limit=10,
        evaluation_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1'
    )

    if configDict['evalOnDataFrame']:
        resultDict = heldOutEval(model, testData)
        testDf['tempIdx'] = range(0, len(testDf))
        testDf['predictions'] = testDf.apply(lambda row: resultDict['eval_preds'][row['tempIdx']], axis=1)
        testDf['probabilities'] = testDf.apply(lambda row: resultDict['eval_probs'][row['tempIdx']], axis=1)
        testDf.drop(columns=['tempIdx'], inplace=True)
        testDf.to_csv(configDict['resultsPath'] + 'evalOutput.tsv', sep="\t", index=False)
    print("Outputting evaluation results")
    outPkl = configDict['resultsPath'] + "resultDict.pkl"
    with open(outPkl, 'wb') as handle:
        pickle.dump(resultDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return resultDict

def evalPlot(configDict, resultDict=None):
    import matplotlib.pyplot as plt
    if resultDict == None:
        with open(configDict['resultsPath']+'resultDict.pkl', 'rb') as handle:
            resultDict = pickle.load(handle)
    f, axes = plt.subplots()


    precision, recall, _ = precision_recall_curve(resultDict['eval_labels'], resultDict['eval_probs'])
    lab = 'Overall AUC=%.3f' % (auc(recall, precision))
    axes.step(recall, precision, label=lab, lw=2, color='black')
    axes.plot([1.0], [configDict['baselinePositiveRate']], marker='o', markersize='3', color='red', label='Baseline')
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.legend(loc='lower left', fontsize='small')

    f.tight_layout()
    f.savefig(configDict['resultsPath']+'resultCurve.png')
    print("Created plot")

def main():
    with open('config.json') as fin:
        configDict = json.load(fin)

    operations = sys.argv
    if len(operations) > 1:
        operationsSet = set(operations[1:])
        tokenizedData = None
        model = None
        evalData = None
        resultDict = None
        if "tokenizeSplit" in operationsSet:
            tokenizedData = createSplitDataset(configDict)
            evalData = tokenizedData[-1]
        if "tokenizeEval" in operationsSet:
            evalData = createEvalDataset(configDict)
        if "train" in operationsSet:
            model = train(configDict, tokenizedData)
        if "eval" in operationsSet:
            resultDict = evalModel(configDict, model, evalData)
        if "plot" in operationsSet:
            evalPlot(configDict, resultDict)

if __name__ == "__main__":
    main()