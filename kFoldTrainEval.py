import pandas as pd
from pdb import set_trace as bp
import numpy as np
import pickle
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import torch
import torch.nn.functional as nnf
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, roc_curve, auc, RocCurveDisplay, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
import sys
import json

def createDataset(configDict):
    def tokenize(batch):
        return tokenizer(batch['trainingText'], padding=True, truncation=True, max_length=500)

    tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    dataDf = pd.read_csv(configDict['processedDataPath'], sep="\t")
    skf = StratifiedKFold(n_splits=configDict['numK'])
    X = dataDf.trainingText
    y = dataDf.label
    folds = []
    foldsIdx = []
    for trainValIdx, testIdx in skf.split(X, y):
        valSize = int(configDict['kFoldValSplitSize']*len(y))
        trainIdx = trainValIdx[valSize:]
        valIdx = trainValIdx[:valSize]

        trainDf = dataDf.iloc[trainIdx]
        valDf = dataDf.iloc[valIdx]
        testDf = dataDf.iloc[testIdx]

        trainDs = Dataset.from_pandas(trainDf)
        trainDs = trainDs.map(tokenize, batched=True, batch_size=len(trainDs))
        trainDs.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        valDs = Dataset.from_pandas(valDf)
        valDs = valDs.map(tokenize, batched=True, batch_size=len(trainDs))
        valDs.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

        testDs = Dataset.from_pandas(testDf)
        testDs = testDs.map(tokenize, batched=True, batch_size=len(trainDs))
        testDs.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
        folds.append((trainDs, valDs, (testDs, testDf)))
    
    if configDict['kFoldTokenizedDataPath'] != False:
        with open(configDict['kFoldTokenizedDataPath'], 'wb') as handle:
            pickle.dump(folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("tokenized datasets")
    return folds

def trainEval(configDict, tokenizedData=None):

    # Modification to trainer object which evaluates over the test-slice and using the more detailed evaluation callback function
    def heldOutEval(model, testDataset):
        trainer = Trainer(
            model=model,
            args=training_args,
            compute_metrics=computeMetricsDetailed,
            train_dataset=trainData,
            eval_dataset=testDataset,
            callbacks=[early_stop_callback]
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
        with open(configDict['kFoldTokenizedDataPath'], 'rb') as handle:
            tokenizedData = pickle.load(handle)

    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience= configDict['earlyStoppingPatience']
    )

    allResults = []
    totalDf = None
    for foldId, foldData in enumerate(tokenizedData):
        model = BertForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        trainData = foldData[0]
        valData = foldData[1]
        testData = foldData[2][0]
        testDf = foldData[2][1]

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

        print("starting training fold: " + str(foldId))
        trainer.train()
        torch.save(model, configDict['kFoldModelPath'] + str(foldId))
        resultDict = heldOutEval(model, testData)
        allResults.append(resultDict)
        if configDict['evalOnDataFrame']:
            testDf['tempIdx'] = range(0, len(testDf))
            testDf['predictions'] = testDf.apply(lambda row: resultDict['eval_preds'][row['tempIdx']], axis=1)
            testDf['probabilities'] = testDf.apply(lambda row: resultDict['eval_probs'][row['tempIdx']], axis=1)
            testDf.drop(columns=['tempIdx'], inplace=True)
            if type(totalDf) == type(None):
                totalDf = testDf
            else:
                totalDf = pd.concat([totalDf, testDf])
        print("Done training fold: " + str(foldId))

    print("Outputting results")
    outPkl = configDict['kFoldResultsPath'] + "kFoldResultDict.pkl"
    with open(outPkl, 'wb') as handle:
        pickle.dump(allResults, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    if type(totalDf) != type(None):
        totalDf.to_csv(configDict['kFoldResultsPath'] + 'kFoldResults.tsv', sep="\t", index=False)
    return allResults

def crossValPlot(configDict, foldsList=None):
    import matplotlib.pyplot as plt
    if foldsList == None:
        with open(configDict['kFoldResultsPath']+'kFoldResultDict.pkl', 'rb') as handle:
            foldsList = pickle.load(handle)

    f, axes = plt.subplots()

    yReal = []
    yProb = []
    for foldId, foldDict in enumerate(foldsList):
        precision = foldDict['eval_precRate']
        recall = foldDict['eval_recRate']
        yReal.append(foldDict['eval_labels'])
        yProb.append(foldDict['eval_probs'])

        if configDict['plotEachFold']:
            lab = 'Fold %d AUC=%.3f' % (foldId+1, auc(recall, precision))
            axes.step(recall, precision, label=lab)

    yReal = np.concatenate(yReal)
    yProb = np.concatenate(yProb)

    precision, recall, _ = precision_recall_curve(yReal, yProb)
    lab = 'Overall AUC=%.3f' % (auc(recall, precision))
    axes.step(recall, precision, label=lab, lw=2, color='black')
    axes.plot([1.0], [configDict['baselinePositiveRate']], marker='o', markersize='3', color='red', label='Baseline')
    axes.set_xlabel('Recall')
    axes.set_ylabel('Precision')
    axes.legend(loc='lower left', fontsize='small')

    f.tight_layout()
    f.savefig(configDict['kFoldResultsPath']+'kFoldResultCurve.png')
    print("Completed plot")

def main():
    with open('config.json') as fin:
        configDict = json.load(fin)

    operations = sys.argv
    if len(operations) > 1:
        operationsSet = set(operations[1:])
        tokenizedData = None
        foldsResultsList = None
        if "tokenize" in operationsSet:
            tokenizedData = createDataset(configDict)
        if "train" in operationsSet:
            foldsResultsList = trainEval(configDict, tokenizedData)
        if "plot" in operationsSet:
            crossValPlot(configDict, foldsResultsList)

if __name__ == "__main__":
    main()