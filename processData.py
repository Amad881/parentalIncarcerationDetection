'''
Description:
This script converts an existing .csv data file into a format the BERT model will accept. 
It shortens the raw text input into a trimmed chunk <=512 tokens long. This shortening is done while maximizing the number of relevant term matches (i.e. prison terminology)

Requirements:
Method call is of the form: python3 processData.py [inputFile.csv] [outoutFile.csv]
Input and output files must both be .csv
Input file must have 'text', 'label', and 'record_id' columns. The output file will have a new trainingText column
'''
import pandas as pd
from pdb import set_trace as bp
import numpy as np
import pickle
import sys
import json

def trimToWindow(row, configDict):

    def groupConsecutive(data, stepsize=1):
        return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    # Loads prison terms from the config file
    prisonTerms = set(configDict['prisonTerms'])

    text = row['text']
    outText = text
    tokens = text.split()
    if len(tokens) > 500:
        
        tokens = text.split()
        tknPrisonCt = np.zeros(len(tokens))
        for idx, token in enumerate(tokens):
            for prisonTerm in prisonTerms:
                if prisonTerm in token:
                    tknPrisonCt[idx] = 1
        windowLen = 500
        tknRangeCt = np.zeros(len(tokens))
        for idx in range(len(tknPrisonCt)):
            tknRangeCt[idx] = np.sum(tknPrisonCt[idx: idx+windowLen])
        
        maxWindowVal = np.max(tknRangeCt)
        bestIdxs = np.where(tknRangeCt==maxWindowVal)[0]

        sequences = groupConsecutive(bestIdxs)
        seqMids = np.zeros(len(sequences))
        for idx, sequence in enumerate(sequences):
            seqMids[idx] = int(np.median(sequence))

        textMid = int(len(tokens)/2)
        startIdx = int(find_nearest(seqMids, textMid))
        if (startIdx + windowLen) > len(tokens):
            startIdx = int(len(tokens) - windowLen)

        outToks = tokens[startIdx:startIdx+windowLen]
        outText = " ".join(outToks)

    return outText

def main():
    with open('config.json') as fin:
        configDict = json.load(fin)
    rawDataFile = configDict['rawData']
    outputFile = configDict['processedDataPath']

    if '.tsv' in rawDataFile:
        df = pd.read_csv(rawDataFile, sep="\t")
    else:
        df = pd.read_csv(rawDataFile)
    df['trainingText'] = df.apply(lambda row: trimToWindow(row, configDict), axis=1)
    df.to_csv(outputFile, sep="\t", index=False)
    return 0

if __name__ == "__main__":
    main()