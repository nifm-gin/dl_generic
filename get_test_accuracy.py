import nibabel as nib
import os
import numpy as np
import sys
import argparse
from multiprocessing import Process
import pandas as pd

def compute_accuracy(preds, gt, threshold):
    preds = (preds > threshold).astype(np.int)
    similarity = (preds == gt).astype(np.int)
    return np.mean(similarity)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments for patches creation.')
    parser.add_argument("--model_path", type=str, required = True,
                        help='Final Model path with inference results')
    parser.add_argument('--gt_path', type=str, required = True,
                        help='GT CSV path')
    
    parser.add_argument('--cross_val', '-cv', action="store_true")
    args = parser.parse_args()
    gt = pd.read_csv(args.gt_path, names = ["Group", "Patient"], index_col = False,skiprows = 1,squeeze=True).set_index('Patient')['Group'].to_dict()
    score = 0
#    state_score = 0
    total = 0
    predictions = []
    ground_truth = []
    if args.cross_val:
        for i in range(10):
            preds =pd.read_csv("%s_%s/infered/predictions.csv" % (args.model_path, i), names = ["subject_id", "group", "slice", "prob"], index_col = False,skiprows = 1,squeeze=True).set_index('subject_id').to_dict("index")
            for sub in preds.keys():
                predictions.append(preds[sub]['prob'])
                #predictions.append(float(preds[sub]['prob'].replace('[', '').replace(']', '')))
                ground_truth.append(gt[sub])

            # print("Test accuracy for fold %s : %s !" % (i, current_score / current_total))
            # #print("Test accuracy on patients state for fold %s : %s !" % (i, current_state_score / current_total))
            # print("Test vs. preds classes for fold %s : %s / %s " % (i, test_gt, list(preds.values())))
            #score += current_score; total += current_total
    else:
        preds =pd.read_csv("%s/infered/predictions.csv" % args.model_path, names = ["subject_id", "group", 'slice', "prob"], index_col = False,skiprows = 1,squeeze=True).set_index('subject_id')['group'].to_dict()
        for sub in preds.keys():
            if "sub-4" in sub:
                total += 1
                test_gt.append(gt[sub])
                if preds[sub] == gt[sub]:
                    score += 1 ; state_score += 1
                elif preds[sub] < 2 and gt[sub] < 3:
                    #Control / patient accuracy
                    state_score += 1
                    
    predictions, ground_truth = np.asarray(predictions), np.asarray(ground_truth)
    acc_max, best_th = 0, 0
    for threshold in np.arange(0.0, 1.0001, 0.0001):
        acc = compute_accuracy(predictions, ground_truth, threshold)
        if acc >= acc_max:
            acc_max = acc; best_th = threshold

    print("Test accuracy : %s ! \n\tBest threshold found : %s" % (acc_max, best_th))
    
    #print("Test accuracy on patients state : %s !" % (state_score / total))
