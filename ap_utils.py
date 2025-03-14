import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import time
from sklearn.metrics._ranking import _binary_clf_curve
import warnings

def partial_average_precision_score(target, prediction):
    # both should be shape (n, c)
    return partial_scores(target, prediction, average_precision_score)

def partial_auc_score(target, prediction):
    return partial_scores(target, prediction, roc_auc_score)

def partial_scores(target, prediction, function):
    class_aps = []
    target[target>.5] = 1
    for c in range(len(target[0])):
        this_target = target[:, c]
        this_preds = prediction[:, c]
        inds = np.where(this_target!=0)[0]
        if len(inds) == 0:
            continue
        this_target = this_target[inds]
        this_target[this_target==-1] = 0
        if len(np.unique(this_target)) != 2:
            continue
        this_ap = function(this_target, this_preds[inds])
        class_aps.append(this_ap)
    return np.mean(class_aps)

def normalized_ap_slow(targets, scores, N=-1):
    if N == -1:
        N = np.sum(targets)
    start = time.time()
    sorted_score_idxs = np.argsort(scores).squeeze()
    sorted_scores = scores[sorted_score_idxs]
    sorted_targets = targets[sorted_score_idxs]
    actual_N = targets.sum()
    all_recalls = [0] 
    all_precisions = []
    correct_pos = 0 
    wrong_pos = 0 
    for i in range(len(sorted_scores)-1,-1, -1):
        if sorted_targets[i]==1:
            correct_pos+=1
        else:
            wrong_pos+=1
        all_precisions.append(((correct_pos/actual_N)*N)/((correct_pos/actual_N)*N+wrong_pos))
        all_recalls.append(correct_pos/actual_N)
    recall_diffs = np.array(all_recalls[1:]) - np.array(all_recalls[:-1])
    all_precisions = np.array(all_precisions)
    return (recall_diffs*all_precisions).sum()

def normalized_ap(targets, score, N=-1):
    if N == -1:
        N = np.sum(targets)
    start = time.time()

    y_true, probas_pred = targets, score
    fps, tps, thresholds = _binary_clf_curve(
        y_true, probas_pred, pos_label=None, sample_weight=None
    )

    normalized_tps = tps / np.sum(y_true) * N
    normalized_ps = normalized_tps+fps
    ps = tps + fps
    precision = np.divide(tps, ps, where=(ps != 0))
    normalized_precision = np.divide(normalized_tps, normalized_ps, where=(ps != 0))

    if tps[-1] == 0:
        print("warning here")
        recall = np.ones_like(tps)
    else:
        recall = tps / tps[-1]

    # reverse the outputs so recall is decreasing
    sl = slice(None, None, -1)
    prec, rec, thresh = np.hstack((normalized_precision[sl], 1)), np.hstack((recall[sl], 0)), thresholds[sl]
    recall_diffs =  rec[:-1] - rec[1:]
    return (recall_diffs*prec[:-1]).sum()

def normalized_ap_wrapper(target, prediction, rate=-1):
    class_aps = []
    if rate == -1:
        N = np.sum(target > .5) / len(target[0])
        rate = N / len(target)

    for c in range(len(target[0])):
        this_target = target[:, c]
        this_preds = prediction[:, c]
        if len(np.unique(this_target)) != 2:
            continue
        N = len(this_target)*rate

        this_ap = normalized_ap(this_target, this_preds, N=N)
        class_aps.append(this_ap)

    return class_aps
    # return np.mean(class_aps)
