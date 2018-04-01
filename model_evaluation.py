import numpy as np
#pre_labels:    predicated labels [[labels_0], [labels_1], ....]
#orig_labels:   original labels of the same shape as pre_labels
# pre = tp / (tp + fp)
# all labels should be int
def gen_precision(pre_labels, orig_labels):
    tp, fp = 0, 0
    count_ins = len(pre_labels)
    for i in range(count_ins):
        count_cand = len(pre_labels[i])
        for j in range(count_cand):
            if int( pre_labels[i][j] ) == 1:
                if int( pre_labels[i][j] == orig_labels[i][j] ):
                    tp = tp + 1
                else:
                    fp = fp + 1
    precision = ( tp + 0.0 ) / ( tp + fp + 0.0)
    return precision
#recall = tp / (tp + fn)
def gen_recall(pre_labels, orig_labels):
    tp, fn = 0, 0
    count_ins = len(pre_labels)
    for i in range(count_ins):
        count_cand = len(pre_labels[i])
        for j in range(count_cand):
            if int( orig_labels[i][j] ) == 1:
                if int( pre_labels[i][j] == orig_labels[i][j] ):
                    tp = tp + 1
                else:
                    fn = fn + 1
    recall = ( tp + 0.0 ) / ( tp + fn + 0.0)
    return recall
# accuracy = (tp + tn) / (tp + tn + fp + fn)
def gen_accuracy(pre_labels, orig_labels):
    count = 0
    t_count = 0
    count_ins = len(pre_labels)
    for i in range(count_ins):
        count_cand = len(pre_labels[i])
        for j in range(count_cand):
            count = count + 1
            if pre_labels[i][j] == orig_labels[i][j]:
                t_count = t_count + 1
    accuracy = ( t_count + 0.0 ) / count
    return accuracy
