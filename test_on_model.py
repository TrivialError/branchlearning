import os
import _pickle
import gzip
import numpy as np
import load_model_do_branching


def test_on_model_dir(model_name, dir):

    all_data_labels = []
    all_pred_labels = []
    count = 0
    for file in os.listdir(dir):
        count = count +1
        data_labels, pred_labels = test_on_model(model_name, file, dir)
        all_data_labels.append(data_labels)
        all_pred_labels.append(pred_labels)
        if count > 2:
            break

    return all_data_labels, all_pred_labels


def test_on_model(model_name, data_file, dir):
    with gzip.open(dir + '/' + data_file) as f:
        data = _pickle.load(f)

    n = data.n

    lp_soln = np.empty((1, n, n))
    adj_mat = np.empty((1, n, n))
    soln_adj_mat = np.empty((1, n, n))
    weight_mat = np.empty((1, n, n))

    lp_soln[0] = data.lp_soln
    soln_adj_mat[0] = data.soln_adj_mat
    adj_mat[0] = data.adj_mat
    weight_mat[0] = data.weight_mat

    edges_labels = data.var_sb_label_dict.items()

    edges, data_labels = [item[0] for item in edges_labels], [item[1] for item in edges_labels]

    predicted_class = \
        load_model_do_branching.load_model_predict(model_name, edges, lp_soln, adj_mat, soln_adj_mat,
                                                   weight_mat, 3, n, len(edges))

    predicted_class = predicted_class.data.cpu().numpy()
    predicted_class.sort()
    max_val = predicted_class[-1]
    min_val = predicted_class[0]
    predicted_labels = [1 for val in predicted_class if val >= max_val - 0.2*(max_val-min_val)]

    return data_labels, predicted_labels
