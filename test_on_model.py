import os
import _pickle
import gzip
import numpy as np
import load_model_do_branching


def test_on_model(model_name, data_file):
    with gzip.open("./DataTest/" + data_file) as f:
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

    predicted_class.sort()
    max_val = predicted_class[-1]
    min_val = predicted_class[0]
    predicted_labels = [1 if val >= max_val - 0.2*(max_val-min_val) for val in predicted_class]

    return data_labels, predicted_labels
