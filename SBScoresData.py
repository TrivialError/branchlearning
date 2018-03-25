import _pickle
import uuid
import gzip


class SBScoresData:

    # n is the number of nodes
    # lp_soln is the matrix containing lp solution values for edges between vertices
    # soln_adj_mat is the matrix obtained by replacing all non-zero values of lp_soln by 1
    # adj_mat is an adjacency matrix for the graph
    # weight_mat is an adjacency matrix for the graph with each nonzero value scaled by its edge weight
    # edge is a tuple of the two nodes for which the SB value of the corresponding edge has been calculated
    # sb_label is the SB score of the given edge
    def __init__(self, name, n, lp_soln, soln_adj_mat, adj_mat, weight_mat, var_sb_label_dict, train=True):
        self.name = name + "-" + str(uuid.uuid4())
        self.n = n
        self.lp_soln = lp_soln
        self.soln_adj_mat = soln_adj_mat
        self.adj_mat = adj_mat
        self.weight_mat = weight_mat
        self.var_sb_label_dict = var_sb_label_dict
        self.train = train

    def save(self):
        print("writing data to a file")
        with gzip.open("./Data/" + self.name, "wb") as f:
            _pickle.dump(self, f)

