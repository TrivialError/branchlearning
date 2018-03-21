import pickle
import uuid


class DataPoint:

    # n is the number of nodes
    # lp_soln is the matrix containing lp solution values for edges between vertices
    # soln_adj_mat is the matrix obtained by replacing all non-zero values of lp_soln by 1
    # adj_mat is an adjacency matrix for the graph
    # weight_mat is an adjacency matrix for the graph with each nonzero value scaled by its edge weight
    # edge is a tuple of the two nodes for which the SB value of the corresponding edge has been calculated
    # sb_label is the SB score of the given edge
    def __init__(self, n, lp_soln, soln_adj_mat, adj_mat, weight_mat, edge, sb_label):
        self.name = str(uuid.uuid4())
        self.n = n
        self.lp_soln = lp_soln
        self.soln_adj_mat = soln_adj_mat
        self.adj_mat = adj_mat
        self.weight_mat = weight_mat
        self.edge = edge
        self.sb_label = sb_label

    def save(self):
        print("writing data to a file")
        pickle.dump(self, open("./Data/" + self.name, "wb"))

