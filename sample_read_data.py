import _pickle
import gzip

with gzip.open("./Data/9c581d0e-2c19-47ad-a9f2-ab9a047d2e23") as f:
    data = _pickle.load(f)

print("name: ", data.name)
print("n: ", data.n)
print("lp_soln: ", data.lp_soln)
print("soln_adj_mat: ", data.soln_adj_mat)
print("adj_mat: ", data.adj_mat)
print("weight_mat: ", data.weight_mat)

# Each key is an edge and each value is a SB label
print("var_sb_label_dict: ", data.var_sb_label_dict)
