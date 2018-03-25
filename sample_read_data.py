import _pickle
import gzip

with gzip.open("./Data/a280-786268f5-80a7-4ff2-9b6f-b90b741c17f0") as f:
    data = _pickle.load(f)

print("name: ", data.name)
print("n: ", data.n)
print("lp_soln: ", data.lp_soln)
print("soln_adj_mat: ", data.soln_adj_mat)
print("adj_mat: ", data.adj_mat)
print("weight_mat: ", data.weight_mat)

# Each key is an edge and each value is a SB label
print("var_sb_label_dict: ", data.var_sb_label_dict)
