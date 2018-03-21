import pickle

data = pickle.load(open("./Data/1feb5f02-582f-49bb-8c48-169448e311e6", 'rb'))

print(data.name)
print(data.n)
print(data.lp_soln)
print(data.soln_adj_mat)
print(data.adj_mat)
print(data.weight_mat)
print(data.edge)
print(data.sb_label)
