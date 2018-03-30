import torch
from neural_networks import test_Net
from neural_networks import graph_embedding_Net    

def load_model(model_name):
    return torch.load(model_name)

def modify_model(trained_model):
    trained_dict = trained_model.state_dict()
    #print trained_dict['_theta12.weight'].size()[0]
    model = test_Net(trained_dict['_theta12.weight'].size()[0])
    model_dict = model.state_dict()
    trained_dict = {k: v for k, v in trained_dict.items() if k in model_dict}
    #model_dict.update(trained_dict)
    model.load_state_dict(trained_dict)
    return model
#data   : a list of length k, and each item is a tuple (node_a_index, node_b_index). i.e. [(1,2), [2,3], .... []]
#output  : a list of length k, each item is a scalar indicating the predication score i.e. [0.5, 0.23, 0, -1.0, ...]
def test(E, W, adj_F, adj_G, iteration_num, t_data, Num_node, bt_size, test_model):
    output = test_model(E, W, adj_F, adj_G, iteration_num, t_data, Num_node, bt_size)
    return output


#main Interface
#t_data : format no change from the orginal one
#E, adj_G, adj_F, W: the same as the original ones but the value of the first dimension should be set to 1   
#since those info is shared among the batches
#iteration_num: the value of T
#Num_node: the number of node of the instance
#bt_size : the length of t_data
def load_model_predict(model_name, t_data, E, adj_G, adj_F, W, iteration_num, Num_node, bt_size):
    model = load_model(model_name)
    test_model = modify_model(model)
    test_output = test(E, W, adj_F, adj_G, iteration_num, t_data, Num_node, bt_size)
    return test_output



