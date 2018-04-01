from test_on_model import *
from neural_networks import *
from model_evaluation import *

model_name_pre = './models/graph_embedding_nn_no_'
model_list = [250, 300, 350, 400, 450, 500, 550, 600]

for m_n in model_list:
    model_name = model_name_pre + str(m_n)
    pre, org = test_on_model_dir(model_name, './DataTest')
    precision = gen_precision(pre, org)
    recall = gen_recall(pre, org)
    accuracy = gen_accuracy(pre, org)
    print ('model_name:', m_n, 'precision:', precision, 'recall:', recall, 'accuracy:',accuracy)
