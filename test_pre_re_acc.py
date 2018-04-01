from test_on_model import *
from neural_networks import *
from model_evaluation import *
pre, org = test_on_model_dir('', './DataTest')
precision = gen_precision(pre, org)
recall = gen_recall(pre, org)
accuracy = gen_accuracy(pre, org)
print ('precision:', precision, 'recall:', recall, 'accuracy:',accuracy)
