import os
import _pickle
import gzip
import random
import queue
import numpy as np


class DataProcessor:
    def __init__(self):
        self.files = os.listdir("./Data")
        self.data = []
        self.epoch = 0

        for file in self.files:
            with gzip.open('./Data/' + file) as f:
                self.data.append(_pickle.load(f))

        self.current_epoch_data = queue.Queue()
        self.active_data = queue.Queue()
        self.current_n = 0
        self.current_lp_soln = None
        self.current_adj_mat = None
        self.current_soln_adj_mat = None
        self.current_weight_mat = None

    def get_data(self, batch_size):

        while self.active_data is None or self.active_data.qsize() < batch_size:
            if self.current_epoch_data.empty():
                self.epoch += 1
                random.shuffle(self.data)
                for d in self.data:
                    self.current_epoch_data.put(d)

            print("fetching new data")
            data = self.current_epoch_data.get()
            edges_labels = data.var_sb_label_dict.items()
            for edge, label in edges_labels:
                self.active_data.put((edge, label))
            self.current_n = data.n
            self.current_lp_soln = data.lp_soln
            self.current_adj_mat = data.adj_mat
            self.current_soln_adj_mat = data.soln_adj_mat
            self.current_weight_mat = data.weight_mat

        edges = [0] * batch_size
        lp_soln_a = np.empty((batch_size, int(self.current_n), int(self.current_n)))
        adj_mat_a = np.empty((batch_size, int(self.current_n), int(self.current_n)))
        soln_adj_mat_a = np.empty((batch_size, int(self.current_n), int(self.current_n)))
        weight_mat_a = np.empty((batch_size, int(self.current_n), int(self.current_n)))
        labels = [0] * batch_size

        for i in range(batch_size):
            edge_label = self.active_data.get()

            edges[i] = edge_label[0]
            lp_soln_a[i] = np.array(self.current_lp_soln)
            adj_mat_a[i] = np.array(self.current_adj_mat)
            soln_adj_mat_a[i] = np.array(self.current_soln_adj_mat)
            weight_mat_a[i] = np.array(self.current_weight_mat)
            labels[i] = edge_label[1]

        return (np.array(edges), lp_soln_a.astype(int), adj_mat_a.astype(int),
                soln_adj_mat_a.astype(int), weight_mat_a.astype(int), np.array(labels))


dataprocessor = DataProcessor()

for i in range(1000):
    print(dataprocessor.get_data(4))

print(dataprocessor.epoch)