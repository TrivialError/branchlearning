import os
import _pickle
import gzip
import random
import queue
import numpy as np


class DataProcessor:
    def __init__(self, batch_size):
        self.files = os.listdir("./DataTest")
        self.data = []
        self.epoch = 0
        self.batch_size = batch_size
        self.num_batches = 0
        self.count0 = 0
        self.count1 = 0

        for file in self.files:
            with gzip.open('./DataTest/' + file) as f:
                self.data.append(_pickle.load(f))

        self.current_epoch_data = queue.Queue()
        self.active_data = queue.Queue()
        self.current_n = 0
        self.current_lp_soln = None
        self.current_adj_mat = None
        self.current_soln_adj_mat = None
        self.current_weight_mat = None

        for d in self.data:
            self.current_epoch_data.put(d)
            self.num_batches += len(d.var_sb_label_dict) // self.batch_size

        print("num_batches: ", self.num_batches)

    def get_data(self):

        while self.active_data is None or self.active_data.qsize() < self.batch_size:
            self.active_data = queue.Queue()
            if self.current_epoch_data.empty():
                self.epoch += 1
                random.shuffle(self.data)
                for d in self.data:
                    self.current_epoch_data.put(d)

            data = self.current_epoch_data.get()
            edges_labels = list(data.var_sb_label_dict.items())
            random.shuffle(edges_labels)
            for edge, label in edges_labels:
                self.active_data.put((edge, label))
            self.current_n = data.n
            self.current_lp_soln = data.lp_soln
            self.current_adj_mat = data.adj_mat
            self.current_soln_adj_mat = data.soln_adj_mat
            self.current_weight_mat = data.weight_mat

        edges = [0] * self.batch_size
        N = [0] * self.batch_size
        lp_soln_a = np.empty((self.batch_size, int(self.current_n), int(self.current_n)))
        adj_mat_a = np.empty((self.batch_size, int(self.current_n), int(self.current_n)))
        soln_adj_mat_a = np.empty((self.batch_size, int(self.current_n), int(self.current_n)))
        weight_mat_a = np.empty((self.batch_size, int(self.current_n), int(self.current_n)))
        labels = [0] * self.batch_size

        for i in range(self.batch_size):
            edge_label = self.active_data.get()

            edges[i] = edge_label[0]
            N[i] = self.current_n
            lp_soln_a[i] = np.array(self.current_lp_soln)
            adj_mat_a[i] = np.array(self.current_adj_mat)
            soln_adj_mat_a[i] = np.array(self.current_soln_adj_mat)
            weight_mat_a[i] = np.array(self.current_weight_mat)
            labels[i] = edge_label[1]
            if edge_label[1] == 0:
                self.count0 += 1
            else:
                self.count1 += 1

        #print("zeros: ", self.count0)
        #print("ones: ", self.count1)
        print("labels: ", labels)

        return (np.array(edges), lp_soln_a.astype(int), adj_mat_a.astype(int),
                soln_adj_mat_a.astype(int), weight_mat_a.astype(int), np.array(labels), N)

