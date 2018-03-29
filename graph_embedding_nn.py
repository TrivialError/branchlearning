import random
import torch
import numpy as np
from torch.autograd import Variable


class graph_embedding_Net(torch.nn.Module):
    # p: the length of vertice representation
    # n: the number of vertex
    # e_num: the number of all edges
    def __init__(self, p):
        self.p = p
        # self.n = n
        super(graph_embedding_Net, self).__init__()
        # theta1:
        self._theta12 = torch.nn.Linear(1, p)
        self._theta11 = torch.nn.Linear(p, p)
        # theta2: compute \theta_2 * \mu_v
        self._theta2 = torch.nn.Linear(p, p)
        # theta4: compute \theta_4 * w(u, v)
        self._theta4 = torch.nn.Linear(1, p)
        # theta3:
        self._theta3 = torch.nn.Linear(p, p)
        # theta6
        self._theta6 = torch.nn.Linear(p, p)
        # theta7
        self._theta7 = torch.nn.Linear(p, p)
        # theta5
        self._theta5 = torch.nn.Linear(2 * p, 2)

    # E : the n*n matrix of fractions for all edges
    # Mu: n*p matrix, the representation of nodes
    # W:  the n*n weights matrix
    # adj_F: adjacent Matrix for fraction solution, adj_M[a, b] is 0 or 1 indicating whether a and b are adjacent
    # adj_G: adjacent Matrix for Graph embedding.
    # e: a tuple of two index (u, v)
    def forward(self, E, Mu_, W, adj_F, adj_G, T, e, n, bt_size):
        p = self.p
        # n = self.n
        if check_cuda():
            Mu = Variable(torch.randn(bt_size, n, p).cuda())
        else:
            Mu = Variable(torch.randn(bt_size, n, p))
        for i in range(T):
            Mu = self.iteration_mu(E, Mu, W, adj_F, adj_G, n, bt_size)
        # Mu representation done!
        # u, v = e  #how to unpack the variable
        if check_cuda():
            Mu_tmp = Variable(torch.randn(bt_size, p).cuda())
        else:
            Mu_tmp = Variable(torch.randn(bt_size, p))
        for i in range(bt_size):
            Mu_tmp[i] = (Mu[i][e[i][0]] + Mu[i][e[i][1]]) / 2
        theta7 = self._theta7(Mu_tmp.view(bt_size, 1, -1))
        theta6 = self._theta6((Mu.sum(1) / n).view(bt_size, 1, -1))
        theta67 = torch.cat((theta6, theta7), 2).clamp(min=0)
        theta5 = self._theta5(theta67.clone().view(bt_size, 1, -1))
        # classifier = torch.nn.functional.softmax(theta5, dim=1)
        classifier = theta5
        return classifier

    def iteration_mu(self, E, Mu, W, adj_F, adj_G, n, bt_size):
        # n = self.n
        p = self.p
        E, Mu, W, adj_F, adj_G = self.init(E, Mu, W, adj_F, adj_G)
        theta1 = self.theta1(adj_F, E, n, bt_size)
        theta2 = self.theta2(adj_G, Mu, n, bt_size)
        theta34 = self.theta34(adj_G, W, n, bt_size)
        Mu = self.theta_relu(theta1, theta2, theta34)
        return Mu

    # theta1 process:  = theta_11(\Sigma_{v \in N(u)} ReLu( theta_12 x_e(u, v)))
    # E: n*n
    # adj_F: n*n
    def theta1(self, adj_F, E, n, bt_size):
        # n = self.n
        p = self.p
        if check_cuda():
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p).cuda())
        else:
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p))

        theta12 = self._theta12(E.view(bt_size, n, n, 1)).clamp(min=0)

        if check_cuda():
            theta11 = Variable(torch.zeros(bt_size, n, self.p).cuda())
        else:
            theta11 = Variable(torch.zeros(bt_size, n, self.p))
        for i in range(n):
            theta11[:, i, :] = (theta12[:, :, i, :].clone() * (
                (adj_F[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta12[:, :, i, :].clone()))).sum(1) / n

        theta11 = self._theta11(theta11.clone())

        return theta11
        # theta2 process: theta2 (\Sigma_{v \in N(u)} \mu_v)

    def theta2(self, adj_G, mu, n, bt_size):
        p = self.p
        theta2 = mu

        theta2 = self._theta2(theta2.clone())

        for i in range(n):
            mu[:, i, :] = (theta2.clone() * (adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta2)).sum(
                1) / n  # mu[i, :] = (theta2.clone() * (adj_G[i, :].view(-1, 1)).expand_as(theta2)) .sum(0) / n
        return mu

    # theta34 process: the same as theta1 process
    def theta34(self, adj_G, W, n, bt_size):
        # n = self.n
        p = self.p
        if check_cuda():
            theta4 = Variable(torch.zeros(bt_size, n, n, self.p).cuda())
        else:
            theta4 = Variable(torch.zeros(bt_size, n, n, self.p))

        theta4 = self._theta4(W.view(bt_size, n, n, 1)).clamp(min=0)

        if check_cuda():
            theta3 = Variable(torch.zeros(bt_size, n, self.p).cuda())
        else:
            theta3 = Variable(torch.zeros(bt_size, n, self.p))
        for i in range(n):
            theta3[:, i, :] = (theta4[:, :, i, :].clone() * (
                adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta4[:, :, i, :].clone())).sum(1) / n
        theta3 = self._theta3(theta3.clone())

        return theta3
        # sum up process: ReLU(theta1 + theta2 + theta3)

    def theta_relu(self, theta1, theta2, theta34):
        return ((theta1 + theta2 + theta34) / 3).clamp(min=0)

    # change ary from numpy array to Variable
    def get_Var(self, ary):
        if type(ary) is not Variable:
            if check_cuda():
                return Variable(torch.from_numpy(ary).cuda())
            else:
                return Variable(torch.from_numpy(ary))
        return ary

    # Get all paras' Variable representation
    def init(self, E, Mu, W, adj_F, adj_G, ):
        return self.get_Var(E), \
               self.get_Var(Mu), \
               self.get_Var(W), \
               self.get_Var(adj_F), \
               self.get_Var(adj_G)


import numpy.random as rd


def get_data():
    N_v = np.ones(g_size).astype(int) * g_n
    return gen_data(g_size), \
           gen_label(g_size), \
           gen_E(g_size), \
           gen_Mu(g_size), \
           gen_W(g_size), \
           gen_F(g_size), \
           gen_G(g_size), \
           N_v


def gen_data(g_size):
    t_data = []
    for i in range(g_size):
        tmp = []
        for j in range(bt_size):
            tmp.append((rd.randint(0, g_n), rd.randint(0, g_n)))
        t_data.append(tmp)
    return np.asarray(t_data)


def gen_label(g_size):
    label = rd.random_sample(g_size * bt_size).reshape(g_size, bt_size)
    return np.where(label > 0.5, 1, 0)


def gen_E(g_size):
    return rd.random_sample(g_n * g_n * g_size * bt_size).reshape(g_size, bt_size, g_n, g_n)


def gen_Mu(g_size):
    return rd.random_sample(g_n * g_p * g_size * bt_size).reshape(g_size, bt_size, g_n, g_p)


def gen_W(g_size):
    return gen_E(g_size)


def gen_F(g_size):
    temp = rd.random_sample(g_size * g_n * g_n * bt_size)
    return np.where(temp > 0.5, 1, 0).reshape(g_size, bt_size, g_n, g_n)


def gen_G(g_size):
    temp = rd.random_sample(g_size * g_n * g_n * bt_size)
    return np.where(temp > 0.5, 1, 0).reshape(g_size, bt_size, g_n, g_n)


def train(net, Mu, eps, _lr, iteration_num, bt_size, dp):
    optimizer = optim.SGD(net.parameters(), lr=_lr)
    criterion = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    num = dp.num_batches
    for i in range(eps):
        for j in range(num):
            optimizer.zero_grad()
            # get batch data
            t_data, E, W, adj_G, adj_F, t_label, N = dp.get_data()
            output = net(E, [], W, adj_F, adj_G, iteration_num, t_data, N[0], bt_size).view(bt_size, -1)
            if check_cuda():
                target = Variable(torch.from_numpy(t_label).cuda()).view(bt_size).long()
            else:
                target = Variable(torch.from_numpy(np.array([t_label]))).view(bt_size).long()
            # print output
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            print(loss.data[0])
            if j % g_size == g_size - 1:
                print('[%d, %5d] loss: %.5f' % (i + 1, j + 1, running_loss / g_size))
                running_loss = 0.0
    print('Finished Training')


def check_cuda():
    return torch.cuda.is_available()


g_n = 30
g_p = 32
g_size = 5  # how many batches
iteration_num = 1
l_rate = 0.01
eps = 2000
bt_size = 8

import torch.optim as optim

# torch.cuda.set_device(0)
net = graph_embedding_Net(g_p)
if check_cuda():
    net.cuda()
print(net)
# t_data, t_label, E, Mu, W, adj_F, adj_G, N_v = get_data()
# train(net, t_data, t_label, E, Mu, W, adj_F, adj_G ,eps, l_rate, iteration_num, N_v, bt_size)
import DataProcessor as dp

_dp = dp.DataProcessor(bt_size)
train(net, [], eps, l_rate, iteration_num, bt_size, _dp)




