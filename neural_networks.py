import random
import torch
import numpy as np
from torch.autograd import Variable

class graph_embedding_Net(torch.nn.Module):
    #p: the length of vertice representation
    #n: the number of vertex
    #e_num: the number of all edges
    def __init__(self, p):
        self.p = p
        #self.n = n
        super(graph_embedding_Net, self).__init__()
        #theta1: 
        self._theta12 = torch.nn.Linear(1, p)
        self._theta11 = torch.nn.Linear(p, p)
        #theta2: compute \theta_2 * \mu_v
        self._theta2 = torch.nn.Linear(p, p)
        #theta4: compute \theta_4 * w(u, v)
        self._theta4 = torch.nn.Linear(1, p)
        #theta3:
        self._theta3 = torch.nn.Linear(p, p)
        #theta6
        self._theta6 = torch.nn.Linear(p, p)
        #theta7
        self._theta7 = torch.nn.Linear(p, p)
        #theta5
        self._theta5 = torch.nn.Linear(2*p, 2)

    #E : the n*n matrix of fractions for all edges
    #Mu: n*p matrix, the representation of nodes
    #W:  the n*n weights matrix 
    #adj_F: adjacent Matrix for fraction solution, adj_M[a, b] is 0 or 1 indicating whether a and b are adjacent
    #adj_G: adjacent Matrix for Graph embedding.
    #e: a tuple of two index (u, v)
    def forward(self, E, Mu, W, adj_F,adj_G, T, e, n, bt_size) :
        p = self.p
        #n = self.n
        if check_cuda():
            Mu = Variable(torch.randn(bt_size, n, p).cuda())
        else:
            Mu = Variable(torch.randn(bt_size, n, p))            
        for i in range(T):
            Mu = self.iteration_mu(E, Mu, W, adj_F, adj_G, n, bt_size)
        #Mu representation done!
        #u, v = e  #how to unpack the variable
        if check_cuda():
            Mu_tmp = Variable(torch.randn(bt_size, p).cuda())
        else:
            Mu_tmp = Variable(torch.randn(bt_size, p))  
        for i in range(bt_size):
            Mu_tmp[i] = ( Mu[i][e[i][0]] + Mu[i][e[i][1]] )/2
        theta7 = self._theta7( Mu_tmp.view(bt_size, 1, -1))       
        theta6 = self._theta6( (Mu.sum(1) / n ) .view(bt_size, 1, -1))       
        theta67 = torch.cat((theta6, theta7), 2).clamp( min = 0)    
        theta5 = self._theta5(theta67.clone().view(bt_size, 1, -1)) 
        #classifier = torch.nn.functional.softmax(theta5, dim=1)
        classifier = theta5
        return classifier

    def iteration_mu(self, E, Mu, W, adj_F, adj_G, n, bt_size):
        #n = self.n
        p = self.p
        E, Mu, W, adj_F, adj_G= self.init(E, Mu, W, adj_F, adj_G)
        theta1 = self.theta1(adj_F, E, n, bt_size)
        theta2 = self.theta2(adj_G, Mu, n, bt_size)
        theta34 = self.theta34(adj_G, W, n, bt_size)
        Mu = self.theta_relu(theta1, theta2, theta34)
        return Mu

    #theta1 process:  = theta_11(\Sigma_{v \in N(u)} ReLu( theta_12 x_e(u, v)))
    #E: n*n
    #adj_F: n*n
    def theta1(self, adj_F, E, n, bt_size):
        #n = self.n
        p = self.p
        if check_cuda():
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p).cuda())
        else:
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p)) 

        theta12 = self._theta12(E.view(bt_size, n , n, 1)).clamp( min = 0 ) 

        if check_cuda():
            theta11 = Variable(torch.zeros(bt_size, n, self.p).cuda() )
        else:
            theta11 = Variable(torch.zeros(bt_size,  n, self.p))            
        for i in range(n):
            theta11[:, i, :] = (theta12[:, :, i, :].clone() * ( (adj_F[:, i, :].contiguous().view(bt_size, -1,1)).expand_as(theta12[:, :, i, :].clone())) ) .sum(1) / n

        theta11 = self._theta11( theta11.clone() )

        return theta11 
    #theta2 process: theta2 (\Sigma_{v \in N(u)} \mu_v)
    def theta2(self, adj_G, mu, n, bt_size):
        p = self.p
        theta2 = mu

        theta2 = self._theta2( theta2.clone() )

        for i in range( n ):
            mu[:, i, :] = (theta2.clone() * (adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta2)) .sum(1) / n #mu[i, :] = (theta2.clone() * (adj_G[i, :].view(-1, 1)).expand_as(theta2)) .sum(0) / n 
        return mu
    #theta34 process: the same as theta1 process
    def theta34(self, adj_G, W, n, bt_size):
        #n = self.n
        p = self.p
        if check_cuda():
            theta4 = Variable(torch.zeros( bt_size, n , n , self.p).cuda())
        else:
            theta4 = Variable(torch.zeros( bt_size, n , n , self.p)) 
        
        theta4 = self._theta4( W.view( bt_size, n, n, 1 ) ).clamp( min = 0) 

        if check_cuda():
            theta3 = Variable(torch.zeros( bt_size, n , self.p).cuda())
        else:
            theta3 = Variable(torch.zeros( bt_size, n , self.p))           
        for i in range(n):
            theta3[:, i, :] = (theta4[:, :, i, :].clone() * (adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta4[:, :, i, :].clone())) .sum(1) / n
        theta3 = self._theta3( theta3.clone() )
        
        return theta3 
    #sum up process: ReLU(theta1 + theta2 + theta3)
    def theta_relu(self, theta1, theta2, theta34):
        return ((theta1 + theta2 + theta34) / 3).clamp(min = 0)
    #change ary from numpy array to Variable
    def get_Var(self,ary):
        if type(ary) is not Variable:
            if check_cuda():
                return Variable(torch.Tensor(ary).cuda())
            else:
                return Variable(torch.Tensor(ary))                
        return ary
    #Get all paras' Variable representation
    def init(self, E, Mu, W, adj_F, adj_G,):
        return self.get_Var(E), \
                 self.get_Var(Mu),\
                 self.get_Var(W), \
                 self.get_Var(adj_F), \
                 self.get_Var(adj_G)

class test_Net(torch.nn.Module):
    #p: the length of vertice representation
    #n: the number of vertex
    #e_num: the number of all edges
    def __init__(self, p):
        self.p = p
        #self.n = n
        super(test_Net, self).__init__()
        #theta1: 
        self._theta12 = torch.nn.Linear(1, p)
        self._theta11 = torch.nn.Linear(p, p)
        #theta2: compute \theta_2 * \mu_v
        self._theta2 = torch.nn.Linear(p, p)
        #theta4: compute \theta_4 * w(u, v)
        self._theta4 = torch.nn.Linear(1, p)
        #theta3:
        self._theta3 = torch.nn.Linear(p, p)
        #theta6
        self._theta6 = torch.nn.Linear(p, p)
        #theta7
        self._theta7 = torch.nn.Linear(p, p)
        #theta5
        self._theta5 = torch.nn.Linear(2*p, 2)

    #E : the n*n matrix of fractions for all edges
    #Mu: n*p matrix, the representation of nodes
    #W:  the n*n weights matrix 
    #adj_F: adjacent Matrix for fraction solution, adj_M[a, b] is 0 or 1 indicating whether a and b are adjacent
    #adj_G: adjacent Matrix for Graph embedding.
    #e: a tuple of two index (u, v)
    def forward(self, E, Mu, W, adj_F,adj_G, T, e, n, bt_size) :
        p = self.p
        #n = self.n
        #For all theta before 5, set bt_size = 1, for theta after5, set bt_size = bt_size
        if check_cuda():
            Mu = Variable(torch.randn(1, n, p).cuda())
        else:
            Mu = Variable(torch.randn(1, n, p))            
        for i in range(T):
            Mu = self.iteration_mu(E, Mu, W, adj_F, adj_G, n, 1)
        #Mu representation done!
        #u, v = e  #how to unpack the variable
        if check_cuda():
            Mu_tmp = Variable(torch.randn(bt_size, p).cuda())
        else:
            Mu_tmp = Variable(torch.randn(bt_size, p)) 
        #Mu only have one instance in the first dimension
        for i in range(bt_size):
            Mu_tmp[i] = ( Mu[0][e[i][0]] + Mu[0][e[i][1]] )/2
        theta7 = self._theta7( Mu_tmp.view(bt_size, 1, -1))
        print(Mu)
        print(n)
        print(bt_size)
        theta6 = self._theta6( (Mu.sum(1) / n ) .view(bt_size, 1, -1))       
        theta67 = torch.cat((theta6, theta7), 2).clamp( min = 0)    
        theta5 = self._theta5(theta67.clone().view(bt_size, 1, -1)) 
        #classifier = torch.nn.functional.softmax(theta5, dim=1)
        classifier = theta5
        return classifier

    def iteration_mu(self, E, Mu, W, adj_F, adj_G, n, bt_size):
        #n = self.n
        p = self.p
        E, Mu, W, adj_F, adj_G= self.init(E, Mu, W, adj_F, adj_G)
        theta1 = self.theta1(adj_F, E, n, bt_size)
        theta2 = self.theta2(adj_G, Mu, n, bt_size)
        theta34 = self.theta34(adj_G, W, n, bt_size)
        Mu = self.theta_relu(theta1, theta2, theta34)
        return Mu

    #theta1 process:  = theta_11(\Sigma_{v \in N(u)} ReLu( theta_12 x_e(u, v)))
    #E: n*n
    #adj_F: n*n
    def theta1(self, adj_F, E, n, bt_size):
        #n = self.n
        p = self.p
        if check_cuda():
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p).cuda())
        else:
            theta12 = Variable(torch.zeros(bt_size, n, n, self.p)) 

        theta12 = self._theta12(E.view(bt_size, n , n, 1)).clamp( min = 0 ) 

        if check_cuda():
            theta11 = Variable(torch.zeros(bt_size, n, self.p).cuda() )
        else:
            theta11 = Variable(torch.zeros(bt_size,  n, self.p))            
        for i in range(n):
            theta11[:, i, :] = (theta12[:, :, i, :].clone() * ( (adj_F[:, i, :].contiguous().view(bt_size, -1,1)).expand_as(theta12[:, :, i, :].clone())) ) .sum(1) / n

        theta11 = self._theta11( theta11.clone() )

        return theta11 
    #theta2 process: theta2 (\Sigma_{v \in N(u)} \mu_v)
    def theta2(self, adj_G, mu, n, bt_size):
        p = self.p
        theta2 = mu

        theta2 = self._theta2( theta2.clone() )

        for i in range( n ):
            mu[:, i, :] = (theta2.clone() * (adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta2)) .sum(1) / n #mu[i, :] = (theta2.clone() * (adj_G[i, :].view(-1, 1)).expand_as(theta2)) .sum(0) / n 
        return mu
    #theta34 process: the same as theta1 process
    def theta34(self, adj_G, W, n, bt_size):
        #n = self.n
        p = self.p
        if check_cuda():
            theta4 = Variable(torch.zeros( bt_size, n , n , self.p).cuda())
        else:
            theta4 = Variable(torch.zeros( bt_size, n , n , self.p)) 
        
        theta4 = self._theta4( W.view( bt_size, n, n, 1 ) ).clamp( min = 0) 

        if check_cuda():
            theta3 = Variable(torch.zeros( bt_size, n , self.p).cuda())
        else:
            theta3 = Variable(torch.zeros( bt_size, n , self.p))           
        for i in range(n):
            theta3[:, i, :] = (theta4[:, :, i, :].clone() * (adj_G[:, i, :].contiguous().view(bt_size, -1, 1)).expand_as(theta4[:, :, i, :].clone())) .sum(1) / n
        theta3 = self._theta3( theta3.clone() )
        
        return theta3 
    #sum up process: ReLU(theta1 + theta2 + theta3)
    def theta_relu(self, theta1, theta2, theta34):
        return ((theta1 + theta2 + theta34) / 3).clamp(min = 0)
    #change ary from numpy array to Variable
    def get_Var(self,ary):
        if type(ary) is not Variable:
            if check_cuda():
                return Variable(torch.Tensor(ary).cuda())
            else:
                return Variable(torch.Tensor(ary))                
        return ary
    #Get all paras' Variable representation
    def init(self, E, Mu, W, adj_F, adj_G,):
        return self.get_Var(E), \
                 self.get_Var(Mu),\
                 self.get_Var(W), \
                 self.get_Var(adj_F), \
                 self.get_Var(adj_G)

def check_cuda():
    #return False
    return torch.cuda.is_available()