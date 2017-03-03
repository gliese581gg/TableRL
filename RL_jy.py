# -*- coding: utf-8 -*-
import numpy as np

class RL_learner:
    def __init__(self,dim_s,dim_a,lr=0.1,discount=0.9,algorithm='Q',learn_type='TD'):
        '''
        make Q value table and set the algorithm (Q learning or SARSA)
        inputs :
            dim_s : dimension of states (int)
            dim_a : dimension of actions (int)
            lr : learning rate (float)
            discount : discount (float)
            algorithm : 'Q' or 'SARSA' (string)
            learn_type : 'TD' or 'MC' (string)
        '''
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.lr = lr
        self.discount = discount
        if algorithm != 'Q' and algorithm != 'SARSA' : raise ValueError('invalid algorithm')
        self.algorithm = algorithm
        if learn_type != 'TD' and learn_type != 'MC' : raise ValueError('invalid learn_type')
        self.learn_type = learn_type
        self.table = np.random.normal(0.,0.001,(self.dim_s,self.dim_a))
    def act(self,state):
        '''
        Select action from current policy
        inputs :
            state : index of state (int)
        output : index of action (int)
        '''
        return np.argmax(self.table[state,:])
    def learn(self,s,a,r,t,ns,na=-1):
        '''
        Update Q value table from transition data
        inputs : 
            s : index of state (int)
            a : index of action (int)
            r : value of reward (in TD case) or return (in MC case) (float)
            t : termination or not (int)
            ns : index of next state (int)            
        output :
            No output.
        '''
        if self.algorithm == 'SARSA' and na < 0 : raise ValueError('invalid na')
        
        if self.learn_type == 'TD' : 
            if self.algorithm == 'Q' :
                td = r+self.discount*(1.-t)*np.max(self.table[ns,:])-self.table[s,a]
            elif self.algorithm == 'SARSA' :
                td = r+self.discount*(1.-t)*self.table[ns,na]-self.table[s,a]
                
        elif self.learn_type == 'MC':
            td = r-self.table[s,a]
            
        self.table[s,a] += self.lr*td
                  
    def cal_epi_return(self,r):
        '''
        calculate episode return (for MC learning case)
        inputs :
            r : numpy array of rewards for one episode (timestep 0~n, shape : [n])
        outputs : episode return (numpy array of shape [n])
        '''
        ret = r.copy()
        for i in range(ret.shape[0]-2,-1,-1): ret[i]+=self.discount*ret[i+1]
        return ret
    




Q = RL_learner(10,4,0.1,0.9,'Q','TD')
print Q.dim_s