# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 16:37:22 2017

@author: gliese581gg
"""
import numpy as np
import RL_jy

class test_mdp:
    def __init__(self):
        self.reset()
    def transition(self,act):
        prev_state = self.state
        action = act
        reward = 0.
        terminate = 0.
        
        if self.state < 3:
            if act == 0 : self.state = self.state*2+1
            else : self.state = self.state*2+2
        if self.state == 3 : reward = 0. ; terminate = 1.
        if self.state == 4 : reward = -1. ; terminate = 1.
        if self.state == 5 : reward = 1. ; terminate = 1.
        if self.state == 6 : reward = 0. ; terminate = 1.
        
        next_state = self.state
        
        if self.state > 2 : self.reset()
        
        return prev_state,action,reward,terminate,next_state
    def reset(self):
        self.state = 0
        
mdp = test_mdp()
rl = RL_jy.RL_learner(7,2,algorithm='SARSA',learn_type='MC')

acc_r = 0.

ss = []
aa = []
rr = []
tt = []
nss = []
naa = []

for i in range(50000):
    eps = max(0.,1. - i/5000.)
    if np.random.random() > eps : a = rl.act(mdp.state)
    else : a = np.random.randint(0,2)

    if i > 0 and rl.learn_type=='TD' : rl.learn(s,pa,r,t,ns,a)
    
    if i > 0 and rl.learn_type=='MC' : 
        ss.append(s);aa.append(pa);rr.append(r);tt.append(t);nss.append(ns);naa.append(a)
        if t == 1 :
            sss = np.array(ss);aaa=np.array(aa);rrr=np.array(rr);ttt=np.array(tt);nsss=np.array(nss);naaa=np.array(naa)
            del ss[:];del aa[:];del rr[:];del tt[:];del nss[:];del naa[:]
            rrrr = rl.cal_epi_return(rrr)
            for ii in range(sss.shape[0]):
                rl.learn(sss[ii],aaa[ii],rrrr[ii],ttt[ii],nsss[ii],naaa[ii])
                
    
    s,pa,r,t,ns = mdp.transition(a)
    acc_r += r
    


    if i % 1000 == 0 : 
        print 'iter ' + str(i) + ' , ' + str(eps)
        print rl.table
        print acc_r
        acc_r = 0.