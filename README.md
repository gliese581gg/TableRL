## TableRL
Simple python implementation of classic SARSA and Q algorithm (with value table)
It can learn with TD or MC

usage : 

import RL_jy

rl = RL_jy.RL_learner(num_states,num_actions,learning_rate,discount,algorithm='SARSA',learn_type='MC')  

#where algorithm : 'Q' or 'SARSA', learn_type:'TD' or 'MC'

action = rl.act(state)

rl.learn(state,action,reward,terminate,next_state,next_action)
