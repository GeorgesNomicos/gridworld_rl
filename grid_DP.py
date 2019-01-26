#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 19:21:56 2019

@author: georgesnomicos
"""
import numpy as np
import time

class grid(object):
    
    def __init__(self, size_grid, gamma):
        self.size_grid = size_grid
        self.action = 4
        self.gamma = gamma
        self.proba = [1, 0, 0, 0] #Desired, Opposite, L (from D), R
        self.terminate = 1
        
    def states_(self):
        s = 0 
        states = np.zeros((self.size_grid, self.size_grid)) 
        for r in range(0, self.size_grid):
            for c in range(0, self.size_grid): 
                s += 1
                states[r, c] = int(s)
        return states
        

    def update(self, state, new_value, out):
        index = np.argwhere(self.states_() == state)
        r = index[0,0]
        c = index[0,1]
        out[r,c] = new_value
        
        return out
        
    def get_rc(self, state):
        index = np.argwhere(self.states_() == state)
        r = index[0,0]
        c = index[0,1]
        return r, c
        
    def policy(self, mode, state):
    #argument is a string either stochastic or deterministic
        if mode == "deterministic":
            policy = np.random.randint(1,4,16) #UP DOWN LEFT RIGHT
            return policy
        elif mode == "stochastic":
            policy = np.array([.55, .15, .15, .15]) #Towards goal 
            return policy
        else:
            print("Check argument")
    
    def t_proba(self):
        p = np.zeros((self.size_grid ** 2, self.size_grid ** 2, self.action))
        rc = np.zeros((4,2))
        states = self.states_()
        for state_from in range(0, self.size_grid ** 2):
          r, c = self.get_rc(state_from + 1)                
          rc = np.array([[r - 1,c],  #U
                         [r + 1, c], #D
                         [r, c - 1], #L
                         [r, c + 1]  #R
                         ])
            
          for a in range(0, self.action): #(1,2,3,4) -> (U,R,L,D)
            if a != 0:
                rc = np.roll(rc, 1, axis = 0)
                    
            for directions in range(0,4):
#                    if a == 1:
#                        proba = np.array([0.7, 0.2, 0.05, 0.05])
#                    elif a == 2:
#                        proba = np.array([0.7, 0.05, 0.05, 0.2])
#                    elif a == 3:
#                        proba = np.array([0.7, 0.2, 0.05, 0.05])
#                    else:
#                       proba = np.array([0.7, 0.05, 0.05, 0.2])
                proba = self.proba    
                r, c = rc[directions]    
                
                if r > (self.size_grid - 1) or r < 0 or\
                c > (self.size_grid - 1) or c < 0 : 
                    state_to = state_from
                else:
                    state_to = states[r, c]
                    state_to -= 1
                    
                
                p[int(state_from), int(state_to), int(a)] +=\
                proba[directions]
                    
        return p
                    
    def t_proba_get(self, state, s_next,a):
        p = self.t_proba() #Necessary???
        return p[state,s_next,a]
    
    def reward_(self,state, s_next, a):
        if s_next == self.terminate:
            reward = 10
        #elif s_next == 3:
        #    reward = -10
        else:
            reward = -1
        
        return reward
            
    def policy_eval(self, policy):
        V = np.zeros((size_grid,size_grid,2))
        p = self.t_proba() 
        delta = 2
        while delta > 0.02:            
            for state in range(0,self.size_grid ** 2):
                v_s = 0
                if state == self.terminate - 1:
                    continue
                for a in range(0,self.action):
                    for s_next in range(0,self.size_grid ** 2):
                        r, c = self.get_rc(s_next+1)
                        v_s += self.policy(policy)[a] *\
                        p[state, s_next, a] *\
                        (self.reward_(state, s_next, a) + self.gamma * V[r,c,1])
                V[:,:,1] = self.update(state + 1, v_s, V[:,:,1]) #Update V 
                #after one state backup
        
#        policy = np.random.randint(1,4,16)
#        for state in range(0,self.size_grid ** 2):
#            b = policy
#            policy = np.argmax(V)
#            if b != policy:
#                print("Policy stable")
#                break
        
        
            print(V[:,:,1])
            delta = np.amax(np.absolute(V[:,:,0]-V[:,:,1]), axis=None)
            V[:,:,0] = V[:,:,1]
            print(delta)    
        
        return V
          
    
start_time = time.time()    
size_grid = 4
gamma = 0.9
grid1 = grid(size_grid, gamma)
out = np.zeros((5,5))

V = grid1.policy_eval("deterministic")
V = np.round(V, decimals = 1, out = None)
print("Value function: ")
print(V[:,:,0])

print("--- %s seconds ---" % (time.time() - start_time))

#trans_p = grid1.t_proba()
#print(trans_p[:,:,3]) #(U,R,L,D)
print(grid1.states_())