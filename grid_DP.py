#!/usr/bin/env python3i
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
        self.proba = [0.25, 0.25, 0.25, 0.25] #Desired, Opposite, L (from D), R
        self.terminate = 1
        
    def states_(self):
        s = 0 
        states = np.zeros((self.size_grid, self.size_grid)) 
        for r in range(self.size_grid):
            for c in range(self.size_grid): 
                s += 1
                states[r, c] = int(s)
        return states
        

    def update(self, state, new_value, out):
        r, c = self.get_idx_from_state(state)
        out[r,c] = new_value
        return out
        
    def get_idx_from_state(self, state):
        index = np.argwhere(self.states_() == state)
        r = index[0,0]
        c = index[0,1]
        return r, c
        
    def policy(self, state):
        policy = np.array([.55, .15, .15, .15]) #Towards goal 
        return policy
    
    def transition_proba_matrix(self):
        transition_proba = np.zeros((self.size_grid ** 2, self.size_grid ** 2, self.action))
        rc_transform = np.zeros((4,2))
        states = self.states_()
        for state_from in range(0, self.size_grid ** 2):
          r, c = self.get_idx_from_state(state_from + 1)                
          rc_transform = np.array([[r - 1,c],  #U
                         [r + 1, c], #D
                         [r, c - 1], #L
                         [r, c + 1]  #R
                         ])
            
          for a in range(self.action): #(1,2,3,4) -> (U,R,L,D)
            if a != 0:
                rc_transform = np.roll(rc_transform, 1, axis = 0)
                    
            for directions in range(0,4):
                proba = self.proba    
                r, c = rc_transform[directions]    
                
                if r > (self.size_grid - 1) or r < 0 or\
                c > (self.size_grid - 1) or c < 0 : 
                    state_to = state_from
                else:
                    state_to = states[r, c]
                    state_to -= 1
                    
                
                transition_proba[int(state_from), int(state_to), int(a)] +=\
                proba[directions]
        self.transition_proba = transition_proba
        return transition_proba
                    
    def get_transition_proba(self, state, s_next,a):
        return self.transition_proba[state, s_next, a]
    
    def get_reward(self, state, s_next, a):
        if s_next == self.terminate:
            reward = 1
        else:
            reward = -1
        return reward
            
    def policy_evaluation(self):
        V = np.zeros((size_grid,size_grid,2))
        p = self.transition_proba_matrix() 
        delta = 2
        while delta > 0.0001:            
            for state in range(0,self.size_grid ** 2):
                v_s = 0
                if state == self.terminate - 1:
                    continue
                for a in range(0,self.action):
                    for s_next in range(0,self.size_grid ** 2):
                        r, c = self.get_idx_from_state(s_next+1)
                        if not (p[state, s_next, a] < 0.0001):
                            v_s += self.policy(state)[a] * p[state, s_next, a] * (self.get_reward(state, s_next, a) + self.gamma * V[r,c,1])
                V[:,:,0] = self.update(state + 1, v_s, V[:,:,0]) #Update V 
                #after one state backup
        
            policy = np.random.randint(1,4,16)
            for state in range(0,self.size_grid ** 2):
                b = policy
                policy = np.argmax(V)
                if np.sum(b-policy)<0.1:
                    print("Policy stable")
                    break

            delta = np.amax(np.absolute(V[:,:,0]-V[:,:,1]), axis=None)
            V[:,:,1] = V[:,:,0]
        
        return V, policy
          
start_time = time.time()    
size_grid = 4
gamma = 0.9
grid1 = grid(size_grid, gamma)

V = grid1.policy_evaluation()
V = np.round(V, decimals = 2, out = None)
print("Value function: ")
print(V[:,:,0])

print("--- %s seconds ---" % (time.time() - start_time))

#trans_p = grid1.t_proba()
#print(trans_p[:,:,3]) #(U,R,L,D)
