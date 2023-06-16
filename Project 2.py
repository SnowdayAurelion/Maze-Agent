# -*- coding: utf-8 -*-
"""
Created on Mon May  1 09:01:41 2023

@author: light
"""

import pandas as pd
import numpy as np
import copy
import sys
import random

#Example data
initial_state = [
    [2, 5, 5, 5, 0, 0, 0, 5, 5, 5],
    [5, 1, 1, 1, 0, 0, 0, 1, 1, 5],
    [5, 0, 0, 1, 0, 1, 0, 1, 1, 5],
    [5, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [5, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    [5, 1, 0, 1, 1, 1, 1, 0, 0, 5],
    [5, 1, 0, 1, 1, 1, 1, 1, 1, 5],
    [5, 1, 0, 1, 1, 1, 1, 1, 1, 5],
    [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
]

# initial_state=np.array(initial_state)

# initial_state=np.array(initial_state)     #leads to an error

class Agent:
    def __init__(self,name,initial_state,percept):
        #Send initial state to Environment class
        self.name=name
        self.data=pd.DataFrame(columns=["Initial state","Belief state","Percept","Goal"])
        
        self.data["Initial state"]=initial_state
        self.data["Belief state"]=initial_state
        self.data["Percept"]=percept
        self.data["Goal"]=None
        
        percept=self.scan()
        self.update(percept)
    def scan(self):     #scans N, E, W, and S. But, we already have the belief state with the info
        belief_state=self.data["Belief state"]  
        #scans N, E, W, and S
        percept="0101"  #same percept for each "sub"-agent
        
        return percept
    
    def predict(self):
        return 
    
    def update(self,percept):   #percept given by itself or class Environment
        #code that updates belief state from the percept
        pass
    
    def move(self,action):
        belief_state=self.data["Belief state"]  #grab data when needed using DataFrame
        Environment.update(action)
        
    def cost(self):
        #each step is one point
        pass
    def goal_test(self):
        pass
    def transition_model(self):     #What's the difference between update and transition model? I think we use transition model to verify update. Removes postion that does not make sense.
        pass
    
class Environment:
    def __init__(self,initial_state=None,agent_name="Pathfinder"):
        self.generate()
        sys.exit()
        #Generate initial state if not given
        if initial_state==None:
            initial_state=self.generate()
        else:
            #If given, then check if it's valid
            if self.valid_map(initial_state):
                pass
            else:
                raise(Exception("Invalid map"))
        
        #Find positon of mini agent (there will be one)
        position=self.find_positions(initial_state,True)
        
        #Make sure agent does not know it's position
        initial_state_clueless=copy.deepcopy(initial_state)
        for i in range(len(initial_state_clueless)):
            for j in range(len(initial_state_clueless)):
                if initial_state_clueless[i][j]==2:
                    initial_state_clueless[i][j]=0
        
        print(initial_state)
        #Get percept for Agent
        percept=self.percept(initial_state,position)
        
        #Give percept to Agent
        self.Agent=Agent(agent_name,initial_state_clueless,percept)
        print(self.Agent.data)
        # print(self.Agent.data)
        #Save data in DataFrame
        self.data=pd.DataFrame(columns=["Initial state","State","Percept"])
        self.data["Initial state"]=initial_state
        self.data["State"]=initial_state
        self.data["Percept"]=percept
        # print(self.data)
        
    def generate(self):
        #Create 16x16 state with just walls
        state=[[0 for i in range(16)] for i in range(16)]
        piece0=[[0,0],
                [0,0]]
        piece1=[[1,0],
                [0,0]]
        piece2=[[0,1],
                [0,0]]        
        piece3=[[0,0],
                [1,0]]
        piece4=[[0,0],
                [0,1]]
        piece5=[[0,1],
                [0,1]]
        piece6=[[0,0],
                [1,1]]
        piece7=[[1,1],
                [1,1]]
        
        rulebook={0:[0,1,2,3,4,5,6,7],1:[1,2,3,4,5,6],2:[1,2,4,5],3:[1,3,4,6],4:[2,3,4,5,6],5:[],6:[],7:[]}
        def check(piece_A,piece_B):
            pieces=list(np.intersect1d(rulebook[piece_A],rulebook[piece_B]))
            if pieces:
                return random.choice(pieces)
            else:
                return 0
        # print(check(1,2)) #should be 1 and 3
        
        piece_state=[[random.randint(1, 7) for i in range(8)] for i in range(8)]
        # piece_state=[[7, 1, 7, 7, 7, 7, 7, 7], [2, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7], [7, 7, 7, 7, 7, 7, 7, 7]] 
        
        print(piece_state,"\n\n\n\n")
        
        # Checks validity of pieces using rulebook
        for i in range(len(piece_state)-1):
            for j in range(1,len(piece_state[0])):
                piece_Q=check(piece_state[i][j],piece_state[i+1][j-1])
                piece_state[i+1][j]=piece_Q
        print(piece_state)
        
        pieces={0:piece0,1:piece1,2:piece2,3:piece3,4:piece4,5:piece5,6:piece6,7:piece7}
        for i in range(len(piece_state)-2):
            # Off set i
            if i!=0:
                i+=2
            for j in range(len(piece_state[0])-2):
                # Get the piece
                piece=pieces[piece_state[i][j]]
                print(piece_state[i][j])
                
                # Off set j
                if i!=0:
                    j+=2
                
                # Replace values in state with values of the piece
                state[i][j]=piece[0][0]
                state[i+1][j]=piece[1][0]
                state[i][j+1]=piece[0][1]
                state[i+1][j+1]=piece[1][1]
                 
        print(state)
        sys.exit()
        pass
    def percept(self,state,position=None):  #percept to give to agent #Might use just to initialize the agent
        #If position=None, find first mini agent
        if position==None:
            position=self.find_positions(state,first=True)
        
        #Initialize percept
        percept=""
        
        #Calculate N, E, W, and S
        print(position)
        #N
        try:
            if position[0]!=0:
                percept+=str(state[position[0]-1][position[1]])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #E
        try:
            if position[1]!=len(state[0]):
                percept+=str(state[position[0]][position[1]+1])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #W
        try:
            if position[1]!=0:
                percept+=str(state[position[0]][position[1]-1])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #S
        try:
            if position[0]!=len(state):
                percept+=str(state[position[0]+1][position[1]])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        
        print(percept)
        
        return percept
    
    def update(self,action):
        pass
    def valid_map(self,state):
        return True
    
    def find_positions(self,state,first=False):
        positions=[]
        
        #Cycle through matrix
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j]==2:
                    positions.append([[i,j]])
                #If first is True, find the first instantce of mini agent
                if state[i][j]==2 and first==True:
                    return [i,j]
            
        return positions

environment=Environment(initial_state)

