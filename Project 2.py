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
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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
        state=self.generate()   
        self.transition_model("S",state)
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
        # Create 16x16 state with just walls
        state=[[0 for i in range(16)] for i in range(16)]
        
        # Create functions that chooses the piece variation
        def choose_piece1():
            
            options = [[[0, 1, 0, 0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0, 1, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 1, 0],
                        [1, 0, 0, 1, 0, 1, 1, 1],
                        [1, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 0, 1, 1, 0, 0]],

                       [[0, 0, 0, 1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 1],
                        [0, 1, 1, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 0, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1]],

                       [[0, 1, 1, 1, 1, 1, 1, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 1, 0, 1, 0],
                        [0, 1, 1, 0, 1, 1, 1, 0]],

                       [[0, 0, 0, 0, 1, 1, 1, 0],
                        [1, 1, 1, 0, 1, 0, 0, 0],
                        [1, 1, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [0, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 1, 0, 1, 1, 1, 1]],

                       [[0, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 1, 0, 1],
                        [1, 1, 0, 1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 1, 0, 0, 0, 1, 0, 1],
                        [0, 1, 1, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 1, 1, 0, 0]]]
            
            return options[random.randint(0,len(options)-1)]
        
        def choose_piece2():
            
            options = [[[1, 0, 1, 1, 1, 1, 1, 1],
                       [1, 0, 0, 0, 0, 0, 1, 1],
                        [1, 0, 1, 1, 1, 0, 0, 1],
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 1, 0, 1],
                        [1, 0, 0, 0, 1, 1, 0, 1],
                        [0, 0, 1, 1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1]],

                       [[1, 1, 1, 0, 0, 0, 1, 1],
                        [1, 0, 1, 0, 1, 0, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 1, 0, 0, 1],
                        [1, 0, 1, 1, 0, 0, 1, 1],
                        [1, 0, 1, 1, 0, 1, 1, 1]],

                       [[1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1]],

                       [[0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 1, 1, 1, 0, 1, 0],
                        [1, 0, 1, 0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 0, 0, 1, 1, 0, 1],
                        [1, 0, 1, 1, 0, 0, 0, 0],
                        [0, 0, 1, 1, 0, 1, 1, 1],
                        [0, 1, 1, 1, 0, 1, 1, 1]],

                       [[1, 1, 1, 1, 1, 1, 0, 1],
                        [1, 1, 0, 0, 0, 0, 0, 1],
                        [1, 1, 0, 1, 0, 1, 1, 1],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 0],
                        [1, 1, 0, 1, 0, 0, 0, 0],
                        [1, 0, 0, 1, 0, 1, 0, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1]]]
            
            return options[random.randint(0,len(options)-1)]
        
        def choose_piece3():
            
            options = [[[1, 1, 1, 0, 1, 1, 1, 1],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                        [0, 1, 0, 1, 1, 1, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 1, 0, 1, 1, 0, 0],
                        [1, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 0, 1, 1, 1]],

                       [[1, 1, 1, 0, 1, 1, 1, 0],
                        [0, 0, 1, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 0, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 1]],

                       [[1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 0, 1, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 1, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 0, 1, 1, 1],
                        [0, 0, 1, 1, 0, 0, 0, 1]],

                       [[0, 1, 1, 0, 1, 1, 1, 0],
                        [0, 0, 0, 0, 1, 0, 1, 0],
                        [1, 1, 0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0],
                        [1, 0, 1, 0, 1, 0, 0, 0],
                        [1, 0, 1, 0, 1, 0, 1, 1]],

                       [[1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 0, 0, 0, 0, 1],
                        [0, 0, 0, 0, 1, 1, 0, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 1, 1, 1, 0, 1, 1]]]

            return options[random.randint(0,len(options)-1)]
    
        def choose_piece4():
            
            options = [[[1, 0, 0, 0, 0, 1, 1, 1],
                       [1, 0, 1, 1, 0, 0, 0, 1],
                        [1, 0, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 0, 0, 0, 1],
                        [0, 0, 0, 1, 1, 1, 0, 0],
                        [1, 0, 1, 1, 0, 1, 1, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0],
                        [1, 1, 1, 1, 1, 3, 0, 1]],

                       [[0, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0, 1, 1, 1],
                        [1, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 3, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 1],
                        [1, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 1, 3, 0, 0, 0],
                        [1, 0, 1, 1, 1, 1, 1, 0],
                        [1, 0, 1, 1, 0, 1, 1, 0],
                        [1, 0, 0, 0, 0, 0, 0, 0]],

                       [[1, 0, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 0, 1, 1, 0, 1, 1, 1],
                        [0, 0, 1, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 1, 1, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 3]],

                       [[3, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 1, 0, 0, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1],
                        [1, 1, 0, 1, 1, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 0, 0],
                        [1, 1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 1, 0, 0, 0, 1]]]
            
            return options[random.randint(0,len(options)-1)]
        
        # Dictionary of pieces we are using
        pieces={1:choose_piece1(),2:choose_piece2(),3:choose_piece3(),4:choose_piece4()}
        
        # Place pieces in state
        
        ## Intiate pieces
        piece1=pieces[1]
        piece2=pieces[2]
        piece3=pieces[3]
        piece4=pieces[4]
        
        ## Insert pieces in state
        for i in range(len(state)):
            for j in range(len(state[0])):
                
                # if in Piece 1's range
                if i<=7 and j<=7:
                    state[i][j]=piece1[i%8][j%8]
                
                # if in Piece 2's range
                if i<=7 and 7<j:
                    state[i][j]=piece2[i%8][j%8]
                
                # if in Piece 3's range
                if 7<i and j<=7:
                    state[i][j]=piece3[i%8][j%8]
                    
                # if in Piece4's range
                if 7<i and 7<j:
                    state[i][j]=piece4[i%8][j%8]
        
        # print(state)
        return state
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
        
        # print(percept)
        
        return percept
    
    def update(self,action,state):
        #Testing with 2 in state
        state[0][0]=2
        
        
        
        pass
    
    def find_positions(self,state,first=False):
        positions=[]
        
        #Cycle through matrix
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j]==2:
                    positions.append([i,j])
                #If first is True, find the first instantce of mini agent
                if state[i][j]==2 and first==True:
                    return [i,j]
            
        return positions
    
    def visualize(self,state):
        # Custom color map
        cmap=ListedColormap(["#ffffff","#666666","#3d85c6","#fbbc04"])
        
        # Create plot
        fig,ax=plt.subplots()
        ax.matshow(state,cmap=cmap)
        ax.axis("off")
        
        # Show plot
        fig.show()
        
        return None
    
    def transition_model(self,action,state):
        # Testing zone
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # self.visualize(state)
        
        # Get positions
        positions=self.find_positions(state)
        
        # Initate results
        result=[]
        
        for position in positions:
            # Find percept
            percept=self.percept(state,position)

            # If North
            if action=="N":
                # Check for a wall
                if percept[0] in "12":
                    # False if there's a wall
                    result.append(False)
                else:
                    result.append(True)
                    
            # If East
            if action=="E":
                if percept[1] in "12":
                    result.append(False)
                else:
                    result.append(True)
                    
            # If West
            if action=="W":
                if percept[2] in "12":
                    result.append(False)
                else:
                    result.append(True)
                    
            # if South
            if action=="S":
                if percept[3] in "12":
                    result.append(False)
                else:
                    result.append(True)
   
        return result
environment=Environment(initial_state)
