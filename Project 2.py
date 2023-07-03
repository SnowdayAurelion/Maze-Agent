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
import time

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
    def __init__(self,initial_state=None,agent_name="Pathfinder",difficulty="M"):
        # Initiate variables
        self.agent_name=agent_name
        self.state=initial_state
        self.goal=None

        # Generate initial state if not given
        if self.state==None:
            self.generate()
        
        # Set difficulty parameters
        
        ## If set to easy
        if difficulty=="E":
            min_dist=5
            max_dist=9
            
        ## If set to medium    
        if difficulty=="M":
            min_dist=17
            max_dist=20
            
        ## If set to hard
        if difficulty=="H":
            min_dist=28
            max_dist=31
        
        # Place Agent and Goal in map
        self.place_agent_and_goal(min_dist,max_dist)
        self.visualize()    
        
        # Find positon of Agent and Goal
        position=self.find_positions(True)
        self.goal=self.find_positions(True,True)
        
        # Make sure agent does not know it's position
        initial_state_clueless=copy.deepcopy(self.state)
        for i in range(len(initial_state_clueless)):
            for j in range(len(initial_state_clueless)):
                if initial_state_clueless[i][j]==2:
                    initial_state_clueless[i][j]=0
        
        # Get percept for Agent
        percept=self.percept(position)
        
        # Create Agent, and give data (state and percept)
        self.Agent=Agent(self.agent_name,initial_state_clueless,percept)
        
        print(self.Agent.data)

        #Save data in DataFrame
        # self.data=pd.DataFrame(columns=["Initial state","State","Percept"])
        # self.data["Initial state"]=initial_state
        # self.data["State"]=initial_state
        # self.data["Percept"]=percept
        
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
                        [1, 1, 1, 1, 1, 0, 0, 1]],

                       [[0, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 0, 0, 1, 1, 1],
                        [1, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 1],
                        [0, 0, 1, 1, 1, 1, 0, 1],
                        [1, 0, 1, 0, 1, 1, 0, 1],
                        [1, 0, 1, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 1, 0, 1]],

                       [[1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 0, 0, 0, 0, 0, 1, 1],
                        [1, 0, 1, 1, 0, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0],
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
                        [0, 0, 0, 0, 0, 0, 1, 0]],

                       [[0, 0, 1, 1, 0, 1, 1, 1],
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
        self.state=state
        return self.state
    
    def percept(self,position=None):  #percept to give to agent #Might use just to initialize the agent
        #If position=None, find first mini agent
        if position==None:
            position=self.find_positions(True)
        
        #Initialize percept
        percept=""
        
        #Calculate N, E, W, and S
        
        #N
        try:
            if position[0]!=0:
                percept+=str(self.state[position[0]-1][position[1]])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #E
        try:
            if position[1]!=len(self.state[0]):
                percept+=str(self.state[position[0]][position[1]+1])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #W
        try:
            if position[1]!=0:
                percept+=str(self.state[position[0]][position[1]-1])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        #S
        try:
            if position[0]!=len(self.state):
                percept+=str(self.state[position[0]+1][position[1]])
            else:
                percept+="1"
        except IndexError:
            percept+="1"
        
        # print(percept)
        
        return percept
    
    def update(self,action):
        # Initate indicator that says if Agent moved
        moved=False
        
        # Get position of Agent
        i=self.find_positions(True)[0]
        j=self.find_positions(True)[1]
        
        # If action is North, verify with transition model
        if action=="N" and self.transition_model("N")[0]:
            
            # Place Agent one step North
            self.state[i-1][j]=2
            
            # Say Agent moved
            moved=True
            
        # If action is East
        if action=="E" and self.transition_model("E")[0]:
            self.state[i][j+1]=2
            moved=True
            
        # If action is West
        if action=="W" and self.transition_model("W")[0]:
            self.state[i][j-1]=2
            moved=True

        # If action is South
        if action=="S" and self.transition_model("S")[0]:
            self.state[i+1][j]=2
            moved=True
            
        # Set old position as 0 if Agent moved
        if moved:
            self.state[i][j]=0
        
        if self.goal_test():
            print("The Agent has reached the goal!")
        
        return self.state
    
    def find_positions(self,first=False,find_goal=False):
        if find_goal:
            piece_type=3
        else:
            piece_type=2
        
        positions=[]
        
        # Cycle through matrix
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                if self.state[i][j]==piece_type:
                    positions.append([i,j])
                    
                # If first is True, find the first instantce of mini agent
                if self.state[i][j]==piece_type and first==True:
                    return [i,j]
            
        return positions
    
    def visualize(self):
        # Custom color map
        cmap=ListedColormap(["#ffffff","#666666","#3d85c6","#fbbc04"])
        
        # Create plot
        fig,ax=plt.subplots()
        ax.matshow(self.state,cmap=cmap)
        ax.axis("off")
        
        # Show plot
        fig.show()
        
        return None
    
    def transition_model(self,action):
        # Testing zone
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # state[random.randint(0,15)][random.randint(0,15)]=2
        # self.visualize()
        
        # Get positions
        positions=self.find_positions()
        
        # Initate results
        result=[]
        
        for position in positions:
            # Find percept
            percept=self.percept(position)

            # If North
            if action=="N":
                # Check for a wall
                if percept[0] in "1":
                    # False if there's a wall
                    result.append(False)
                else:
                    result.append(True)
                    
            # If East
            if action=="E":
                if percept[1] in "1":
                    result.append(False)
                else:
                    result.append(True)
                    
            # If West
            if action=="W":
                if percept[2] in "1":
                    result.append(False)
                else:
                    result.append(True)
                    
            # if South
            if action=="S":
                if percept[3] in "1":
                    result.append(False)
                else:
                    result.append(True)
                    
        # print(result)
        return result
    
    def goal_test(self):
        # If the position of the agent is same as goal return True
        if self.find_positions(True)==self.goal:
            return True
        else:
            return False
        
    def place_agent(self):
        while True:
            i=random.randint(0,15)
            j=random.randint(0,15)
            
            # If there is a free space, place the agent
            if self.state[i][j]==0:
                self.state[i][j]=2
                return self.state
            
    def place_goal(self):
        while True:
            i=random.randint(0,15)
            j=random.randint(0,15)
            
            # If there is a free space, place the goal
            if self.state[i][j]==0:
                self.state[i][j]=3
                return self.state
 
    def M_distance(self,position1,position2):
        return abs(position1[0]-position2[0])+abs(position1[1]-position2[1])
    
    def place_agent_and_goal(self,min_dist=17,max_dist=20):  
        # Save original state
        original_state=self.state

        while True:
            # Reset the state on each loop
            self.state=copy.deepcopy(original_state)
            
            # First place the agent
            self.place_agent()
            agent_position=self.find_positions(True)
            
            # Place goal
            self.place_goal()
            goal_position=self.find_positions(True,True)
            
            # Check if Manhattan distance between them is far enough
            if min_dist<self.M_distance(agent_position,goal_position)<max_dist:
                return self.state
            
environment=Environment()
