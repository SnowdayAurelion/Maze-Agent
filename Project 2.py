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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
    def __init__(self,name,initial_state,percept,Environment):        
        # Initiate variables
        self.name=name
        self.initial_state=initial_state
        self.state=copy.deepcopy(self.initial_state)
        self.enviro=Environment
        # Save positions of all the mini-agents (avoids rescanning)
        self.positions=[]
        
        # Place mini-agents at possible places
        self.place_agents(percept,True)
        
        # Place data for first index
        self.save_data()
        
        self.visualize()
        
        # Run the Agent until it finds the goal
        i=0
        while i<10:
            i+=1
            self.run()
        
        # Testing
        # print(self.scan())
        # sys.exit()
        # print(self.enviro.percept())
        # print(self.data)
        
        # Run Agent
        # while True:
        #     self.run()
        
    def run(self):
        # Check if at goal
        if self.enviro.goal_test():
            print("Goal was found")
            return None
        
        # Get percept from Environment
        percept=self.enviro.percept()
        
        # Update belief state with percept to remove old false mini-agents
        self.update(percept)
        
        # Choose a move
        self.choose_action()
        
        # Save the data
        self.save_data()
        
        # Officially move the Agent
        self.move()
        
        # Show new state of Agent
        self.visualize()
        
    def place_agents(self,percept=None,place_positions=False):
        # Find positions with same percept as given
        if percept!=None:
            for i in range(len(self.initial_state)):
                for j in range(len(self.initial_state[0])):
                    if percept==self.scan([i,j]).replace("2","0") and self.state[i][j]==0:
                        self.positions.append([i,j])
        
        # Reset state
        self.state=copy.deepcopy(self.initial_state)
        
        # Place positions from self.positions
        if place_positions:
            for position in self.positions:
                self.state[position[0]][position[1]]=2
                    
    def save_data(self):
        # Check if self.data exists
        if not "data" in dir(self):
            self.data=pd.DataFrame(data={"Belief state":[self.state],
                              "Positions":[self.positions],
                              "Percept":self.enviro.percept(),
                              "Move":None},index=[0],)
            return self.data
        
        # If it exists, then add on new data
        new_data=pd.DataFrame(data={"Belief state":[self.state],
                          "Positions":[self.positions],
                          "Percept":self.enviro.percept(),
                          "Move":self.action},index=[0])

        self.data=pd.concat([self.data,new_data],ignore_index=True)    
        return self.data
                
    def scan(self,position=None):  #percept to give to agent #Might use just to initialize the agent
        # If position=None, find first mini agent
        if position==None:
            positions=self.find_positions(True)     #Need find_positons
        
        # Initialize percept
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

    def find_positions(self,first=False):
        positions=[]
        
        # Cycle through matrix
        for i in range(len(self.state)):
            for j in range(len(self.state[0])):
                if self.state[i][j]==2:
                    positions.append([i,j])
                    
                # If first is True, find the first instantce of mini agent
                if self.state[i][j]==2 and first==True:
                    return [i,j]
        
        return positions
        
    def predict(self):
        return 
    
    def update(self,percept):
        # If the position is not the same as the percept, then remove mini-agent
        for position in self.positions:
            if self.scan(position).replace("2","0")!=percept:
                self.positions.remove(position)
                
        return None
    
    def move(self):
        # Initiate new positions to replace self.positions
        new_positions=[]
        
        # Move each mini-agent to the choosen direction
        for position in self.positions:
            
            # Get position of Agent
            i=position[0]
            j=position[1]
            
            # If action is North, verify with transition model
            if self.action=="N" and self.transition_model("N",position):
                
                # Place Agent one step North
                new_positions.append([i-1,j])
                
            # If action is East
            if self.action=="E" and self.transition_model("E",position):
                new_positions.append([i,j+1])
                
            # If action is West
            if self.action=="W" and self.transition_model("W",position):
                new_positions.append([i,j-1])
    
            # If action is South
            if self.action=="S" and self.transition_model("S",position):
                new_positions.append([i+1,j])

        # Update self.positions
        self.positions=new_positions
        
        # Place new mini-agents
        self.place_agents(place_positions=True)
        
        # Officially update the Environment state
        self.enviro.update(self.action)
            
        return None
        
    def cost(self,action,position):
        #each step is one point
        return random.randint(0,10)
    
    def transition_model(self,action,position):
        # Find percept
        percept=self.scan(position)

        # If North
        if action=="N":
            # Check for a wall
            if percept[0] in "1":
                # False if there's a wall
                return False
            else:
                return True
                
        # If East
        if action=="E":
            if percept[1] in "1":
                return False
            else:
                return True
                
        # If West
        if action=="W":
            if percept[2] in "1":
                return False
            else:
                return True
                
        # if South
        if action=="S":
            if percept[3] in "1":
                return False
            else:
                return True
    
    def choose_action(self):
        # Initiate costs of all mini-agents
        costs=[]
        
        # For every position
        for position in self.find_positions():
            # Initiate cost of specific mini-agent
            mini_agent_costs=[]           
            
            # For every action
            for action in ["N","E","W","S"]:
                # If direction is possible, find cost of being at that new position
                if self.transition_model(action, position):
                   mini_agent_costs.append(self.cost(action,[position[0],position[1]]))
                else:
                    # Give high cost if it's not possible
                    mini_agent_costs.append(100)
                
            # After append to costs
            costs.append(mini_agent_costs)
         
        # print(costs)
        # Return the direction that gives the lowest cost
        
        # Initiate final costs to find direction from
        final_costs=[]
        
        for i in range(4):
            c=0
            
            # Add up the costs for each direction
            for mini_agent_costs in costs:
                c+=mini_agent_costs[i]
            
            final_costs.append(c)
        
        # Translate index to direction
        # print(final_costs)
        # print(["N","E","W","S"][final_costs.index(min(final_costs))])
        self.action=["N","E","W","S"][final_costs.index(min(final_costs))]
        return ["N","E","W","S"][final_costs.index(min(final_costs))]
    
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
        self.Agent=Agent(self.agent_name,initial_state_clueless,percept,self)
        
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
        
        # self.visualize()
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
data=environment.Agent.data