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
    def __init__(self,name,initial_state,percept,Environment,goal=None):        
        # Initiate variables
        self.name=name
        self.initial_state=initial_state
        self.state=copy.deepcopy(self.initial_state)
        self.enviro=Environment
        self.goal=goal
        # Save positions of all the mini-agents (avoids rescanning)
        self.positions=[]
        
        # Place mini-agents at possible places
        self.place_agents(percept,True)
        
        # Place data for first index
        self.save_data()
        
        self.visualize()
        
        # Run the Agent until it finds the goal
        i=0
        while i<100:
            i+=1
            self.run()
            
            # Stopping code when there's one position or performance test
            if len(self.positions)==1:
                break
        
        
        # Testing
        # print(self.scan())
        # sys.exit()
        # print(self.enviro.percept())
        # print(self.data)
        # self.data.to_hdf("Project 2 Data.h4", "key")
        # print(self.action_to_percept(self.data["Positions"].iloc[-1][0], "N"))
        
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
        
        # Testing
        # print(self.possible_actions())
        
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
        
        # Eliminate mini-agents until there's one
        # if len(self.positions)!=1:
        if True:
            # print("Length of positions",len(self.positions))
            self.action=self.single_out_agent()
            return self.action
        
        print("No longer using single out")
        print("Positions",self.positions)
        sys.exit()
        
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
    
    ## AI section
    
    def single_out_agent(self):
        # Find first
        
        # positions=self.data["Positions"].iloc[-1]
        positions=self.positions
        
        # Scan mini agents' positions
        for position in positions:
            # Find possible actions for percept and current position
            possible_actions_percept=set(self.possible_actions())
            possible_actions_position=set(self.possible_actions(position))
            
            # Make sure the possible actions for position intersects with possible actions for percept
            if possible_actions_percept & possible_actions_position:
                
                # Possible actions are their intersection
                possible_actions=possible_actions_percept.intersection(possible_actions_position)
                
                # For every possible action
                for action in possible_actions:
                    # Find the new percept in that direction
                    moved_percept=self.action_to_percept(action,position).replace("2","0")
                    
                    # Check if the moved_percept is unique in the previous history of percepts
                    if moved_percept not in set(self.data["Percept"]):
                        # if moved_percept in percept:
                        print(possible_actions_percept,possible_actions_position,position)
                        print(moved_percept,set(self.data["Percept"]))
                        print(action,"\n")
                        return action
                    
                    # for i,percept in enumerate(self.data["Percept"]):
                    #     if moved_percept not in self.data["Percept"][:i]:
                    #         print(possible_actions_percept,possible_actions_position)
                    #         print(moved_percept,self.data["Percept"][:i])
                    #         return action

        # If it ran out of options, go a random possible direction
        possible_actions=set(self.possible_actions())
        # If it's not stuck in a corner, don't go back where it came from
        if len(possible_actions)>1:
            # print("Don't go back",possible_actions,self.reverse_action(self.data["Move"].iloc[-1]))
            possible_actions.remove(self.reverse_action(self.data["Move"].iloc[-1]))
            
            ## Choose option closest to the goal
            # Find mini-agents with same same possible moves
            positions_hcost={}
            for position in self.positions:
                if possible_actions & set(self.possible_actions(position)):
            
                    # Find possible positions' h cost for each action
                    for action in possible_actions:
                        if action not in positions_hcost.keys():
                            positions_hcost[action]=self.enviro.M_distance(self.action_to_position(action, position), self.goal)
                        else:
                            positions_hcost[action]+=self.enviro.M_distance(self.action_to_position(action, position), self.goal)
            
            # Choose action with lowest cost
            print("positions_hcost",positions_hcost)
            print(possible_actions)
            lowest_cost=min(positions_hcost.values())
            for key,value in positions_hcost.items():
                if lowest_cost==value:
                    action=key
            
            # action=positions_hcost[positions_hcost.values().index(min(positions_hcost.values()))]
            print(action)
            return action
            

            # possible_positions
            # h_costs={}
            # if
            
        print(possible_actions_percept,possible_actions_position)
        print("Stuck in corner",self.possible_actions())
        action=random.choice(tuple(possible_actions))
        print(action,"\n")
        return action
    
    def action_to_percept(self,action,position):
        # Based on action, update position
        i=position[0]
        j=position[1]
        
        if action=="N":
            i+=-1
        elif action=="E":
            j+=1
        elif action=="W":
            j+=-1
        elif action=="S":
            i+=1
        
        # Find percept of new position
        return self.scan([i,j])
    
    def action_to_position(self,action,position):
        i=position[0]
        j=position[1]
        
        if action=="N":
            i+=-1
        elif action=="E":
            j+=1
        elif action=="W":
            j+=-1
        elif action=="S":
            i+=1
            
        # Return new position
        return [i,j]
    
    def possible_actions(self,position=None):
        # Initiate list
        possible_actions=[]
        
        # If the position is given, find the percept. Else assume it's the most recent percept
        if position!=None:
            percept=self.scan(position).replace("2","0")
        else:
            percept=self.enviro.percept()
        
        # Find open squares
        for i,number in enumerate(percept):
            if number=="0" or number=="3":
                # N
                if i==0:
                    possible_actions.append("N")
                
                # E
                if i==1:
                    possible_actions.append("E")
                
                # W
                if i==2:
                    possible_actions.append("W")
                
                # S
                if i==3:
                    possible_actions.append("S")
        
        return possible_actions
    
    def reverse_action(self,action):
        if action=="N":
            return "S"
        if action=="E":
            return "W"
        if action=="W":
            return "E"
        if action=="S":
            return "N"
    
    # def performance_test(self):
    #     # This function will end the code once there is a single mini agent and
    #     # then export to a .txt file amount of moves. The values in the .txt
    #     # file then can be averaged.
    #     # print(len(self.positions),self.positions)
    #     if len(self.positions)==1:
    #         with open("Moves until single mini-agent.txt","a") as file:
    #             file.write(str(len(self.data)-1)+"\n")
    #             print("performance_test: single mini-agent detected, saving # of moves in .txt")
    #             return True
    
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
        self.Agent=Agent(self.agent_name,initial_state_clueless,percept,self,self.goal)
        
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
            
class Performance_Test:
    def __init__(self,iterations):
        self.result=[]
        self.i=iterations
        
        self.run()
        
    def run(self):
        i=0
        while i<self.i:
            i+=1
            
            # Call onto the class
            envir=Environment()
            
            # Once the Agent is done running in the Environment class, get data
            self.result.append(len(envir.Agent.data)-1)
            
        # Give result to user
        self.calculate()
    
    def calculate(self):
        print("The average moves after {} iterations is:\n{}".format(self.i,np.average(self.result)))
        print("Moves",self.result)

Performance_Test(100)

# environment=Environment()
# data=environment.Agent.data

