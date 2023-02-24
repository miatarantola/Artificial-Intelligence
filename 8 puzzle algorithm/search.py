"""
CS311 Programming Assignment 1

Full Name(s): Mia Tarantola

Brief description of my heuristic: I added linear conflicts to the manhattan distance. It is more efficient
because the manhattan disance does not account for this, but if two tiles need to go "through" each other, 
each tile must move atleast once (penalty of +2 for each conflict)

I also added the last move  and corner tile heuristic accounting for linear conflict overlap

these additions only add to the manhattan dist w/o going over the true value

TODO Briefly describe your heuristic and why it is more efficient
"""

import argparse, itertools, random
from typing import Callable, List, Optional, Sequence, Tuple
from collections import deque
from queue import PriorityQueue
import numpy as np


# Problem constants. The goal is a "blank" (0) in bottom right corner
BOARD_SIZE = 3
GOAL = tuple(range(1, BOARD_SIZE**2)) + (0,)


def inversions(board: Sequence[int]) -> int:
    """Return the number of times a larger 'piece' precedes a 'smaller' piece in board"""
    return sum(
        (a > b and a != 0 and b != 0) for (a, b) in itertools.combinations(board, 2)
    )


class Node:
    def __init__(self, state: Sequence[int], parent: "Node" = None, cost=0):
        """Create Node to track particular state and associated parent and cost

        State is tracked as a "row-wise" sequence, i.e., the board (with _ as the blank)
        1 2 3
        4 5 6
        7 8 _
        is represented as (1, 2, 3, 4, 5, 6, 7, 8, 0) with the blank represented with a 0

        Args:
            state (Sequence[int]): State for this node, typically a list, e.g. [0, 1, 2, 3, 4, 5, 6, 7, 8]
            parent (Node, optional): Parent node, None indicates the root node. Defaults to None.
            cost (int, optional): Cost in moves to reach this node. Defaults to 0.
        """
        self.state = tuple(state)  # To facilitate "hashable" make state immutable
        self.parent = parent
        self.cost = cost

    def is_goal(self) -> bool:
        """Return True if Node has goal state"""
        return self.state == GOAL

    def expand(self) -> List["Node"]:
        """Expand current node into possible child nodes with corresponding parent and cost"""
        index_0 = self.state.index(0)
        
        row = index_0//BOARD_SIZE
        col = index_0-(row*BOARD_SIZE)
      
        children = []

        """if in the upper left corner"""
        if index_0==0:
            """move down"""
            new_state1 = Node._swap(self,0, 0, 1,0)
            n1 = Node(new_state1,parent = self,cost = self.cost+1)
            children.append(n1)

            """move right"""
            new_state2 = Node._swap(self,0,0,0,1)
            n2 = Node(new_state2, parent = self, cost = self.cost +1)
            children.append(n2)

        #empty space in bottom right corner
        elif index_0 == (BOARD_SIZE**2)-1:
            #move to the left
            new_state1 = Node._swap(self,BOARD_SIZE-1,BOARD_SIZE-1, BOARD_SIZE-1, BOARD_SIZE-2)
            n1 = Node(new_state1, parent = self, cost = self.cost+1)
            children.append(n1)


            #move up
            new_state2 = Node._swap(self, BOARD_SIZE-1, BOARD_SIZE-1, BOARD_SIZE-2, BOARD_SIZE-1)
            n2 = Node(new_state2, parent = self, cost = self.cost +1)
            children.append(n2)
        
        #empty space in bottom left corner
        elif index_0 == (BOARD_SIZE**2)-BOARD_SIZE:

            #move up
            new_state1 = Node._swap(self, BOARD_SIZE-1,0, BOARD_SIZE-2, 0)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move right
            new_state2 = Node._swap(self, BOARD_SIZE-1,0,BOARD_SIZE-1,1 )
            n2 = Node(new_state2, parent = self, cost = self.cost+1)
            children.append(n2)
        
        #empty space in top right corner
        elif index_0 == BOARD_SIZE-1:

            #move down
            new_state1 = Node._swap(self, 0, BOARD_SIZE-1, 1, BOARD_SIZE-1)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move left
            new_state2 = Node._swap(self, 0, BOARD_SIZE-1,0,BOARD_SIZE-2)
            n2 = Node(new_state2, parent = self, cost = self.cost+1)
            children.append(n2)
        
        #empty space in top row
        elif row ==0:

            #move down
            new_state1 = Node._swap(self, 0, col, 1, col)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move right
            new_state2 = Node._swap(self, 0, col, 0, col+1)
            n2 = Node(new_state2, parent = self, cost = self.cost +1 )
            children.append(n2)

            #move left
            new_state3 = Node._swap(self, 0, col, 0, col-1)
            n3 = Node(new_state3, parent = self, cost = self.cost +1 )
            children.append(n3)

        #empty space bottom row
        elif row == BOARD_SIZE-1:

            #move up
            new_state1 = Node._swap(self, BOARD_SIZE-1, col, BOARD_SIZE-2, col)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move right
            new_state2 = Node._swap(self, BOARD_SIZE-1, col, BOARD_SIZE-1, col+1)
            n2 = Node(new_state2, parent = self, cost = self.cost +1 )
            children.append(n2)

            #move left
            new_state3 = Node._swap(self, BOARD_SIZE-1, col, BOARD_SIZE-1, col-1)
            n3 = Node(new_state3, parent = self, cost = self.cost +1 )
            children.append(n3)

        #empty space in first column
        elif col ==0:

            #move up
            new_state1 = Node._swap(self, row, 0, row-1, col)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move down
            new_state2 = Node._swap(self, row, 0, row+1, col)
            n2 = Node(new_state2, parent = self, cost = self.cost +1 )
            children.append(n2)

            #move right
            new_state3 = Node._swap(self, row, 0, row, col+1)
            n3 = Node(new_state3, parent = self, cost = self.cost +1 )
            children.append(n3)
        
        #empty space in last column
        elif col == BOARD_SIZE-1:

            #move left
            new_state1 = Node._swap(self, row, col, row, col-1)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move down
            new_state2 = Node._swap(self, row, col, row-1, col)
            n2 = Node(new_state2, parent = self, cost = self.cost +1 )
            children.append(n2)

            #move up
            new_state3 = Node._swap(self, row, col, row+1, col)
            n3 = Node(new_state3, parent = self, cost = self.cost +1 )
            children.append(n3)

        #empty space in the center 
        else:
            
            #move left
            new_state1 = Node._swap(self, row, col, row, col-1)
            n1 = Node(new_state1, parent = self, cost = self.cost +1 )
            children.append(n1)

            #move right
            new_state2 = Node._swap(self, row, col, row, col+1)
            n2 = Node(new_state2, parent = self, cost = self.cost +1 )
            children.append(n2)

            #move down
            new_state3 = Node._swap(self, row, col, row+1, col)
            n3 = Node(new_state3, parent = self, cost = self.cost +1 )
            children.append(n3)

            #move up
            new_state4 = Node._swap(self, row, col, row-1, col)
            n4 = Node(new_state4, parent = self, cost = self.cost +1 )
            children.append(n4)




            
            
             



        # TODO: Implement this function to generate child nodes based on the current state
        

        return children

    def _swap(self, row1: int, col1: int, row2: int, col2: int) -> Sequence[int]:
        """Swap values in current state bewteen row1,col1 and row2,col2, returning new "state" to construct a Node"""
        state = list(self.state)
        state[row1 * BOARD_SIZE + col1], state[row2 * BOARD_SIZE + col2] = (
            state[row2 * BOARD_SIZE + col2],
            state[row1 * BOARD_SIZE + col1],
        )
        return state

    def __str__(self):
        return str(self.state)

    # The following methods enable Node to be used in types that use hashing (sets, dictionaries) or perform comparisons. Note
    # that the comparisons are performed exclusively on the state and ignore parent and cost values.

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __lt__(self, other):
        return self.state < other.state


def bfs(initial_board: Sequence[int], max_depth=12) -> Tuple[Optional[Node], int]:

    """Perform breadth-first search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 13.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement BFS. Your function should return a tuple containing the solution node and number of unique node explored
   
    frontier = deque([Node(initial_board)]) #initialize frontier
    reached = [] #initialize reached

    while len(frontier)>0: #while frontier not empty
        
        curr_node = frontier.popleft() #pop first node
        reached.append(curr_node.state) #add node to reached
            
        if curr_node.cost == max_depth+1: #if past the allowed depth
            return (None, len(reached)) #return none

        elif curr_node.is_goal(): #if current node is the goal return node and len reached
            return (curr_node, len(reached))

        else:
            for node in curr_node.expand(): #expand node
                if node.state not in reached: #if it hasnt been reached
                    frontier.append(node) #add to frontier and reached
                    reached.append(node.state)
    


def manhattan_distance(node: Node) -> int:
    """Compute manhattan distance heuristic for node, i.e., g(node) + f(node)"""
    # TODO: Implement the Manhattan distance heuristic (sum of Manhattan distances to goal location)
    distance = 0

    for i in range(len(node.state)):
        
        curr_num = node.state[i]
        if curr_num !=0:

            #get current coordinates
            row = i//BOARD_SIZE
            col = i-(row*BOARD_SIZE)
            
            goal_index = GOAL.index(curr_num)
            #goal coordinates
            goal_row = goal_index//BOARD_SIZE
            goal_col = goal_index-(goal_row*BOARD_SIZE)

            #find mahattan distance
            distance += abs(row-goal_row)+abs(col-goal_col)

    #add cost
    return distance+node.cost

def extra_heuristic(node:Node) -> int:
    lists = list(node.state)
    
    def grid(initial_board):
        #turns the initial tuple board into a board size x boards size matrix
        grid = np.zeros((BOARD_SIZE,BOARD_SIZE))
     
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                grid[i][j] = initial_board.pop(0)
        
        return grid

    def index_2d(grid, num):
        #finds the coordinates of a given number in a 2d matrix 
        
        for y, x in enumerate(grid):
            if num in x:
                return y, np.where(x==num)[0][0] #returns (row,col)
   
        
    reached =[]
    puzzle = grid(lists)
    goal = grid(list(GOAL))
    num_conflicts = 0 #number of conflicts
 
    nums_in_conflict=[] #numbers involved in conflicts

    for i in range(BOARD_SIZE):
        for j in range((BOARD_SIZE)):

            correct_pos = index_2d(goal,puzzle[i][j]) #find correct postion of the current num
          
            if puzzle[i][j]==0: #dont include 0
                continue
            elif puzzle[i][j] == goal[i][j]: #if in the correct spot --> continue
                continue

            #correct row
            elif correct_pos[0] == i:
               
                #for each item in row
       
                for k in puzzle[correct_pos[0]]:
            

                    #if that item 
                    if k != puzzle[i][j]:
                        #if that number is in the same row
                        if k in goal[correct_pos[0]]:

                            #if the curr index of k is in between the current nums local and goal index
                            if (min(correct_pos[1],j)<=np.where(puzzle[correct_pos[0]]==k)[0][0]<=max(correct_pos[1],j)) :
                                
                                #accounts for [x 1 2] not a conflcit
                                if (puzzle[i][j]>k) and (j>np.where(puzzle[correct_pos[0]]==k)[0][0]):
                                    continue
                                
                                #accounts for [4 5 x] not a conflict
                                elif (puzzle[i][j]<k) and (j<np.where(puzzle[correct_pos[0]]==(k))[0][0]):
                                    continue
                                else:
                                    if puzzle[i][j] in nums_in_conflict and k in nums_in_conflict:
                                        continue #both numbers can be in the reached already
                                    else:
                                        if puzzle[i][j] not in nums_in_conflict:
                                                nums_in_conflict.append(puzzle[i][j])
                                        nums_in_conflict.append(k) #add num to list
                                        num_conflicts +=1 #increase count
                                
           #correct col
            elif correct_pos[1] == j:
               
               #for items in correct col
                for m in puzzle[:,correct_pos[1]]:
                    if m != puzzle[i][j]:
                        if m in goal[:,j]:

                            #if m is in between curr nums goal and local col index 
                            if (min(correct_pos[0],i)<=np.where(puzzle[:,correct_pos[1]]==m)[0][0]<=max(correct_pos[0],i)):
                                # accounts for a similar case to rows but in the vertical direction
                                
                                if (puzzle[i][j]<m) and (i<np.where(puzzle[:,correct_pos[1]]==m)[0][0]):
                                    continue
                                elif (puzzle[i][j]>m) and i>np.where(puzzle[:,correct_pos[1]]==(m))[0][0]:
                                    continue
                                
                                else:
                                    if puzzle[i][j] in nums_in_conflict and m in nums_in_conflict:
                                        continue #both cant be in
                                    else:
                                        num_conflicts+=1 #increase count
                                        if puzzle[i][j] not in nums_in_conflict:
                                                nums_in_conflict.append(puzzle[i][j]) #add to list
                                        nums_in_conflict.append(m) #add to list

    #last move     (korf and taylor)                           
    num1 = (BOARD_SIZE**2)-1 #correct num right to the left of correct space
    goal1 = index_2d(goal,num1)  #index of num 1
    num2 = goal[-2][-1] #correct num right above correct space
    goal2 = index_2d(goal,num2) #index
    last_move_tally=0
    
    ''' if num1 not in puzzle[:,-1]: #if number to left not in the last col
        if (num1 in puzzle[:,goal1[1]]) and (num1 in nums_in_conflict): #if in goal col and in conflict
            last_move_tally+=0 #do nothing those 2 moves from conflict could move it
        else:
            last_move_tally+=1 #add one
    
    if num2 not in puzzle[-1]: #number right above not in last row
        
        if (num2 in puzzle[goal2[0]]) and (num2 in nums_in_conflict): #if num is goal row and in conflict
       
            last_move_tally+=0 #do nothing
        else:
            
            last_move_tally+=1 #add tally must move '''
    
    #corner (korf and taylor)  
    corner_tally = 0
    #upper right
    if (goal[0][-2] == puzzle[0][-2])& (goal[0][-1] != puzzle[0][-1]): #number to the left of the corner is right but corner not
        if puzzle[0][-2] not in nums_in_conflict: #num to left of corner not in conflict

            corner_tally+=1 #add tally
    elif (goal[1][-1] == puzzle[1][-1] and (goal[0][-1] != puzzle[0][-1])): #number below corner is correct but corner not
        if puzzle[1][-1] not in nums_in_conflict: #if num not in conflict

            corner_tally +=1
    elif (goal[0][-2] == puzzle[0][-2])& (goal[0][-1] != puzzle[0][-1])&(goal[1][-1] == puzzle[1][-1]): #both spots are right, corner not
        if (puzzle[0][-2] not in nums_in_conflict) &(puzzle[1][-1] not in nums_in_conflict): #both not in conflict

            corner_tally+=4 #add 4 (korf and taylor)  

    #upper left follow above pattern
    if (goal[0][0]!= puzzle[0][0]) & (goal[0][1] == puzzle[0][1]):
        if puzzle[0][1] not in nums_in_conflict:

            corner_tally+=1
    elif(goal[0][0]!=puzzle[0][0])&(goal[1][0]==puzzle[1][0]):
        if goal[1][0] not in nums_in_conflict:

            corner_tally +=1
    elif (goal[0][0]!=puzzle[0][0])&(goal[1][0]==puzzle[1][0])&(goal[0][1] == puzzle[0][1]):
        if (goal[1][0] not in nums_in_conflict) & (puzzle[0][1] not in nums_in_conflict):

            corner_tally+=4
    
    #bottom left 
    if (goal[-1][0]!= puzzle[-1][0])&(goal[-2][0]==puzzle[-2][0]):
        if (puzzle[-2][0] not in nums_in_conflict) & (puzzle[-1][0]!=0 )&( puzzle[-2][0]!=0):

            corner_tally+=1
    elif(goal[-1][0]!=puzzle[-1][0])& (goal[-1][1]==puzzle[-1][1]):
        if puzzle[-1][1] not in nums_in_conflict:

            corner_tally+=1
    elif (goal[-1][0]!=puzzle[-1][0])& (goal[-1][1]==puzzle[-1][1]) & (goal[-2][0]==puzzle[-2][0]):
        if (puzzle[-1][1] not in nums_in_conflict) & (puzzle[-2][0] not in nums_in_conflict):
     
            corner_tally+=4
    #bottom right overlap too hard

            
    #add all elements   
    return num_conflicts+last_move_tally+corner_tally


def custom_heuristic(node: Node) -> int:
    # TODO: Implement and document your _admissable_ heuristic function
    #add custom heuristic to manhattan

    if node.is_goal():
        return 0
    else:
        return manhattan_distance(node)+extra_heuristic(node)



def astar(
    initial_board: Sequence[int],
    max_depth=12,
    heuristic: Callable[[Node], int] = manhattan_distance,
) -> Tuple[Optional[Node], int]:
    """Perform astar search to find 8-squares solution

    Args:
        initial_board (Sequence[int]): Starting board
        max_depth (int, optional): Maximum moves to search. Defaults to 13.
        heuristic (_Callable[[Node], int], optional): Heuristic function. Defaults to manhattan_distance.

    Returns:
        Tuple[Optional[Node], int]: Tuple of solution Node (or None if no solution found) and number of unique nodes explored
    """
    # TODO: Implement A* search
    curr = Node(initial_board)
    frontier = PriorityQueue() #PQ 
    frontier.put((heuristic(curr),curr)) #add first node
    reached = dict() 
    fn = heuristic(curr) #fn


    while (frontier.qsize!=0) :#when not empty
        poppedX = frontier.get() #pop first node (lowest value) and value
        X = poppedX[1] #get node
        

        if X.state == GOAL: #if node is goal
         
            return (X, len(reached)) #return node and # unique 
        
        if(X.cost>=max_depth): #if over allowed depth, no sol
            return (None, len(reached))
           
        else:
            successors = X.expand() #expand node
            for node in successors:
                if (node in reached): #if node reached 
                    if (node.cost>=reached[node]): #if node cost higher than curr
                        continue #nothing
                    if(node.cost<reached[node]): #lower --> replace
                        reached[node]= fn + X.cost +1
                else:
                    fn = heuristic(node) 
                    reached[node]=fn + X.cost +1
                    frontier.put((fn,node)) #add node
                    


if __name__ == "__main__":

    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(
        description="Run search algorithms in random inputs"
    )
    parser.add_argument(
        "-a",
        "--algo",
        default="bfs",
        help="Algorithm (one of bfs, astar, astar_custom)",
    )
    parser.add_argument(
        "-i",
        "--iter",
        type=int,
        default=1000,
        help="Number of iterations",
    )
    parser.add_argument(
        "-s",
        "--state",
        type=str,
        default=None,
        help="Execute a single iteration using this board configuration specified as a string, e.g., 123456780",
    )

    args = parser.parse_args()

    num_solutions = 0
    num_cost = 0
    num_nodes = 0

    if args.algo == "bfs":
        algo = bfs
    elif args.algo == "astar":
        algo = astar
    elif args.algo == "astar_custom":
        algo = lambda board: astar(board, heuristic=custom_heuristic)
    else:
        raise ValueError("Unknown algorithm type")

    if args.state is None:
        iterations = args.iter
        while iterations > 0:
            init_state = list(range(BOARD_SIZE**2))
            random.shuffle(init_state)

            # A problem is only solvable if the parity of the initial state matches that
            # of the goal.
            if inversions(init_state) % 2 != inversions(GOAL) % 2:
                continue

            solution, nodes = algo(init_state)
            if solution:
                num_solutions += 1
                num_cost += solution.cost
                num_nodes += nodes

            iterations -= 1
    else:
        # Attempt single input state
        solution, nodes = algo([int(s) for s in args.state])
        if solution:
            num_solutions = 1
            num_cost = solution.cost
            num_nodes = nodes

    if num_solutions:
        print(
            "Iterations:",
            args.iter,
            "Solutions:",
            num_solutions,
            "Average moves:",
            num_cost / num_solutions,
            "Average nodes:",
            num_nodes / num_solutions,
        )
    else:
        print("Iterations:", args.iter, "Solutions: 0")