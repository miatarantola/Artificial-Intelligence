"""
CS311 Programming Assignment 2

Full Name(s): Mia Tarantola

Brief description of my solver: Implemented forward tracking and most constrained. Yes, it performed better than AC3 solver.
This algorithm looks forward as well as backwards and the most constrained heauristic helps to find failures faster.
 My time also improved after implementing it.

"""

import argparse, time
from os import remove
import copy
from typing import Dict, List, Optional, Set, Tuple

# You are welcome to add constants, but do not modify the pre-existing constants

# Length of side of a Soduku board
SIDE = 9

# Length of side of "box" within a Soduku board
BOX = 3

# Domain for cells in Soduku board
DOMAIN = range(1, 10)

# Helper constant for checking a Soduku solution
SOLUTION = set(DOMAIN)


def check_solution(board: List[int], original_board: List[int]) -> bool:
    """Return True if board is a valid Sudoku solution to original_board puzzle"""
    # Original board values are maintained
    for s, o in zip(board, original_board):
        if o != 0 and s != o:
            return False
    for i in range(SIDE):
        # Valid row
        if set(board[i * SIDE : (i + 1) * SIDE]) != SOLUTION:
            return False
        # Valid column
        if set(board[i : SIDE * SIDE : SIDE]) != SOLUTION:
            return False
        # Valid Box
        box_row, box_col = (i // BOX) * BOX, (i % BOX) * BOX
        box = set()
        for r in range(box_row, box_row + BOX):
            box.update(board[r * SIDE + box_col : r * SIDE + box_col + BOX])
        if box != SOLUTION:
            return False
    return True


def backtracking_search(neighbors: List[List[int]], queue: Set[Tuple[int, int]], domains: List[List[int]]) -> Tuple[Optional[List[int]], int]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable

    Returns:
        Tuple[Optional[List[int]], int]: Solution or None indicating no solution found and the number of recursive backtracking calls
    """
    # Track the number of recursive calls to backtrack
    recursions = 0
    assignment = generate_assignments(domains)

    # Defining a function within another creates a closure that has access to variables in the 
    # enclosing scope (e.g., to neighbors, etc). To be able to reassign those variables we use
    # the 'nonlocal' specification
    def backtrack(assignment: Dict[int, int]) -> Optional[Dict[int, int]]:
        """Backtrack search recursive function
        Args:
            assignment (Dict[int, int]): Values currently assigned to variables (variable index as key)

        Returns:
            Optional[Dict[int, int]]: Valid assignment or None if assignment is inconsistent
        """
        nonlocal recursions # Enable us to reassign the recursions variable in the enclosing scope

        #if length of assignments = num of spots on board

        if (len(assignment) ==SIDE**2):
            return assignment
        
        #picks one index that hasn't been assigned
        curr_index = get_unassigned(assignment)

        #possibly order better
        for possibility in domains[curr_index]:
  
            #pick assignment
            assignment[curr_index] = possibility
 
            #we remove these from domain
            removed =[]

            #for items in domain, add anything other than curr guess to removed
            for x in domains[curr_index]:
                if x!= possibility:
                    removed.append((curr_index,x))
            
            #means domain only have 1 now
            domains[curr_index] = [possibility]

            #run AC3
            is_consistent, items_removed = AC3(domains,neighbors,[(constraint,curr_index) for constraint in neighbors[curr_index]])    
            
            if len(items_removed)>0: #items were removed
                removed.extend(items_removed) #add to list
                
            #if AC3 is consistent
            if is_consistent == True:

                #recurse through
                result = backtrack(assignment)

                #if result didn't fail, return solution
                if (result != None):
                    return result
            
            #if it failed and is not consist we need to restore
            for index1, val in removed:
                domains[index1].append(val)
        
        #delete assignment and choose diff path
       
        if len(domains[curr_index])!=0:
            del assignment[curr_index]

        recursions += 1
        return None
    
    result = backtrack(assignment)
    
    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, recursions

def get_unassigned(assignment: Dict[int, int]):
    for i in range(81):
        if i not in assignment:
            return i
   




def AC3(domains,neighbors,queue=None):
    """
    performs AC3 algorithm
    Args:neighbors (List[List[int]]): Indices of neighbors for each variable
        domains (List[List[int]]): Domains for each variable
    Output: returns true if arc is consistent, false if otherwise and the list of removed values

    """
    def arc_reduction(index1, index2):
        removed=[]
        switched = False

        for val in domains[index1].copy(): #iterate through the values in the first positions domain
            found = False

            for item in domains[index2]: #iterate though vals in the second position's domain
                if val!=item: #if there is a value in one and not the other
                    found = True
                
            if found == False: #if domains are the same
                domains[index1].remove(val) #remove val
                removed.append((index1,val)) #add val to removed list
                switched = True
        return switched, removed

    removed = []

    if queue ==None: #if queue is empty
        queue=[]
        for indices in range(0,SIDE**2): #add curr index and all its neighbors
            queue = queue + [(indices,items) for items in neighbors[indices]]

    while len(queue)>0:
        #get first item        
        index,constraint = queue.pop()
     

        #see is anythign was changed and what was removed
        switched,removal= arc_reduction(index, constraint)

        #if something was removed
        if removal:
            removed.extend(removal)
            
        if switched:
            #means not arc consist
            if len(domains[index]) ==0:
                return False, removed
            
            #if removed value check neighbors
            else:
                queue = queue+[(index,n) for n in neighbors[index] if n!=constraint]
    return True, removed

def sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using backtracking search with the AC3 algorithm

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and a count of calls to recursive backtracking function
    """
    
    domains = [[val] if val else list(DOMAIN) for val in board]

    neighbors = []
    queue = set()


    neighbors=[]
    i=0
    while i<81:
        #initialize new set, so don't have to worry about duplicates
        indiv_neighbors=set()

        #find row and col
        row = i//SIDE
        col = i-(row*SIDE)

        #add row and col numbers to neighbors
        indiv_neighbors.update(list(range(SIDE*(row),SIDE*(row+1))))
        indiv_neighbors.update(list(range(col,SIDE**2,SIDE)))
        
        #find the box row and col
        box_row = int(row//(SIDE**.5))
        box_col = int(col/(SIDE**.5))
        box = set()
        for r in range(3):
            box_part = set(range((box_row*SIDE*BOX)+BOX*box_col+(r*SIDE),(box_row*SIDE*BOX)+BOX*box_col+3+(r*SIDE)))
            box.update(box_part)

        #indiv_neighbors.update(BOX[box])
        indiv_neighbors.update(box)


        #remove self
        indiv_neighbors.remove(i)

        #convert back to list
        indiv_neighbors = list(indiv_neighbors)

        #add to list of neighbors
        neighbors.append(indiv_neighbors)
        i+=1
    
    for i in range (len(domains)):
        if len(domains[i]) ==1:
            for n in neighbors[i]:
                if domains[i][0] in domains[n]:
                    domains[n].remove(domains[i][0])
   
    
    # TODO: Complete the initialization of the neighbors and queue data structures

    return backtracking_search(neighbors, queue, domains)

def generate_assignments(domain):
    """
    Generates the assignment dictionary: [value] = assignment
    Args: domains (List[List[int]]): Domains for each variable
    Output: intial assignment dictionary: Dict[int, int]
    """
    assignment=dict()
    for i in range(SIDE**2):
        if len(domain[i])==1:
            assignment[i] = domain[i][0]
    return assignment

############################################################################################################
############################################################################################################
# my algorithms
pruned = dict()

def my_get_unassigned(assignment: Dict[int, int],domains):
    """
    finds the position with the most contraints
    Args: assignment (Dict[int, int]): Values currently assigned to variables (variable index as key)
        domains (List[List[int]]): Domains for each variable
    Output: returns index with the most constraints
    """

    index = 0
    least = float('inf') #set the minimum to infinity
    for i in range(SIDE**2): #iterate thorugh positions
        if i not in assignment: #if index not assigned
            if len(domains[i])<least: #if length less the curr minimum
                least = len(domains[i]) #set new min length
                index = i #set new minimum index
    return index


def consistent(neighbors, assignment,index, val):
    """
    determines if arc is consistent
    Args: assignment (Dict[int, int]): Values currently assigned to variables (variable index as key)
          neighbors (List[List[int]]): Indices of neighbors for each variable
          index: curr index
          val: current assignment guess
    Output: returns True if consistent, False otherwise
    """

    is_consist = True
    for n in neighbors[index]: #iterate thorugh neighbors

        if n in assignment: #if neighbors is assigned
            if assignment[n] == val: #if neighbors assignment is the same as the current guess
                is_consist = False #not consistent
    return is_consist


def my_backtracking_search(neighbors: List[List[int]], queue: Set[Tuple[int, int]], domains: List[List[int]]) -> Tuple[Optional[List[int]], int]:
    """Perform backtracking search on CSP using AC3

    Args:
        neighbors (List[List[int]]): Indices of neighbors for each variable
        queue (Set[Tuple[int, int]]): Variable constraints; (x, y) indicates x must be consistent with y
        domains (List[List[int]]): Domains for each variable

    Returns:
        Tuple[Optional[List[int]], int]: Solution or None indicating no solution found and the number of recursive backtracking calls
    """
    # Track the number of recursive calls to backtrack
    recursions = 0
    assignment = generate_assignments(domains)

    # Defining a function within another creates a closure that has access to variables in the 
    # enclosing scope (e.g., to neighbors, etc). To be able to reassign those variables we use
    # the 'nonlocal' specification
    def backtrack(assignment: Dict[int, int], domains: List[List[int]]) -> Optional[Dict[int, int]]:
        """Backtrack search recursive function
        Args:
            assignment (Dict[int, int]): Values currently assigned to variables (variable index as key)

        Returns:
            Optional[Dict[int, int]]: Valid assignment or None if assignment is inconsistent
        """
        nonlocal recursions # Enable us to reassign the recursions variable in the enclosing scope

        if len(assignment)==SIDE**2:
            return assignment 
        
        curr = my_get_unassigned(assignment,domains) #get index with the most contraints

        for guess in domains[curr]: #for possible values in curr index's domain
            consist = consistent(neighbors, assignment,curr,guess) #see if it is consistent
       
            if consist: #if yes                                     FORWARD TRACKING
                assignment[curr]=guess                              #make assignment guess
                if len(domains)>0:                                  #if there are unassigned indices
                    for n in neighbors[curr]:                       #iterate through neighbors
                        if n not in assignment:                     # if neighbor not assigned
                            if guess in domains[n]:                 #if guess in neighbors domain
                                domains[n].remove(guess)            #remove guess from neighbors domain
                                pruned[curr].append((n,guess))      #add (neighbor,guess) to pruned list

                result = backtrack(assignment,domains)              #recurse
                
                if result!=None:                                    #if result isn't none
                    return result                                   #return result

                if curr in assignment:                              #undo if none, if curr in assignment
                    for (neighbor,val) in pruned[curr]:             #for each neighbor,guess pair in pruned
                        domains[neighbor].append(val)               #add back to neighbor's domain
                    pruned[curr]=[]                                 #reset pruned list
                    del assignment[curr]                            #delete assignment

        recursions += 1
       
        
        return None
    
    result = backtrack(assignment,domains)
    
    # Convert assignment dictionary to list
    if result is not None:
        result = [result[i] for i in range(SIDE * SIDE)]
    return result, recursions



def my_sudoku(board: List[int]) -> Tuple[Optional[List[int]], int]:
    """Solve Sudoku puzzle using your own custom solver

    Do not change the signature of this function

    Args:
        board (List[int]): Flattened list of board in row-wise order. Cells that are not initially filled should be 0.

    Returns:
        Tuple[Optional[List[int]], int]: Solution as flattened list in row-wise order, or None, if no solution found and a count of calls to recursive backtracking function
    """
    domains = [[val] if val else list(DOMAIN) for val in board]

    neighbors = []
    queue = set()

    neighbors=[]
    i=0
    while i<81:
        #initialize new set, so don't have to worry about duplicates
        indiv_neighbors=set()

        #find row and col
        row = i//SIDE
        col = i-(row*SIDE)

        #add row and col numbers to neighbors
        indiv_neighbors.update(list(range(SIDE*(row),SIDE*(row+1))))
        indiv_neighbors.update(list(range(col,SIDE**2,SIDE)))
        
        #find the box row and col
        box_row = int(row//(SIDE**.5))
        box_col = int(col/(SIDE**.5))
        box = set()
        for r in range(3):
            box_part = set(range((box_row*SIDE*BOX)+BOX*box_col+(r*SIDE),(box_row*SIDE*BOX)+BOX*box_col+3+(r*SIDE)))
            box.update(box_part)

        #indiv_neighbors.update(BOX[box])
        indiv_neighbors.update(box)

        #remove self
        indiv_neighbors.remove(i)

        #convert back to list
        indiv_neighbors = list(indiv_neighbors)

        #add to list of neighbors
        neighbors.append(indiv_neighbors)
        
        i+=1

    

   
    for i in range (len(domains)):

        if len(domains[i]) ==1:
            pruned[i]=domains[i]
            for n in neighbors[i]:
  
                if domains[i][0] in domains[n]:
                    domains[n].remove(domains[i][0])
        else:
            pruned[i] = []
   
    
    # TODO: Complete the initialization of the neighbors and queue data structures

    return my_backtracking_search(neighbors, queue, domains)
    


if __name__ == "__main__":
    # You should not need to modify any of this code
    parser = argparse.ArgumentParser(description="Run sudoku solver")
    parser.add_argument(
        "-a",
        "--algo",
        default="ac3",
        help="Algorithm (one of ac3, custom)",
    )
    parser.add_argument(
        "-l",
        "--level",
        default="easy",
        help="Difficulty level (one of easy, medium, hard)",
    )
    parser.add_argument(
        "-t",
        "--trials",
        default=1,
        type=int,
        help="Number of trials for timing",
    )
    parser.add_argument("puzzle", nargs="?", type=str, default=None)

    args = parser.parse_args()

    # fmt: off
    if args.puzzle:
        board = [int(c) for c in args.puzzle]
        if len(board) != SIDE*SIDE or set(board) > (set(DOMAIN) | { 0 }):
            raise ValueError("Invalid puzzle specification, it must be board length string with digits 0-9")
    elif args.level == "easy":

        board = [
            0,4,2,0,0,0,3,0,0,0,0,2,0,0,1,0
        ]
    elif args.level == "medium":
        board = [
            0,4,0,0,9,8,0,0,5,
            0,0,0,4,0,0,6,0,8,
            0,5,0,0,0,0,0,0,0,
            7,0,1,0,0,9,0,2,0,
            0,0,0,0,8,0,0,0,0,
            0,9,0,6,0,0,3,0,1,
            0,0,0,0,0,0,0,7,0,
            6,0,2,0,0,7,0,0,0,
            3,0,0,8,4,0,0,6,0,
        ]
    elif args.level == "hard":
        board = [
            1,2,0,4,0,0,3,0,0,
            3,0,0,0,1,0,0,5,0,  
            0,0,6,0,0,0,1,0,0,  
            7,0,0,0,9,0,0,0,0,    
            0,4,0,6,0,3,0,0,0,    
            0,0,3,0,0,2,0,0,0,    
            5,0,0,0,8,0,7,0,0,    
            0,0,7,0,0,0,0,0,5,    
            0,0,0,0,0,0,0,9,8,
        ]
    else:
        raise ValueError("Unknown level")
    # fmt: on

    if args.algo == "ac3":
        solver = sudoku
    elif args.algo == "custom":
        solver = my_sudoku
    else:
        raise ValueError("Unknown algorithm type")

    times = []
    for i in range(args.trials):
        test_board = board[:] # Ensure original board is not modified
        start = time.perf_counter()
        solution, recursions = solver(test_board)

        if solution!=None:
            for i in range(0,81,9):
                print(solution[i:i+9])

        end = time.perf_counter()
        times.append(end - start)
        if solution and not check_solution(solution, board):
            raise ValueError("Invalid solution")

        if solution:
            print(f"Trial {i} solved with {recursions} recursions")
            print(solution)
        else:
            print(f"Trial {i} not solved with {recursions} recursions")

    print(
        f"Minimum time {min(times)}s, Average time {sum(times) / args.trials}s (over {args.trials} trials)"
    )
