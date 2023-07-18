import numpy as np
import random
import gym
from gym import spaces 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



""" Initialization of parameters """

# global variables for the GridWorld
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

GREEN = 0
BLUE = 1
RED = 2
ORANGE = 3

GRIDWORLD_SIZE = 9

EPOCHS = 5 # Number of trajectories to be generated
VISUALIZATION_ON = True
RANDOM_GOALS = False
FileNames = ["daten/DELETEME_TEACHER.txt", "daten/DELETEME_TEST.txt"]


""" Food class """
class Food():
    """
    functionality:  initalizes a Food
    input:          color: color of the Food (Green, Blue, Red or Orange)
                    position: position of the food in the GridWorld as a 1D index
    usage:          helper class for the GridWorld
    """
    def __init__(self, color, position):
        self.color = color
        self.position = position



""" GridWorld class """
class GridWorld(gym.Env):
    """
    functionality:  initializes a GridWorld
    input:          n: edge-length of the GridWorld -> Gridworld has n^2 positions
                    preferred_color: color the agent wants to reach
                    with_wall: whether or not the GridWorld contains a wall
    usage:          GridWorld-states will represent the trajectories of the agent
    """
    def __init__(self, n=GRIDWORLD_SIZE, with_wall=True):
        self.n = n
        self.n_states = self.n ** 2 + 1  # all possible states
        self.preferred_color = GREEN     #self.random_color()
        if RANDOM_GOALS:
            self.preferred_color = self.random_color()
        self.with_wall = with_wall
        self.food = self.compute_food_positions()
        self.terminal_state = self.food[self.preferred_color].position  # terminal state: position of goal (favoured food)
        self.wall = self.compute_wall()
        self.distracting_states = self.compute_distractors()  # all not-preferred foods and the wall 
        self.absorbing_state = self.n_states - 1
        self.step_count = 0
        self.visited = []
        self.done = False
        self.start_state = np.random.randint(self.n_states - 2)

        while self.start_state in self.distracting_states or self.start_state == self.terminal_state:  # No objects at the same
            self.start_state = np.random.randint(self.n_states - 2)                                    # coordinates

        self._reset()
        self.action_space = spaces.Discrete(4)



    """ Distractors and Foods """

    """
    functionality:  randomly assigns the preferred_color of food
    input:
    output:         preferred_color: GREEN, BLUE, RED, ORANGE
    usage:          goal of agent should be randomly assigned
    """
    def random_color(self):
        preferred_color = random.randint(0,3)
        
        return preferred_color

    """
    functionality:  generates 4 random food-positions as 1D indices 
    input:          
    output:         foods: list with 4 food-objects at random positions
    usage:          GridWorld needs foods
    """
    def compute_food_positions(self):            
        positions = [-1, -1, -1, -1]
        for x in range(0, 4):
            y = np.random.randint(self.n_states - 2)     # y is a random position
            while y in positions:                        # not at an occupied position!
                y = np.random.randint(self.n_states - 2)
            positions[x] = y

        foods = []
        for x in range(0, 4):                    # foods are placed at positions
            foods.append(Food(x, positions[x]))  # food-sequence: green, blue, red, orange

        return foods

    """
    functionality:  puts all undesired GridWorld-objects (foods and wall) into one list
    input:          
    output:         result: list with all undesired objects (foods and wall)
    usage:          agent needs this to know where (not) to go
    """
    def compute_distractors(self):               
        result = []
        for i in range(0, 4):
            if i != self.preferred_color:
                result.append(self.food[i].position)
        for i in range(0, len(self.wall)):
            result.append(self.wall[i])

        return result

    """
    functionality:  randomly generates the wall-position (5 fields) as 1D indices 
    input:          
    output:         wall: list with the 5 neighboring wall-positions or empty list
    usage:          GridWorld with a wall is a more interesting challenge for the agent     
    """
    def compute_wall(self):
        wall = []
        rand = random.random()
        not_done = True
        food_positions = [-1, -1, -1, -1]
        for i in range(0, 4):
            food_positions[i] = self.food[i].position

        if not self.with_wall or rand < 0.2:  # GridWorld without wall
            return wall

        elif rand < 0.6:                      # GridWorld with horizontal wall
            while not_done:
                wall = [-1, -1, -1, -1, -1]
                row = np.random.randint(0, 9)
                startcol = np.random.randint(0, 4)
                for x in range(0, 5):
                    col = startcol + x
                    pos = self.coord2ind([row, col])
                    if pos not in food_positions:
                        wall[x] = pos
                        if x == 4:
                            not_done = False
                    else:
                        break
            return wall

        else:                                 # GridWorld with vertical wall
            while not_done:
                wall = [-1, -1, -1, -1, -1]
                col = np.random.randint(0, 9)
                startrow = np.random.randint(0, 5)
                for x in range(0, 5):
                    row = startrow + x
                    pos = self.coord2ind([row, col])
                    if pos not in food_positions:
                        wall[x] = pos
                        if x == 4:
                            not_done = False
                    else:
                        break
            return wall
        
        

    """ Move, FindPath and generate Output """

    """
    functionality:  the coordinates of the agent are changed according to the action he performs
    input:          action: the next action the agent performs
    output:         res: [7 x n x n] array (7 arrays: start, agent, four foods, wall), represents the GridWorld-state after the action
                    state: current position of the agent
                    done: whether agent has reached his goal or not
    usage:          agent performs a step in the GridWorld
    """
    def _step(self, action):
        assert self.action_space.contains(action)  # is agent's action valid?

        for distractor in self.distracting_states:
            if self.state == distractor:
                print("FEHLER! Position ist besetzt.")

        if self.state == self.terminal_state:  # goal reached
            self.done = True
            datei = open(FileNames[1], 'a')# export ending of trajectory
            datei.write("\n" + "s")
            
            datei = open(FileNames[0], 'a')
            datei.write("\n" + "s")

            return self.state, self.done, None
        
        
        [row, col] = self.ind2coord(self.state)  # transformation in 2D
    
        if action == UP:
            row = max(row - 1, 0)                # changing the coordinates of the agent
        elif action == DOWN:
            row = min(row + 1, self.n - 1)
        elif action == RIGHT:
            col = min(col + 1, self.n - 1)
        elif action == LEFT:
            col = max(col - 1, 0)

        self.visited.append([row, col])         # saves the already walked path, 2D
        new_state = self.coord2ind([row, col])  # transformation back to 1D

        self.state = new_state
    
        self.step_count += 1
        
        
        if(VISUALIZATION_ON):
            cmap = mcolors.ListedColormap(
                ["white", "saddlebrown", "gainsboro", "black", "yellow", "green", "blue", "red", "orange"])
            a = self.print_me()
            plt.imshow(a, cmap=cmap)
            plt.show()
            self.print_details()


        res = np.zeros([7, self.n, self.n])  # 1. start, 2.agent, 3.-6. foods, 7. wall
        [startrow, startcol] = self.ind2coord(self.start_state)
        res[0][startrow][startcol] = 1  # start
        res[1][row][col] = 1            # agent
        j = 2
        for food in self.food:          # food
            [foodrow, foodcol] = self.ind2coord(food.position)
            res[j][foodrow][foodcol] = 1
            j += 1

        for i in self.wall:             # wall
            [wallrow, wallcol] = self.ind2coord(i)
            res[6][wallrow][wallcol] = 1

        datei = open(FileNames[1], 'a') # export GridWorld state
        datei.write("\n" + str(res))

        teacher = np.zeros(4)
        teacher[action] = 1
        datei = open(FileNames[0], 'a') # export action
        datei.write("\n" + str(teacher))

        return res, self.state, self.done, None
    
    """
    functionality: print initial gridworld state and export it to text file
    input: -
    output: -
    usage: get complete trajectory by running this before using _step() to walk path
    """
    def initial_state(self):
        if(VISUALIZATION_ON):
            cmap = mcolors.ListedColormap(
                ["white", "saddlebrown", "gainsboro", "black", "yellow", "green", "blue", "red", "orange"])
            a = self.print_me()
            plt.imshow(a, cmap=cmap)
            plt.show()
            self.print_details()
        
        [row, col] = self.ind2coord(self.state)  # transformation in 2D
        
        res = np.zeros([7, self.n, self.n])  # 1. start, 2.agent, 3.-6. foods, 7. wall
        [startrow, startcol] = self.ind2coord(self.start_state)
        res[0][startrow][startcol] = 1  # start
        res[1][row][col] = 1            # agent
        j = 2
        for food in self.food:          # food
            [foodrow, foodcol] = self.ind2coord(food.position)
            res[j][foodrow][foodcol] = 1
            j += 1

        for i in self.wall:             # wall
            [wallrow, wallcol] = self.ind2coord(i)
            res[6][wallrow][wallcol] = 1

        datei = open(FileNames[1], 'a') # export GridWorld state
        datei.write("\n" + str(res))
        
        return
    
    
    """
    functionality:  an optimal path is computed via a breadth-first-search, starting at the end (terminal-state) 
                    and working backwards towards the initial position of the agent
    input:          
    output:         arr_dir: 2D array containing all directions from start to goal
    usage:          computes an optimal path for the agent to take in the GridWorld
    """
    def compute_path(self):  
        arr_dis = np.ones((self.n, self.n), dtype=int) * np.inf  # 2D array, saves distances, initialized with infinity
        arr_dir = np.ones((self.n, self.n), dtype=int) * 9       # 2D array, saves directions, initialized with 9

        coord = self.terminal_state  # terminal_state as index, single integer value
        q = []
        q.append(coord)                     
        [row, col] = self.ind2coord(coord)  
        arr_dis[row][col] = 0 # distance at the goal is 0

        while ((q != []) and (coord != self.start_state)):       # queue not empty and not yet at start
            [row, col] = self.ind2coord(coord)  

            liste = [0, 1, 2, 3]                         # Helper-list to randomize for-loop
            for i in range(4, 0, -1):                    # for all four actions:
                direction = liste[np.random.randint(i)]  # get a random direction
                liste.remove(direction)                  # remove the used direction from liste

                if direction == 0 and self.up_is_free([row, col]) and (
                arr_dis[row][col] + 1) < (arr_dis[row - 1][col]):   # update iff free and decreased distance
                    arr_dis[row - 1][col] = arr_dis[row][col] + 1   # increase distance by 1 or keep old distance (inf)
                    arr_dir[row - 1][col] = DOWN                    # we come from below: agent should go DOWN
                    q.append(self.coord2ind([row - 1, col]))        # enqueue new coordinate (1D)

                elif direction == 1 and self.right_is_free([row, col]) and (
                arr_dis[row][col] + 1) < (arr_dis[row][col + 1]):   # update iff free and decreased distance
                    arr_dis[row][col + 1] = arr_dis[row][col] + 1   # see above
                    arr_dir[row][col + 1] = LEFT                    # we come from left: agent should go LEFT
                    q.append(self.coord2ind([row, col + 1]))        # enqueue new coordinate (1D)

                elif direction == 2 and self.down_is_free([row, col]) and (
                arr_dis[row][col] + 1) < (arr_dis[row + 1][col]):   # see above
                    arr_dis[row + 1][col] = arr_dis[row][col] + 1   # see above
                    arr_dir[row + 1][col] = UP                      # we come from above: agent should go UP
                    q.append(self.coord2ind([row + 1, col]))        # see above

                elif direction == 3 and self.left_is_free([row, col]) and (
                arr_dis[row][col] + 1) < (arr_dis[row][col - 1]):   # see above
                    arr_dis[row][col - 1] = arr_dis[row][col] + 1   # see above
                    arr_dir[row][col - 1] = RIGHT                   # we come from right: agent should go RIGHT
                    q.append(self.coord2ind([row, col - 1]))        # see above
                    
            q.pop(0)  # dequeue current coord
            if (q != []):
                coord = q[0]

        return arr_dir

    """
    functionality:  the step-function is repeatedly performed on the directions of compute_path
    input:          
    output:         
    usage:          the agent walks along the computed path, a trajectory is produced
    """
    def walk_path(self):

        [row, col] = self.ind2coord(self.start_state)
        [termrow, termcol] = self.ind2coord(self.terminal_state)
        path = self.compute_path()
        nextdir = path[row][col]

        if nextdir == 9:
            self.done = True
            return
        
        # if nextdir is 9, then the goal is unreachable => agent stands still, trajectory is iterrupted
        
        while not (termrow == row and termcol == col) and not (nextdir == 9):
            self._step(nextdir)

            if nextdir == 0:
                row = row - 1
            if nextdir == 1:
                col = col + 1
            if nextdir == 2:
                row = row + 1
            if nextdir == 3:
                col = col - 1
            
            nextdir = path[row][col]
    
        self._step(0)   # <--- last step of trajectory, necessary for correct data-export
        


    """ helper functions """
    
    def up_is_free(self, coordinates):
        [row, col] = coordinates
        return self.is_free([row - 1, col])

    def down_is_free(self, coordinates):
        [row, col] = coordinates
        return self.is_free([row + 1, col])

    def left_is_free(self, coordinates):
        [row, col] = coordinates
        return self.is_free([row, col - 1])

    def right_is_free(self, coordinates):
        [row, col] = coordinates
        return self.is_free([row, col + 1])

    def is_free(self, coordinates):
        [row, col] = coordinates
        if row > 8 or col > 8 or row < 0 or col < 0:
            return False
        for x in self.distracting_states:
            [xrow, xcol] = self.ind2coord(x)
            if xrow == row and xcol == col:
                return False
        return True

    def ind2coord(self, index): # from 1D to 2D
        assert (index >= 0)  
        col = index // self.n
        row = index % self.n
        return [row, col]

    def coord2ind(self, coord): # from 2D to 1D
        [row, col] = coord
        assert (row < self.n)
        assert (col < self.n)
        return col * self.n + row

    def _reset(self):
        self.state = self.start_state if not isinstance(self.start_state, str) else np.random.randint(self.n_states - 2)
        self.done = False
        return self.state
    


    """ visualization """
    # !!! UNCOMMENT THIS AND CODE FURTHER ABOVE FOR TRAJECTORY VISUALIZATION !!!
    if(VISUALIZATION_ON):
        def print_details(self):
            print('self-state:', end=' ')
            print(self.ind2coord(self.state))
            print('goal-state:', end=' ')
            print(self.ind2coord(self.terminal_state))
            print('distracting states:', end=' ')
            for x in self.distracting_states:
                print(self.ind2coord(x), end=', ')
            print('')
    
        def print_me(self):
            twod_representation = np.zeros((self.n, self.n))
    
            for step in self.visited:
                i = step[0]
                j = step[1]
                twod_representation[i, j] = 2
    
            for brick in self.wall:
                coords = self.ind2coord(brick)
                i = coords[0]
                j = coords[1]
                twod_representation[i, j] = 1     # brown wall
    
            for state in range(self.n_states):
                coords = self.ind2coord(state)
                i = coords[0]
                j = coords[1]
                if state == self.state and state == self.start_state:  # agent at start position, show start position colour
                    twod_representation[i, j] = 4  # yellow
                elif state == self.state:
                    twod_representation[i, j] = 3  # black agent
                elif state == self.food[0].position:
                    twod_representation[i, j] = 5  # green food
                elif state == self.food[1].position:
                    twod_representation[i, j] = 6  # blue food
                elif state == self.food[2].position:
                    twod_representation[i, j] = 7  # red food
                elif state == self.food[3].position:
                    twod_representation[i, j] = 8  # orange food
                elif state == self.start_state:  # start position without agent, show start position colour
                    twod_representation[i, j] = 4  # yellow
    
            return twod_representation



""" data generation """
"""produces epochs-many trajectories and prints which trajectory is currently produced"""
epochs = EPOCHS  
for i in range(epochs):
    if i%100 ==0:
        print(i) 
    Example = GridWorld()
    Example.initial_state()
    while not Example.done:
        Example.walk_path()
