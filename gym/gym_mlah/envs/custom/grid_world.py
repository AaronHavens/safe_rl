import numpy as np
import gym
from gym import spaces

NOTHING = 0
LEFT = 1
RIGHT = 2
DOWN = 3
UP = 4

PIT_REWARD = -10.0
GOAL_REWARD = 100.0
STEP_REWARD = -1.0

class GridWorld(gym.Env):

    def __init__(self,n_dims=11,goal=[10,0],seed=None,pose=None):
        if seed is not None:
            np.random.seed(seed)
        self.dim = n_dims
        self.set_goal(goal)
        self.action_space = spaces.Discrete(5)
        self.low = np.array([0, 0])
        self.high = np.array([self.dim-1, self.dim-1])
        self.observation_space =  spaces.Box(self.low, self.high)
        self.start = [0,0]
        self.pit_map = [[0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [1,0,1,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0],
                        [2,0,0,0,0,0,0,0,0,0,0]]
        self.state = self.reset()

    def step(self,action):
        if(action == NOTHING):
            row = self.state[0]
            col = self.state[1]

        elif(action == LEFT):
            row = self.state[0]
            col = max(self.state[1] - 1, 0)

        elif(action == RIGHT):
            row = min(self.state[0] + 1, self.dim-1)
            col = self.state[1]

        elif(action == DOWN):
            row = self.state[0]
            col = min(self.state[1] + 1, self.dim-1)

        elif(action == UP):
            row = max(self.state[0] - 1, 0)
            col = self.state[1]

        state_ = np.array([row,col],dtype=int)


        if self.pit_map[state_[0]][state_[1]] == 2:
            reward = GOAL_REWARD
            self.done = True
        elif self.pit_map[state_[0]][state_[1]] == 1:
            reward = PIT_REWARD
        else:
            reward = STEP_REWARD + 10*(-self.l2_dist(state_)+self.l2_dist(self.state))
        self.state = state_
        return np.array(self.state,dtype=np.float64), reward, self.done, {}

    def set_goal(self,goal):
        self.goal = np.array(goal, dtype=int)

    def reset(self):
        if self.start is None:
            row,col = self.grid_select()
        else:
            row = self.start[0]
            col = self.start[1]

        self.state = np.array([row,col],dtype=int)
        self.done = False
        return self.state

    def l2_dist(self,state):
        inner = (self.goal[0]-state[0])**2 + (self.goal[1]-state[1])**2
        return np.sqrt(inner)

    def grid_select(self):
        row = np.random.randint(0,self.dim)
        col = np.random.randint(0,self.dim)
        return row,col

    def render(self, mode='human'):
        print('GridWorld')
        bar = ''
        for i in range(self.dim):
            bar += '----'
        bar += '-'
        print(bar)
        for i in range(self.dim):
            line = '|'
            for j in range(self.dim):
                if(np.array_equal(self.goal,np.array([i,j],dtype=int))):
                    pol_symb = 'G'
                elif(np.array_equal(self.state,np.array([i,j],dtype=int))):
                    pol_symb = 'A'
                else:
                    pol_symb = ' '

                line += ' %s |' % pol_symb
            print(line)
            print(bar)
        print('')
