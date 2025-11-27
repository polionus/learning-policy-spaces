import jax
import numpy as np
from dataclasses import replace
from karel.world import (WorldState, 
                         move, 
                         turn_left, 
                         turn_right, 
                         pick_marker, 
                         put_marker, 
                         front_is_clear,
                         left_is_clear,
                         right_is_clear,
                         markers_present,
                         to_image,
                         )
MAKE_ENV = True

P = 0.7

#Let's remind ourselves how the state works: 
### The state has 3 dims, the last dim is a flag for each of the states. (in the state table)
### the first two dims are the grid themselves. 

### how to see which states have been visited: 
### Save the state tensor for all visited states. 
### Then, we can find the locations where the agent has been.


STATE_TABLE = {
    0: "Karel facing North",
    1: "Karel facing East",
    2: "Karel facing South",
    3: "Karel facing West",
    4: "Wall",
    5: "0 marker",
    6: "1 marker",
    7: "2 markers",
    8: "3 markers",
    9: "4 markers",
    10: "5 markers",
    11: "6 markers",
    12: "7 markers",
    13: "8 markers",
    14: "9 markers",
    15: "10 markers",
}



class FunctionalEnv:

    def __init__(self, world_state: WorldState, seed: int):
        self.seed = seed
        self._state = world_state
        self.key = jax.random.key(seed)
        self.visited_squares = np.zeros((self._state.s.shape[0], self._state.s.shape[1]))

        ## You should also capture the very first square
        x, y, _ = self._state.hero_pos
        self.visited_squares[x,y] = 1
     

    def _get_new_key(self):
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def __getattr__(self, name):
        # Called only if attribute not found on FunctionalEnv itself
        return getattr(self._state, name)
    
    def __repr__(self):
        return repr(self._state)
    
    
    def move(self):
        self._state = jax.jit(move)(self._state)
        x, y, _ = self._state.hero_pos
        self.visited_squares[x, y] = 1

        return self._state
    
    def turn_left(self):
        self._state = jax.jit(turn_left)(self._state)
        return self._state
    
    def turn_right(self):
        self._state = jax.jit(turn_right)(self._state)
        # self._state = turn_right(self._state)
        return self._state
    
    def pick_marker(self):

        self._state = jax.jit(pick_marker)(self._state)
        return self._state
    
    def put_marker(self):

        self._state = jax.jit(put_marker)(self._state)
        return self._state
    
    def front_is_clear(self):
        
        self._state, flag = jax.jit(front_is_clear)(self._state)
        return flag

    def left_is_clear(self):
        
        self._state, flag = jax.jit(left_is_clear)(self._state)

        return flag
    
    def right_is_clear(self):
        self._state, flag = jax.jit(right_is_clear)(self._state)     

        return flag
    
    def markers_present(self):
        self._state, flag = jax.jit(markers_present)(self._state)
        return flag
    
    def get_visited_squares(self):
        return self.visited_squares
    
    def make_map_from_visitation_table(self, p: float):

        '''
            This function assumes that the program has already finished execution inside the enviornment, and hence self._state.hero_pos reflects it final location.
        
        '''

        ## find unvisited squares
        Xs, Ys = np.where(self.visited_squares == 0) 
        
        ## set walls in unvisited squares.
        for x, y in zip(Xs, Ys):
            if np.random.random() > p: 
                self._state = replace(self._state, s = self._state.s.at[x, y, 4].set(True))

        ## set markers at the final state the agent resides in (using the STATE_TABLE above)
        x_final, y_final, _ = self._state.hero_pos
        self._state = replace(self._state, s = self._state.s.at[x_final, y_final, 6].set(True))
    







    