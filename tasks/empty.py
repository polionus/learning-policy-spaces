import numpy as np
from karel.world import WorldState, make_world_state, STATE_TABLE 
from .task import Task

### In order to create the state, the last dimension of the state is the type . 

class EmptyTask(Task):

    #This function is called once in the task.py file.
    def generate_state(self):
        state = np.zeros(
            (self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool
        )

        state[2, 2, 0] = True
        return make_world_state(state)

    def reset_state(self) -> None:
        world_state = super().reset_state()
        return world_state

    ### This function is called when running the program or policy in the env.
    def get_reward(self, world_state: WorldState):

        terminated = False
        reward = 0

        return terminated, reward
