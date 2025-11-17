import jax
from karel.world import WorldState, run_action


### TODO: Add a move axis option to the get state function to do all of the move axis stuff here, along with a flag in the init!
### When using jax, I probably do not need this class, If I implement the world object as a pytree module, eqx.Module, I can use vmap to batch.


### TODO: Understand how the world batch contained and used the information about tasks
### TODO: substittue the 'replace' command with a custom function that uses tree utils to modify the class (this is probably faster.) new_state = update_state(current_state, context)


### I probably do not have to reimplement the tasks into jax. 
def step(world_state: WorldState, action: jax.Array):
    new_world_state = run_action(world_state, action)

    return new_world_state.s, new_world_state


### I am not exactly sure, whether or not this object needs to exist. If my worlds are batched over with vmap, I don't need to make batch? 
# # ### or maybe I need to do the magic I did to make this into a batch.
# class WorldBatch:

#     def __init__(self, states: np.ndarray):
#         self.worlds: list[World] = []
#         for s in states:
#             self.worlds.append(World(s))

#     def step(self, actions):
#         assert len(self.worlds) == len(actions)
#         for w, a in zip(self.worlds, actions):
#             if (
#                 a < 5
#             ):  # Action 5 is the "do nothing" action, for filling up empty space in the array
#                 w.run_action(a)
#         return np.array([w.get_state() for w in self.worlds])
