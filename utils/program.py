from dsl.base import Program
from dsl import DSL
from tasks.task import Task

from dsl.functional_wrapper import FunctionalEnv
def evaluate_program(self, task, program):
    world_state = FunctionalEnv(task.reset_state(), seed = self.seed)
    reward = 0.0
    
    for _ in program.run_generator(world_state):
        ### BUG: Somewhere the number of API calls should go up but it's not.
        terminated, instant_reward = task.get_reward(world_state)

        reward += instant_reward
        if terminated or world_state.crashed:
            break
    return reward, world_state

def execute_program(
    program_str: str, task_envs: list[Task], dsl: DSL
) -> tuple[Program, int, float]:
    try:
        program = dsl.parse_str_to_node(program_str)
    except (
        AssertionError
    ):  # In case of invalid program (e.g. does not have an ending token)
        return Program(), 0, -float("inf")
    
    # Evaluate single program
    mean_reward = 0.0
    for task_env in task_envs:
        mean_reward += evaluate_program(program)
    num_evaluations = 1
    mean_reward /= len(task_envs)
    
    return program, num_evaluations, mean_reward    
