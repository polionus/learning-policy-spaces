from dsl.program_generator import ProgramGenerator
from dsl import DSL
from tasks import EmptyTask
from tqdm import tqdm 
from dsl.functional_wrapper import FunctionalEnv
import tyro 
import pickle
from utils.time import get_time_stamp
from utils.path_consts import PATH_TO_DATASETS
from karel.world import WorldState, run_and_trace
import os

### TODO: 1. Finish Debugging the base.py file. 2. Write down the dataset script and run it. 3. Write down ideas on how to implement it. 

all_tokens = [
    "DEF",
    "run",
    "m(",
    "m)",
    "move",
    "turnRight",
    "turnLeft",
    "pickMarker",
    "putMarker",
    "r(",
    "r)",
    "R=0",
    "R=1",
    "R=2",
    "R=3",
    "R=4",
    "R=5",
    "R=6",
    "R=7",
    "R=8",
    "R=9",
    "R=10",
    "R=11",
    "R=12",
    "R=13",
    "R=14",
    "R=15",
    "R=16",
    "R=17",
    "R=18",
    "R=19",
    "REPEAT",
    "c(",
    "c)",
    "i(",
    "i)",
    "e(",
    "e)",
    "IF",
    "IFELSE",
    "ELSE",
    "frontIsClear",
    "leftIsClear",
    "rightIsClear",
    "markersPresent",
    "noMarkersPresent",
    "not",
    "w(",
    "w)",
    "WHILE",
]


class ProblemGenerator:

    def __init__(self, dsl: DSL, 
                 num_progs: int, 
                 seed: int, 
                 program_generator: ProgramGenerator,
                 task,
                 num_problem_per_program: int = 1, 
                 shard_size: int = 1000,
                 ):

        self.dsl = dsl
        self.num_prob_per_prog = num_problem_per_program
        self.program_generator = program_generator
        # self.program_list = self._generate_programs(num_progs)
        self.seed = seed
        self.shard_size = shard_size
        
        self.task = task

      
    def _evaluate_program(self, task, program):
        world_state = FunctionalEnv(task.reset_state(), seed = self.seed)
        reward = 0.0
        
        for _ in program.run_generator(world_state):
            ### BUG: Somewhere the number of API calls should go up but it's not.
            terminated, instant_reward = task.get_reward(world_state)

            reward += instant_reward
            if terminated or world_state.crashed:
                break
        return reward, world_state
    

    def _generate_programs(self):
        
        all_progs = []

        ### Calls to the program generator will make new programs each time since we have saved the program generator and it is stateful. 
        for _ in tqdm(range(self.shard_size)):
            program = self.program_generator.generate_program()
            all_progs.append(program)

        return all_progs

    def make_problem_for_program(self, program, p: float):
    
        ### I wasn't returning the new enviornment, and the program generator     
        _,w0 = self._evaluate_program(self.task, program)
        w0.make_map_from_visitation_table(p=p)

        return w0
        
        
    def visualize_problem_program_as_gif(self, program, world_state: WorldState):
        run_and_trace(world_state, program, 'problem_program_trace.gif')

        
    def make_dataset_program_for_problem(self, p: float):

        '''
            This dataset generator will generate both the programs and their corresponding problems, and you will be able read the programs using dsl.parse_int_to_node to get the actual program that can be evaluated in the environment.
        '''
        self.program_list = self._generate_programs()

        maps = []
        programs = []
        for program in tqdm(self.program_list):
            
            for _ in range(self.num_prob_per_prog):
                w0 = self.make_problem_for_program(program, p)
                #self.visualize_problem_program_as_gif(program, w0)
                
                ## gather the data: 
                programs.append(self.dsl.parse_node_to_int(program))   
                maps.append(w0._state.s)

        dataset = {'progs': programs, 'maps': maps}

        return dataset

def main(save: bool = False, 
         num_progs: int = 100, 
         shard_size: int = 1000,
         p: float = 0.9, 
         seed: int = 0, 
         num_prob_per_prog: int = 3):
  

    task = EmptyTask(seed)
    dsl = DSL.init_default_karel()
    program_generator = ProgramGenerator(dsl, seed)
    problem_generator = ProblemGenerator(dsl, 
                                         num_progs=num_progs,
                                         seed = seed, 
                                         program_generator=program_generator,
                                         task = task,
                                         num_problem_per_program=num_prob_per_prog,
                                         shard_size=shard_size,
                                         )
    assert num_progs % shard_size == 0, "shard_size must divide num_progs"

    num_iterations = num_progs//shard_size
    for index in range(num_iterations):
        dataset = problem_generator.make_dataset_program_for_problem(p=p)
    
        if save:
            timestamp = get_time_stamp()
            os.makedirs(f"{PATH_TO_DATASETS}/{timestamp}", exist_ok=True)
            with open(f"{PATH_TO_DATASETS}/{timestamp}/problem_dataset-shard{index}.pkl", "wb") as f:
                pickle.dump(dataset, f)

    return dataset

if __name__ == "__main__":
    maps = tyro.cli(main)

 

    
    

    

