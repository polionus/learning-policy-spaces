from __future__ import annotations
import time
import torch
import os

from functools import partial

###TODO: Fix the multiprocessing problem and why passing the model 
from multiprocessing import Pool, set_start_method



# from logger.collector import Collector

from dsl import DSL
from search.top_down import TopDownSearch
from tasks.policy.policy_ import Policy
from logger.stdout_logger import StdoutLogger
from config import Config
from src.config.runtime import Runtime
from vae.models.sketch_vae import SketchVAE
from vae.models import StrimmedVAE, BaseVAE
from src.utils.experiment_result import ExperimentResult, build_experiment_data_class
from src.utils.metadata import get_meta_data

# Use network to execute actions instead of programs
metadata = get_meta_data()

#### I don't need multi-processing, I can vmap all the search and run it on the GPU.

def execute_policy(z, task_envs):
        mean_reward = 0.0
        for task_env in task_envs:
            # we reinstantiate again just to
            # reconfigure the policy to the right config
            policy = Policy(z, Runtime.policy_model, task_env)
            mean_reward += task_env.evaluate_policy(policy)
        num_evals = 1
        mean_reward /= len(task_envs)
        return z, num_evals, mean_reward


##### We are going to write down a jax refactor of this, so we need to make sure that we are 

class LatentSearch:
    """Implements the CEM method from LEAPS paper."""

    def __init__(
        self, model: BaseVAE, task_cls, dsl: DSL):
        self.model = model
        if issubclass(type(self.model), SketchVAE):
            self.dsl = dsl.extend_dsl()
        else:
            self.dsl = dsl
        self.device = self.model.device
        self.population_size = Config.search_population_size
        self.elitism_rate = Config.search_elitism_rate
        self.n_elite = int(Config.search_elitism_rate * self.population_size)

        self.sigma_min = Config.search_sigma_min
        self.sigma_rate = Config.search_sigma_exponential_decay

        
        self.number_executions = Config.search_number_executions
        self.number_iterations = Config.search_number_iterations
        self.sigma = Config.search_sigma
        self.model_hidden_size = Config.model_hidden_size
        self.task_envs = [task_cls(i) for i in range(self.number_executions)]
        self.program_filler = TopDownSearch()
        self.filler_iterations = Config.search_topdown_iterations
        output_dir = os.path.join("output", Config.experiment_name, "latent_search")
        os.makedirs(output_dir, exist_ok=True)

        ### Collector: We should initialize the collector here. Perhpas I need to import the collector here
        ### and give it the output.
        self.output_file = os.path.join(output_dir, f"model_type_{Config.model_name}-task_{Config.env_task}-model_{Config.model_hidden_size}-seed_{Config.search_seed}.csv")
       
        #Define the experiment

        ####TODO: find a way to only define these fields inside the experiment class, since their names are the same, it is probably possible to automatically read them
        
        ###NOTE: the parentheses after the build is important, to instantiate the dataclass!
        experiment = build_experiment_data_class(config_dict=metadata)()
        
        #Define the experiment result
        self.collector = ExperimentResult(experiment, 
                                           fields=['time', 'num_evaluations', 'best_reward'])

        ## I can worry about these later.
        self.trace_file = os.path.join(output_dir, f"seed_{Config.model_seed}.gif")
        self.restart_timeout = Config.search_restart_timeout
        self.seed = Config.search_seed
        torch.manual_seed(self.seed)

    def search_sigma_anneal(self):
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_rate)

    def get_neighbors(self, new_indices, elite_population):
        new_population = []
        for index in new_indices:
            sample = elite_population[index]
            new_population.append(
                sample
                + self.sigma * torch.randn_like(sample, device=self.device)
            )
        population = torch.stack(new_population)
        return population

    def init_population(self) -> torch.Tensor:
        """Initializes the CEM population from a normal distribution.

        Returns:
            torch.Tensor: Initial population as a tensor.
        """
        return torch.stack(
            [
                torch.randn(self.model_hidden_size, device=self.device)
                for _ in range(self.population_size)
            ]
        )

    def execute_population(
        self, population: torch.Tensor
    ) -> tuple[list[str], torch.Tensor, int]:

        # for all the different z vectors, execute their corresponding policies, and get the reward
        
        if Config.multiprocessing_active:
            with Pool() as pool:
                fn = partial(execute_policy, task_envs = self.task_envs)
                #The torch tensor is not an iterable compatible with the pool funciton, so you need to make it into a list
                results = pool.map(fn, population)
        else:

            ###TODO: make this work better by not reinstantiating the model over and over. 
            ###This will allow you to use multiprocessing to make runs super fast
            results = [
                execute_policy(task_envs=self.task_envs, z = z) for z in population
            ]

        rewards = []
        for z, num_evals, reward in results:
            # reward = self.execute_policy(task_envs=self.task_envs, z=z)
            rewards.append(reward)
            self.num_evaluations += num_evals #This could be simplfied in the future
            if reward > self.best_reward:
                
                self.best_reward = reward
                self.best_program = z.clone()

                StdoutLogger.log(
                    "Latent Search", f"New best reward: {self.best_reward}"
                )
                StdoutLogger.log(
                    "Latent Search", f"New best program: {self.best_program}"
                )
                StdoutLogger.log(
                    "Latent Search", f"Number of evaluations: {self.num_evaluations}"
                )

                ### Another place to use the collector instead.
                t = time.time() - self.start_time
                self.collector.collect({'time':t, 'num_evaluations':self.num_evaluations, 'best_reward':self.best_reward})

            if self.best_reward >= 1.0:
                self.converged = True
                break
                
        return torch.tensor(rewards, device=self.device)

    def search(self) -> tuple[str, bool, int]:
        """Main search method. Searches for a program using the specified DSL that yields the
        highest reward at the specified task.

        Returns:
            tuple[str, bool]: Best program in string format and a boolean value indicating
            if the search has converged.


        """

        population = self.init_population()
        self.converged = False
        self.num_evaluations = 0
        counter_for_restart = 0
        self.best_reward = -float("inf")
        self.best_program = None
        prev_mean_elite_reward = -float("inf")
        self.start_time = time.time()

        for iteration in range(1, self.number_iterations + 1):
            
            rewards = self.execute_population(population)

            if self.converged:
                break

            best_indices = torch.topk(rewards, self.n_elite).indices
            elite_population = population[best_indices]
            mean_elite_reward = torch.mean(rewards[best_indices])

            StdoutLogger.log(
                "Latent Search",
                f"Iteration {iteration} mean elite reward: {mean_elite_reward}, num_evals: {self.num_evaluations}",
            )

            if mean_elite_reward.cpu().numpy() == prev_mean_elite_reward:
                counter_for_restart += 1
            else:
                counter_for_restart = 0
            if counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
                population = self.init_population()
                counter_for_restart = 0
                StdoutLogger.log("Latent Search", "Restarted population.")
            else:

                 
                if Config.search_method == "CEM":

                    if Config.search_reduce_to_mean:
                        new_indices = torch.zeros(self.population_size, device=self.device, dtype = torch.int16)
                        elite_population = torch.mean(elite_population, dim=0).repeat(self.n_elite, 1)

                    else: 
                        new_indices = torch.ones(elite_population.size(0), device=self.device).multinomial(
                    self.population_size, replacement=True)
    
                elif Config.search_method == "CEBS":
                    ###TODO: Implement the K/E non random version of this later
                    # assert self.population_size % self.n_elite == 0
                    new_indices = torch.arange(self.n_elite).repeat_interleave(int(self.population_size/self.n_elite))
                
                population = self.get_neighbors(new_indices=new_indices, elite_population=elite_population)
            prev_mean_elite_reward = mean_elite_reward.cpu().numpy()

            ## At the end of the iteration, anneal the sigma:
            self.search_sigma_anneal()


        # best_program_nodes = self.dsl.parse_str_to_node(self.best_program)
        # self.task_envs[0].trace_program(best_program_nodes, self.trace_file, 1000)

        if not self.converged:

            ### Yet another location to use the collector.
            t = time.time() - self.start_time
            self.collector.collect({'time':t, 'num_evaluations':self.num_evaluations, 'best_reward':self.best_reward})


        p = Policy(self.best_program, self.model, self.task_envs[0])

        ### TODO: Come back and fix this flag.
        if False:
            self.task_envs[0].trace_policy(
                p, image_name=f"./logs/gifs/{Config.env_task}-{0}.gif"
            )
        self.collector.save()
        return self.best_program, self.converged, self.num_evaluations
