from abc import abstractmethod, ABC
from typing import Callable, List
from functools import partial
from multiprocessing import Pool, set_start_method
from dsl import DSL
from dsl.base import Program
from tasks.task import Task
from config import Config

import jax
import jax.numpy as jnp
from jax.lax import top_k

from search_config import get_logging_message
set_start_method('fork')


### TODO: create a search method config file in which you determine the logging and collection logic:
### TODO: understand how the task_cls thing works
### TODO: Learn to integrate the trace policy method
### TODO: Make self static: the parameters of self should not be changing, use arguments in functinos to pass into other functions.

###NOTE: the only things we should jit are computations that happen over and over, with the same logic. 


### TODO: come back and correct the typing in the inputs and outputs of the functions.
class LatentSearch(ABC):


    def __init__(self, execute_policy: Callable, model, task_cls, config = Config, search_method, collector = None, logger = None,):

        super().__init__()
        self.execute_policy = execute_policy
        self.model = model
        self.task_envs = [task_cls(index) for index in range(config.search_number_executions)]
        self.collector = collector
        self.logger = logger
        self.search_method = search_method


        self.number_of_iterations = config.search_number_iterations
        self.population_size = config.search_population_size
        self.elitism_rate = config.search_elitism_rate
        self.n_eite = int(config.search_elitism_rate * self.population_size)
        self.sigma_min = config.search_sigma_min
        self.sigma_rate = config.search_sigma_exponential_decay
        self.sigma_rate = config.search_sigma
        self.seed = config.search_seed
        self.restart_timeout = config.search_restart_timeout
        self.latent_size = config.model_hidden_size

        ### TODO: come back and jaxify everything here.


    def search_sigma_anneal(self):
        self.sigma = max(self.sigma)


    def init_search(self):

        population = self.init_population()
        
        self.counter_for_restart = 0
        self.num_evaluations = 0
        self.best_reward = float("-inf")
        self.best_policy = None
        self.converged = False
        self.prev_mean_elite_reward = float("-inf")

        return population

    def maybe_restart_population(self, mean_elite_reward, population):
        restarted_population = False

        if mean_elite_reward == self.prev_mean_elite_reward:
            self.counter_for_restart += 1
        else: 
            self.counter_for_restart = 0

        if self.counter_for_restart >= self.restart_timeout and self.restart_timeout > 0:
            restarted_population = True
            population = self.init_population()
            self.counter_for_restart = 0
            
            if self.logger:
                self.logger.log("Restarted Population")
        return population, restarted_population

    @abstractmethod
    def copy_policy(self, _policy):
        pass

    @partial(jax.jit, static_argnames = ("self",))
    def get_neighbors(self, new_elite_indices, elite_population, key):
        samples = elite_population[new_elite_indices]
        new_population = samples + self.sigma * jax.random.normal(key, shape=samples.shape, dtype = jnp.float32)
        return new_population

    @partial(jax.jit, static_argnames=("self",))
    def init_population(self, key):
        return jax.random.normal(key, (self.population_size, self.latent_size), dtype=jnp.float32)
        

    def execute_population(self, population, multi_processing: bool = False):
        
        if multi_processing:
            ready_for_multi_func = partial(self.execute_policy, task_envs = self.task_envs)
            with Pool() as pool:
                results = pool.map(ready_for_multi_func, population)
        else:
            results = [self.execute_policy(task_envs = self.task_envs, policy = _policy) for _policy in population]

        rewards = []
        for _policy, num_evals, reward in results:
            rewards.append(reward)
            self.num_evaluations += num_evals

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_policy = self.copy_policy(_policy)

                if self.collector:
                    self.collector.collect({
                        "time": None,
                        "num_evalutations": self.num_evaluations, 
                        "best_reward": self.best_reward
                    })

                if self.logger:
                    self.logger.log(get_logging_message(best_reward=self.best_reward, 
                                                        best_policy=self.best_policy, 
                                                        num_evaluations=self.num_evaluations))
                    
        return rewards
        
    def search(self):
        population = self.init_search()

        for iteration in range(self.number_of_iterations):
            rewards = self.execute_population(population)

            if self.converged:
                break

            _, best_indices =  top_k(rewards, self.n_eite)
            elite_population = population[best_indices]
            mean_elite_reward = jnp.mean(rewards[best_indices])

            if self.logger:
                ###TODO: please add the logging message here. 
                self.logger.log(None)

            population, restarted_population = self.maybe_restart_population(population=population)
            
            if not restarted_population:
                new_elite_indices = self.search_method.step(self.population_size, elite_population)

                population = self.get_neighbors(new_elite_indices, elite_population)

            self.prev_mean_elite_reward = mean_elite_reward
            self.search_sigma_anneal()

        if not self.converged:
            self.collector.collect({
                        "time": None,
                        "num_evalutations": self.num_evaluations, 
                        "best_reward": self.best_reward
                    })

        return self.best_policy, self.converged, self.num_evaluations




def execute_program(program_str: str, task_envs: List[Task], dsl: DSL):

    ### TODO: figure out how to get rid of this for loop
    program = dsl.parse_str_to_node(program_str)
    mean_reward = 0.0
    for task_env in task_envs:
        mean_reward += task_env.evaluate_program(program)
    num_evaluations = 1
    mean_reward /= len(task_envs)

    return program, num_evaluations, mean_reward

def execute_policy(z: jax.Array, task_envs):
    
    mean_reward = 0.0
    for task_env in task_envs:
        #### TODO: In order to change this, you need to get your neural networks straight
        pass





@partial(jax.jit, static_arganmes = ('reduce_to_mean',))
def CEM(population_size, elite_population, key, reduce_to_mean = False):
    if reduce_to_mean:
        new_indices = jnp.zeros(population_size)
        elite_population = jnp.mean(elite_population)
    else:
        ##TODO: make sure the size is input correctly for elite population
        new_indices = jax.random.choice(key, 
                                        elite_population.size(0), 
                                        shape = (population_size,), 
                                        replace=True)
        
    return new_indices

@jax.jit
def CEBS(population_size, n_elite):
    new_indices = jnp.repeat(
        jnp.arange(n_elite),
        int(population_size / n_elite)    
    )
    return new_indices

