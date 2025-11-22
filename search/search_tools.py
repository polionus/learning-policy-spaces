import jax.numpy as jnp
from logger.logger import logger
from dsl import DSL
from config import SearchConfig
from functools import partial
from typing import List
from utils.program import execute_program 
from multiprocessing import Pool
import equinox as eqx
import jax


def get_neighbors(sigma: float, new_indices:jax.Array, elite_population: jax.Array, key: jax.Array):
    mu = elite_population[new_indices]
    population = jax.random.normal(key, shape= mu.shape) + sigma
    return population

def search_sigma_anneal(sigma: float, sigma_min: float, sigma_rate: float):
    sigma = max(sigma_min, sigma * sigma_rate)

def init_population(latent_size: int, population_size: int, key: jax.Array) -> jax.Array:
    return jax.random.normal(key, shape = (population_size, latent_size))


def execute_population(
        model: eqx.Module, population: jax.Array, dsl: DSL, task_envs: List
    ) -> tuple[list[str], jax.Array, int]:
        """Runs the given population in the environment and returns a list of mean rewards, after
        `Config.search_number_executions` executions.

        Args:
            population (torch.Tensor): Current population as a tensor.

        Returns:
            tuple[list[str], int, torch.Tensor]: List of programs as strings, list of mean rewards
            as tensor and number of evaluations as int.
        """
        programs_tokens = model.decode_vector(population)

        ## At this step, the programs are turned into string (from tokens), and in order to be executed, they need to be turned into nodes. 
        programs_str = [
            dsl.parse_int_to_str(prog_tokens) for prog_tokens in programs_tokens
        ]
        
        #TODO: Learn to use/modify this to better use the multi-processing in the code.
        ### Use vmap here?

        if SearchConfig.multiprocessing:
            with Pool() as pool:
                fn = partial(execute_program, task_envs=task_envs, dsl=dsl)
                results = pool.map(fn, programs_str)
        else:
            results = [
                execute_program(p, task_envs, dsl) for p in programs_str
            ]
        return results


def process_population_results(results, dsl: DSL, best_reward: float, num_evaluations: int):
    rewards = []
    converged = False
    for p, num_eval, r in results:
        program_str = dsl.parse_node_to_str(p)
        rewards.append(r)
        num_evaluations += num_eval

        #Since the best reward is kept from previous runs, the best program is not changed between populations, unless a program that has even better return is found.
        if r > best_reward:
            best_reward = r
            best_program = program_str
            logger.info(
                    "Latent Search", f"New best reward: {best_reward}"
            )
            logger.info(
                "Latent Search", f"New best program: {best_program}"
            )
            logger.info(
                "Latent Search", f"Number of evaluations: {num_evaluations}"
            )

            ## For now I am not collecting the data. We will get to that with aim later.
            #collector.collect({'time':t, 'num_evaluations':num_evaluations, 'best_reward':best_reward})
            
        if best_reward >= 1.0:
            converged = True
            break

    return jnp.array(rewards), best_reward, num_evaluations, best_program, converged