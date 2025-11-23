from __future__ import annotations
from typing import Callable
import time
import torch
import jax
import jax.numpy as jnp
import search_tools as st
from utils.program import execute_program
from dsl import DSL
from multiprocessing import Pool
from functools import partial
from logger.logger import logger

from vae.models.base_vae import BaseVAE
from config import SearchConfig, TrainConfig, Config
from tasks.task import Task
from aim import Run


#TODO: Change this to a eqx.Module for jaxification.
def search(n_elite: int, number_iterations: int, search_method: Callable, key: jax.Array) -> tuple[str, bool, int]:
        """Main search method. Searches for a program using the specified DSL that yields the
        highest reward at the specified task.

        Returns:
            tuple[str, bool]: Best program in string format and a boolean value indicating
            if the search has converged.
        """

        ### initialize search
        search_state = st.init_search_state()
       
        for iteration in range(1, number_iterations + 1):
            rewards = st.execute_population(search_state.population)

            if search_state.converged:
                break

            _, topk_indices = jax.lax.top_k(rewards, n_elite)
            elite_population = search_state.population[topk_indices]
            mean_elite_reward = jnp.mean(rewards[topk_indices])

            logger.info(
                "Latent Search",
                f"Iteration {iteration} mean elite reward: {mean_elite_reward}",
            )
            
            search_state = st.maybe_continue_population(mean_elite_reward, search_state, elite_population, key)
            search_state.prev_mean_elite_reward = mean_elite_reward
            

        if not search_state.converged:
            ## TODO: Gotta track the best reward here.
            pass
        return search_state

class LatentSearch:
    """Implements the CEM method from LEAPS paper."""

    def __init__(self, model: BaseVAE, task_cls: type[Task], dsl: DSL, run: Run):
        self.model = model
        
        self.run = run
        self.dsl = dsl
        self.search_method = st.get_search_method()   
        self.model_hidden_size = TrainConfig.hidden_size

        ### Understand and correct this
        self.task_envs = [task_cls(i) for i in range(SearchConfig.number_executions)]
       


