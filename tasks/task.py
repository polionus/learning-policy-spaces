from __future__ import annotations
from abc import ABC, abstractmethod
import copy
from typing import Union
import numpy as np

from config import Config
from dsl.base import Program
from karel.world import WorldState


class Task(ABC):

    def __init__(self, seed: Union[None, int] = None):
        if seed is None:
            self.rng = np.random.RandomState(Config.env_seed)
        else:
            self.rng = np.random.RandomState(seed)
        self.env_height = Config.env_height()
        self.env_width = Config.env_width()
        self.initial_state = self.generate_state() ### I need to make sure that this works well with the worldstate change.
        self.reset_state()

    def set_state(self, state: WorldState):
        self.state = state

    def get_state(self) -> WorldState:
        return self.state

    def reset_state(self) -> None:
        # self.state = copy.deepcopy(self.initial_state)
        return self.initial_state
    
    @abstractmethod
    def generate_state(self) -> WorldState:
        pass

    @abstractmethod
    def get_reward(self, world_state: WorldState) -> tuple[bool, float]:
        pass

    def evaluate_policy(self, policy):
        self.reset_state()
        reward = 0.0
        for _ in range(Config.episode_length):
    
            policy_terminated = policy.execute_policy(self.state)
            terminated, instant_reward = self.get_reward(self.state)
            reward += instant_reward
            if terminated or self.state.is_crashed() or policy_terminated:
                break
        return reward

    ### We should now be able to understand 
    ### how the run generator receives a world state. 
    def evaluate_program(self, program: Program) -> float:
        self.reset_state()
        reward = 0.0
        for _ in program.run_generator(self.state):
            terminated, instant_reward = self.get_reward(self.state)
            reward += instant_reward
            if terminated or self.state.is_crashed():
                break
        return reward

    def trace_program(self, program: Program, image_name="trace.gif", max_steps=50):
        from PIL import Image

        self.reset_state()
        im = Image.fromarray(self.state.to_image())
        im_list = []
        for _ in program.run_generator(self.state):
            terminated, _ = self.get_reward(self.state)
            im_list.append(Image.fromarray(self.state.to_image()))
            if len(im_list) > max_steps or terminated or self.state.is_crashed():
                break
        im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)

    def trace_policy(self, policy, image_name="policy_trace.gif", max_steps=50):
        from PIL import Image

        self.reset_state()
        im = Image.fromarray(self.state.to_image())
        im_list = []
        for i in range(Config.episode_length):
            policy_terminated = policy.execute_policy(self.state)
            terminated, instant_reward = self.get_reward(self.state)
            im_list.append(Image.fromarray(self.state.to_image()))
            if terminated or self.state.is_crashed() or policy_terminated:
                break
        im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)
