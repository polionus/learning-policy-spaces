import numpy as np
from scipy import spatial
from karel.world import World, STATE_TABLE
from .task import Task


class CorridorRunner(Task):

    def generate_state(self):
        state = np.zeros(
            (self.env_height, self.env_width, len(STATE_TABLE)), dtype=bool
        )

        # Top and bottom walls
        state[0, :, 4] = True
        state[self.env_height - 1, :, 4] = True

        # Vertical side walls (optional, just for framing)
        state[:, 0, 4] = True
        state[:, self.env_width - 1, 4] = True

        # Define the valid row (e.g., middle of the grid)
        valid_row = self.env_height // 2
        self.valid_positions = [
            [valid_row, col] for col in range(1, self.env_width - 1)
        ]

        # Choose initial and marker positions in the valid row
        initial_col = self.rng.randint(1, self.env_width - 3)
        marker_col = self.rng.randint(initial_col + 1, self.env_width - 2)

        self.initial_position = [valid_row, initial_col]
        state[valid_row, initial_col, 1] = True  # Karel's position

        self.marker_position = [valid_row, marker_col]
        state[:, :, 5] = True
        state[valid_row, marker_col, 6] = True  # Marker presence
        state[valid_row, marker_col, 5] = False

        self.initial_distance = spatial.distance.cityblock(
            self.initial_position, self.marker_position
        )
        self.previous_distance = self.initial_distance

        return World(state)

    def reset_state(self) -> None:
        super().reset_state()
        self.previous_distance = self.initial_distance

    def get_reward(self, world_state: World):
        terminated = False
        reward = 0

        karel_pos = world_state.get_hero_loc()
        current_distance = spatial.distance.cityblock(
            [karel_pos[0], karel_pos[1]], self.marker_position
        )

        reward = (self.previous_distance - current_distance) / self.initial_distance

        if [karel_pos[0], karel_pos[1]] not in self.valid_positions:
            reward = -1
            terminated = True

        if karel_pos == tuple(self.marker_position):
            terminated = True

        self.previous_distance = current_distance

        return terminated, reward
