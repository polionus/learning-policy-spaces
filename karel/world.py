# Adapted from https://github.com/bunelr/GandRL_for_NPS/blob/master/karel/world.py
from __future__ import annotations
from typing import Tuple
from dataclasses import replace

import os
import jax.numpy as jnp
import jax
import equinox as eqx
import numpy as np




### BUG: Something is wrong about the hero_pos. It is not batched!
### BUG: The axes I selected to be batched over 

from config import Config

MAX_API_CALLS = 10000
MAX_MARKERS_PER_SQUARE = 10

STATE_TABLE = {
    0: "Karel facing North",
    1: "Karel facing East",
    2: "Karel facing South",
    3: "Karel facing West",
    4: "Wall",
    5: "0 marker",
    6: "1 marker",
    7: "2 markers",
    8: "3 markers",
    9: "4 markers",
    10: "5 markers",
    11: "6 markers",
    12: "7 markers",
    13: "8 markers",
    14: "9 markers",
    15: "10 markers",
}

ACTION_TABLE = {
    0: "Move",
    1: "Turn left",
    2: "Turn right",
    3: "Pick up a marker",
    4: "Put a marker",
}

CRASHABLE = Config.env_is_crashable
LEAPS_BEHAVIOR = Config.env_enable_leaps_behaviour


@jax.jit
def make_world_state(state: jax.Array):

    numAPICalls: int = jnp.int32(0)
    crashed: bool = jnp.bool(False)
    s = state.astype(bool)
    
    
    ### NOTE: The size input is necessary to make jax transformations possible (dynamic output sizes are not allowed.)
    x, y, d = jnp.where(s[:, :, :4] > 0, size = 4)
    hero_pos = jnp.array([x[0], y[0], d[0]])
    markers_grid = s[:, :, 5:].argmax(axis=2)
    terminated = jnp.bool(False)
    

    return WorldState(s = s, 
                      numAPICalls=numAPICalls,
                      crashed=crashed,
                      hero_pos=hero_pos,
                      markers_grid=markers_grid,
                      terminated=terminated)



class WorldState(eqx.Module):

    numAPICalls: int
    crashed: bool
    s: jax.Array
    hero_pos: jax.Array
    markers_grid: jax.Array
    terminated: bool


    # def __init__(self, s: jax.Array = None):
    #     if s is not None:
    #         self.s = s.astype(bool)
    #     x, y, d = jnp.where(self.s[:, :, :4] > 0)
    #     self.hero_pos = jnp.array([x[0], y[0], d[0]])
    #     self.markers_grid = self.s[:, :, 5:].argmax(axis=2)
    #     self.terminated = False

    def __eq__(self, other: WorldState) -> bool:
        if self.crashed != other.crashed:
            return False
        return (self.s == other.s).all()
    
    def __ne__(self, other: WorldState) -> bool:
        return not (self == other)



def hero_at_pos(world_state: WorldState, r: int, c: int) -> bool:
    row, col, _ = world_state.hero_pos
    return row == r and col == c


### Make this branchless when you come back.
def is_clear(world_state: WorldState, r: int, c: int) -> bool:
    
    conds = jnp.array([
        jnp.logical_or(r < 0, c < 0),
        jnp.logical_or(r >= world_state.s.shape[0], c >= world_state.s.shape[1]),
    ])

    case = jnp.select(conds, jnp.arange(len(conds)), default = len(conds))

    branches = (
        lambda _: False,
        lambda _: False,
        lambda _: jnp.logical_not(world_state.s[r, c, 4])
    )

    # jax.debug.breakpoint()
    
    return jax.lax.switch(case, branches, operand=None)
    

def front_is_clear(world_state: WorldState) -> Tuple[WorldState, bool]:
    
    r, c, d = world_state.hero_pos

    flag = ((d == 0) * is_clear(world_state, r - 1, c)
        + (d == 1) * is_clear(world_state, r, c + 1)
        + (d == 2) * is_clear(world_state, r + 1, c)
        + (d == 3) * is_clear(world_state, r, c - 1))


    def _on_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return world_state, False 

    def _on_not_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return replace(world_state, 
                                                                 numAPICalls = world_state.numAPICalls + 1, 
                                                                 crashed = world_state.numAPICalls + 1 > MAX_API_CALLS), flag

    return jax.lax.cond(world_state.crashed, _on_crashed, _on_not_crashed, world_state)


def left_is_clear(world_state: WorldState) -> Tuple[WorldState, bool]:

    r, c, d = world_state.hero_pos


    flag = ((d == 0) * is_clear(world_state, r, c - 1)
            + (d == 1) * is_clear(world_state, r - 1, c)
            + (d == 2) * is_clear(world_state, r, c + 1)
            + (d == 3) * is_clear(world_state, r + 1, c))

    def _on_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return world_state, False ### Instead of None, we return a dummy false, and not change the world state.

    def _on_not_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return replace(world_state, 
                                                                 numAPICalls = world_state.numAPICalls + 1, 
                                                                 crashed = world_state.numAPICalls + 1 > MAX_API_CALLS), flag

    return jax.lax.cond(world_state.crashed, _on_crashed, _on_not_crashed, world_state)



def right_is_clear(world_state: WorldState) -> Tuple[WorldState, bool]:
    
    r, c, d = world_state.hero_pos

    flag = ((d == 0) * is_clear(world_state, r, c + 1)
            + (d == 1) * is_clear(world_state, r + 1, c)
            + (d == 2) * is_clear(world_state, r, c - 1)
            + (d == 3) * is_clear(world_state, r - 1, c))

    def _on_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return world_state, False ### This none is probably prolematic. See where the -1 sentinel used to be None.

    def _on_not_crashed(world_state: WorldState) -> Tuple[WorldState, bool]: return replace(world_state, 
                                                                 numAPICalls = world_state.numAPICalls + 1, 
                                                                 crashed = world_state.numAPICalls + 1 > MAX_API_CALLS), flag

    return jax.lax.cond(world_state.crashed, _on_crashed, _on_not_crashed, world_state)


def markers_present(world_state: WorldState) -> Tuple[WorldState, bool]:
    
    r, c, _ = world_state.hero_pos
    new_world_state = replace(world_state, 
                              numAPICalls = world_state.numAPICalls + 1, 
                              crashed = world_state.numAPICalls + 1 > MAX_API_CALLS)

    
    return new_world_state, world_state.markers_grid[r, c] > 0

def pick_marker(world_state: WorldState) -> WorldState:

    r, c, _ = world_state.hero_pos
    num_marker = world_state.markers_grid[r, c]

    def _on_0(world_state: WorldState):
        crashed = jnp.where(CRASHABLE, True, world_state.crashed)
        s = world_state.s
        markers_grid = world_state.markers_grid
        return s, markers_grid, crashed
    
    def _on_not_zero(world_state: WorldState):
        s = world_state.s.at[r, c, 5 + num_marker].set(False).at[r, c, 4 + num_marker].set(True)
        markers_grid = world_state.markers_grid.at[r, c].set(world_state.markers_grid[r, c] - 1) 
        return s, markers_grid, world_state.crashed
    
    s, markers_grid, crashed = jax.lax.cond(num_marker == 0, _on_0, _on_not_zero, world_state)
    crashed = jnp.logical_or(crashed, world_state.numAPICalls + 1 > MAX_API_CALLS)

    return replace(world_state, 
                   crashed = crashed, 
                   markers_grid = markers_grid, 
                   s = s,
                   numAPICalls = world_state.numAPICalls + 1)


def put_marker(world_state: WorldState) -> WorldState:

    r, c, _ = world_state.hero_pos
    num_marker = world_state.markers_grid[r, c]

    def _on_max_marker(world_state: WorldState):
        crashed = jnp.where(CRASHABLE, True, world_state.crashed)
        s = world_state.s
        markers_grid = world_state.markers_grid
        return s, markers_grid, crashed
    
    def _on_not_max_marker(world_state: WorldState):
        s = world_state.s.at[r, c, 5 + num_marker].set(False).at[r, c, 6 + num_marker].set(True)
        markers_grid = world_state.markers_grid.at[r, c].set(world_state.markers_grid[r, c] + 1) 
        return s, markers_grid, world_state.crashed
    
    s, markers_grid, crashed = jax.lax.cond(num_marker == 0, _on_max_marker, _on_not_max_marker, world_state)
    crashed = jnp.logical_or(crashed, world_state.numAPICalls + 1 > MAX_API_CALLS)

    return replace(world_state, 
                   crashed = crashed, 
                   markers_grid = markers_grid, 
                   s = s,
                   numAPICalls = world_state.numAPICalls + 1)                

def move(world_state : WorldState) -> WorldState:
    
    r, c, d = world_state.hero_pos
    new_r = r
    new_c = c

    new_r = new_r + jnp.where(d == 0, -1, jnp.where(d == 2, +1, 0))
    new_c = new_c + jnp.where(d == 1, +1, jnp.where(d == 3, -1, 0))

    # sosis = replace(world_state, crashed = True)
    # hello = jnp.logical_and(jnp.logical_not(is_clear(world_state, new_r, new_c)), CRASHABLE)
    predicate = jnp.logical_and(jnp.logical_not(is_clear(world_state, new_r, new_c)), CRASHABLE)
    world_state = jax.lax.cond(predicate, 
                               lambda world_state: replace(world_state, crashed = (world_state.crashed + 1).astype(jnp.bool)), 
                               lambda world_state : world_state,
                               world_state)
    

    #if not is_clear(world_state, new_r, new_c) and CRASHABLE:
        #world_state = replace(world_state, crashed = True)
    def _condition_body(world_state, new_r, new_c):
        s = world_state.s
        s = s.at[r, c, d].set(False)
        s = s.at[new_r, new_c, d].set(True)
        hero_pos = jnp.array([new_r, new_c, d])
        world_state = replace(world_state, s = s, hero_pos = hero_pos)

        return world_state
    
    def _on_leaps_behavior(world_state):
        world_state = turn_left(world_state)
        world_state = turn_left(world_state)
        return world_state


    pred = jnp.logical_and(jnp.logical_not(world_state.crashed), is_clear(world_state, new_r, new_c))

    world_state = jax.lax.cond(pred, 
                               _condition_body, 
                               lambda world_state, new_r, new_c : jax.lax.cond(LEAPS_BEHAVIOR, _on_leaps_behavior, lambda world_state : world_state, world_state),
                               world_state, new_r, new_c)
    
    return replace(world_state, 
                   numAPICalls = world_state.numAPICalls + 1, 
                   crashed = world_state.numAPICalls + 1 > MAX_API_CALLS)



def turn_left(world_state: WorldState) -> WorldState:
    
    def _condition_body(world_state: WorldState):
        r, c, d = world_state.hero_pos
        new_d = (d - 1) % 4
        s = world_state.s
        s = s.at[r, c, d].set(False)
        s = s.at[r, c, new_d].set(True)
        hero_pos = jnp.array([r, c, new_d])
        return replace(world_state, hero_pos = hero_pos, s = s)

    world_state = jax.lax.cond(world_state.crashed, lambda world_state : world_state, _condition_body, world_state)

    return replace(world_state, 
                   numAPICalls = world_state.numAPICalls + 1, 
                   crashed = world_state.numAPICalls + 1 > MAX_API_CALLS)


def turn_right(world_state: WorldState) -> WorldState:
   
    
    def _condition_body(world_state: WorldState):
        r, c, d = world_state.hero_pos
        new_d = (d + 1) % 4
        s = world_state.s
        s = s.at[r, c, d].set(False)
        s = s.at[r, c, new_d].set(True)
        hero_pos = jnp.array([r, c, new_d])
        return replace(world_state, hero_pos = hero_pos, s = s)
    
    
    world_state = jax.lax.cond(world_state.crashed,lambda world_state: world_state,  _condition_body, world_state)
    


    return replace(world_state, 
            crashed = world_state.numAPICalls + 1 > MAX_API_CALLS,
            numAPICalls = world_state.numAPICalls + 1)


def run_action(world_state: WorldState, action: int) -> WorldState:


    def _on_default(world_state: WorldState) -> WorldState: return world_state

    # jax.debug.breakpoint()
    branches = (
        move, 
        turn_left,
        turn_right, 
        pick_marker,
        put_marker,
        _on_default
    )

    case = jnp.asarray(action, jnp.int32).squeeze()

    # jax.debug.breakpoint()

    return jax.lax.switch(case, branches, world_state) 

### NOTE: There is a self.assets argument, that will make the environment into an image. This function should exist outside of the world class and world state, 
### and accept the world state as input.

### Do these need to be in the same class?
### Let's not do this.

def equal_makers(one: WorldState, other: WorldState) -> bool:
            return (one.s[:, :, 5:] == other.s[:, :, 5:]).all()


def to_image(world_state: WorldState) -> jnp.ndarray:
        assets: dict[str, jax.Array] = {}
        rows = world_state.s.shape[0]
        cols = world_state.s.shape[1]

        
        grid_size = 100
        if len(assets) == 0:
            from PIL import Image

            files = [
                "agent_0",
                "agent_1",
                "agent_2",
                "agent_3",
                "blank",
                "marker",
                "wall",
            ]
            for f in files:
                assets[f] = np.array(
                    Image.open(os.path.join("assets", f"{f}.PNG"))
                )

        img = np.ones((rows * grid_size, cols * grid_size))
        hero_r, hero_c, hero_d = world_state.hero_pos
        for r in range(rows):
            for c in range(cols):
                if world_state.s[r][c][4] == 1:
                    asset = assets["wall"]
                elif r == hero_r and c == hero_c:
                    if np.sum(world_state.s[r, c, 6:]) > 0:
                        asset = np.minimum(
                            assets[f"agent_{hero_d}"], assets["marker"]
                        )
                    else:
                        asset = assets[f"agent_{hero_d}"]
                elif np.sum(world_state.s[r, c, 6:]) > 0:
                    asset = assets["marker"]
                else:
                    asset = assets["blank"]
                img[
                    (r) * grid_size : (r + 1) * grid_size,
                    c * grid_size : (c + 1) * grid_size,
                ] = asset

        return img


def run_and_trace(world_state, program, image_name="trace.gif"):
    from PIL import Image

    im = Image.fromarray(to_image(world_state))
    im_list = []

    step = 0
    for _ in program.run_generator(world_state):
        
        ### This will only trace the program for 300 steps. 
        if step > 300:
            break
        im_list.append(Image.fromarray(to_image(world_state)))
        step += 1
    im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)
        

### TODO: Come back and fix these.
class World:
    

    def __init__(self, world_state: WorldState):
        self._state = world_state
        
    def __getattribute__(self, name):
        return getattr(self._state, name)
    ### perhaps I need to simply use the s and just use shape[0].
    @property
    def rows(self):
        return self.s.shape[0]

    @property
    def cols(self):
        return self.s.shape[1]

    ### TODO: Come back to this later, this looks complicated.
    @classmethod
    def from_json(cls, json_object):
        rows = json_object["rows"]
        cols = json_object["cols"]
        s = jnp.zeros((rows, cols, 16), dtype=bool)
        hero = json_object["hero"].split(":")
        heroRow = int(hero[0])
        heroCol = int(hero[1])
        heroDir = World.get_dir_number(hero[2])
        s[rows - heroRow - 1, heroCol, heroDir] = True

        if json_object["blocked"] != "":
            for coord in json_object["blocked"].split(" "):
                coord_split = coord.split(":")
                r = int(coord_split[0])
                c = int(coord_split[1])
                s[rows - r - 1, c, 4] = (
                    True  # For some reason, the original program uses rows - r - 1
                )

        s[:, :, 5] = True
        if json_object["markers"] != "":
            for coord in json_object["markers"].split(" "):
                coord_split = coord.split(":")
                r = int(coord_split[0])
                c = int(coord_split[1])
                n = int(coord_split[2])
                s[rows - r - 1, c, n + 5] = True
                s[rows - r - 1, c, 5] = False

        return cls(s)

    @classmethod
    def from_string(cls, worldStr: str):
        lines = worldStr.replace("|", "").split("\n")
        # lines.reverse()
        rows = len(lines)
        cols = len(lines[0])
        s = jnp.zeros((rows, cols, 16), dtype=bool)
        for r in range(rows):
            for c in range(cols):
                if lines[r][c] == "*":
                    s[r][c][4] = True
                elif lines[r][c] == "M":  # TODO: could also be a number
                    s[r][c][6] = True
                else:
                    s[r][c][5] = True

                if lines[r][c] == "^":
                    s[r][c][0] = True
                elif lines[r][c] == ">":
                    s[r][c][1] = True
                elif lines[r][c] == "v":
                    s[r][c][2] = True
                elif lines[r][c] == "<":
                    s[r][c][3] = True
        return cls(s)

    def to_json(self) -> dict:
        obj = {}

        obj["rows"] = self.rows
        obj["cols"] = self.cols
        if self.crashed:
            obj["crashed"] = True
            return obj

        obj["crashed"] = False

        markers = []
        blocked = []
        hero = []
        for r in range(self.rows - 1, -1, -1):
            for c in range(0, self.cols):
                if self.s[r][c][4]:
                    blocked.append("{0}:{1}".format(r, c))
                if self.hero_at_pos(r, c):
                    hero.append("{0}:{1}:{2}".format(r, c, self.heroDir))
                if jnp.sum(self.s[r, c, 6:]) > 0:
                    markers.append("{0}:{1}:{2}".format(r, c, jnp.sum(self.s[r, c, 6:])))

        obj["markers"] = " ".join(markers)
        obj["blocked"] = " ".join(blocked)
        obj["hero"] = " ".join(hero)

        return obj

    def to_string(self) -> str:
        worldStr = ""
        # worldStr += str(self.heroRow) + ', ' + str(self.heroCol) + '\n'
        if self.crashed:
            worldStr += "CRASHED\n"
        hero_r, hero_c, hero_d = self.get_hero_loc()
        for r in range(0, self.rows):
            rowStr = "|"
            for c in range(0, self.cols):
                if self.s[r][c][4] == 1:
                    rowStr += "*"
                elif r == hero_r and c == hero_c:
                    rowStr += self.get_hero_char(hero_d)
                elif jnp.sum(self.s[r, c, 6:]) > 0:
                    num_marker = self.s[r, c, 5:].argmax()
                    if num_marker > 9:
                        rowStr += "M"
                    else:
                        rowStr += str(num_marker)
                else:
                    rowStr += " "
            worldStr += rowStr + "|"
            if r != self.rows - 1:
                worldStr += "\n"
        return worldStr

    def to_image(self) -> jnp.ndarray:
        grid_size = 100
        if len(self.assets) == 0:
            from PIL import Image

            files = [
                "agent_0",
                "agent_1",
                "agent_2",
                "agent_3",
                "blank",
                "marker",
                "wall",
            ]
            for f in files:
                self.assets[f] = jnp.array(
                    Image.open(os.path.join("assets", f"{f}.PNG"))
                )

        img = jnp.ones((self.rows * grid_size, self.cols * grid_size))
        hero_r, hero_c, hero_d = self.hero_pos
        for r in range(self.rows):
            for c in range(self.cols):
                if self.s[r][c][4] == 1:
                    asset = self.assets["wall"]
                elif r == hero_r and c == hero_c:
                    if jnp.sum(self.s[r, c, 6:]) > 0:
                        asset = jnp.minimum(
                            self.assets[f"agent_{hero_d}"], self.assets["marker"]
                        )
                    else:
                        asset = self.assets[f"agent_{hero_d}"]
                elif jnp.sum(self.s[r, c, 6:]) > 0:
                    asset = self.assets["marker"]
                else:
                    asset = self.assets["blank"]
                img[
                    (r) * grid_size : (r + 1) * grid_size,
                    c * grid_size : (c + 1) * grid_size,
                ] = asset

        return img

    def run_and_trace(self, program, image_name="trace.gif"):
        from PIL import Image

        im = Image.fromarray(self.to_image())
        im_list = []
        for _ in program.run_generator(self):
            im_list.append(Image.fromarray(self.to_image()))
        im.save(image_name, save_all=True, append_images=im_list, duration=75, loop=0)


if __name__ == "__main__":
    world = World.from_string(
        "|  |\n"
        + "| *|\n"
        + "|  |\n"
        + "|  |\n"
        + "| *|\n"
        + "|  |\n"
        + "|  |\n"
        + "| *|\n"
        + "|  |\n"
        + "|^ |"
    )

    print(world.to_string())

    if world.right_is_clear():
        world.turn_right()
        world.move()
        world.put_marker()
        world.turn_left()
        world.turn_left()
        world.move()
        world.turn_right()
    while world.front_is_clear():
        world.move()
        if world.right_is_clear():
            world.turn_right()
            world.move()
            world.put_marker()
            world.turn_left()
            world.turn_left()
            world.move()
            world.turn_right()

    print(world.to_string())
