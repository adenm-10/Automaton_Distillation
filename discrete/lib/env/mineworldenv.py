import functools
from collections import Counter
from random import Random
from typing import Tuple, TypeVar, Union, List, Dict, Collection
import os
import copy
from copy import deepcopy

import numpy as np
from gym import spaces
from math import sin, cos, sqrt, pi, exp

from discrete.lib.env.gridenv import GridEnv
from discrete.lib.env.saveloadenv import SaveLoadEnv
from discrete.lib.env.util import element_add

bounding_persist = False
bounding_dist = 7

try:
    bounding_persist = os.environ.get("bounding_persist") == 'True'
    bounding_dist = int(os.environ.get("bounding_dist"))
except Exception as e:
    print(f"Exception:\n\t{e}")
    bounding_persist = False
    bounding_dist = 7

print(f"Bounding Persist: {type(bounding_persist)}, {bounding_persist}")
print(f"Bounding Distance: {type(bounding_dist)}, {bounding_dist}")

Goal_reached_Dist = 0.4

COLLISION_DIST = 0.35
TIME_DELTA = 0.1
hit_wall_reward = -0.1

class MineWorldTileType:
    """A single special tile in the mine world"""

    def __init__(self, consumable: bool, inventory_modifier: Counter, action_name: str, grid_letter: str,
                 wall: bool = False, reward: int = 0, terminal: bool = False, inventory_requirements: Counter = None,
                 movement_requirements: Counter = None):
        """
        :param consumable: Does this tile disappear after being activated
        :param inventory_modifier: How does this modify the inventory (e.g. wood -2, desk +1)
        :param action_name: What atomic proposition should be true the round that this tile is activated
        :param grid_letter: What letter should be displayed on the grid
        """
        self.consumable = consumable
        self.inventory = inventory_modifier
        self.action_name = action_name
        self.grid_letter = grid_letter
        self.wall = wall
        self.reward = reward
        self.terminal = terminal
        self.inventory_requirements = inventory_requirements or Counter()
        self.movement_requirements = movement_requirements or Counter()

    def apply_inventory(self, prev_inventory: Counter):
        """
        Get the new inventory of the player after interacting with this tile, or errors if the player is unable to
        interact with the tile
        :param prev_inventory: The current inventory of the player
        """

        # Apply all the inventory changes and make sure that no item is negative
        new_inv = prev_inventory.copy()
        new_inv.update(self.inventory)
        if any([(new_inv[i] < 0) for i in new_inv]):
            raise ValueError()
        else:
            return new_inv

    def meets_requirements(self, current_inventory: Counter):
        inv_requirements_temp = current_inventory.copy()
        inv_non_neg_temp = current_inventory.copy()

        inv_requirements_temp.subtract(self.inventory_requirements)
        inv_non_neg_temp.update(self.inventory)

        requirements_ok = not any([(inv_requirements_temp[i] < 0) for i in inv_requirements_temp])
        inv_non_neg_ok = not any([(inv_non_neg_temp[i] < 0) for i in inv_non_neg_temp])

        return requirements_ok and inv_non_neg_ok
    
    def move_requirements(self, current_inventory: Counter):
        inv_requirements_temp = current_inventory.copy()
        inv_requirements_temp.subtract(self.movement_requirements)

        requirements_ok = not any([(inv_requirements_temp[i] < 0) for i in inv_requirements_temp])

        return requirements_ok

    @staticmethod
    def from_dict(dict):
        wall = dict.get("wall", False)
        reward = dict.get("reward", 0)
        terminal = dict.get("terminal", False)
        inventory_requirements = Counter(dict.get("inventory_requirements", {}))
        return MineWorldTileType(consumable=dict["consumable"], inventory_modifier=Counter(dict["inventory_modifier"]),
                                 action_name=dict["action_name"], grid_letter=dict["grid_letter"], wall=wall,
                                 reward=reward, terminal=terminal, inventory_requirements=inventory_requirements)


T = TypeVar("T")
MaybeRand = Union[T, str]


class TilePlacement:
    def __init__(self, tile: MineWorldTileType, fixed_placements: Collection[Tuple[int, int]] = tuple(),
                 random_placements: int = 0):
        self.tile = tile
        self.fixed_placements = fixed_placements
        self.random_placements = random_placements

    @staticmethod
    def from_dict(dict):
        tile = MineWorldTileType.from_dict(dict["tile"])
        fixed_raw = dict.get("fixed_placements", [])
        fixed_placements = [tuple(coord) for coord in fixed_raw]
        random_placements = dict.get("random_placements", 0)
        return TilePlacement(tile=tile,
                             fixed_placements=fixed_placements,
                             random_placements=random_placements)


class InventoryItemConfig:
    def __init__(self, name: str, default_quantity: int, capacity: int):
        """
        :param name: Name of the item, like wood or iron
        :param default_quantity: How many of these items to start with
        :param capacity: Maximum amount of this item the agent can hold. Also used for scaling of NN inputs.
        """
        self.name = name
        self.default_quantity = default_quantity
        self.capacity = capacity

    @staticmethod
    def from_dict(dict):
        return InventoryItemConfig(**dict)


class MineWorldConfig:
    def __init__(self, shape: Tuple[int, int], initial_position: Union[Tuple[int, int], None],
                 placements: List[TilePlacement], inventory: List[InventoryItemConfig], tile_shape: Tuple[int, int]=None):
        self.placements = placements
        self.shape = shape
        self.initial_position = initial_position
        self.inventory = inventory
        self.tile_shape = tile_shape

    @staticmethod
    def from_dict(dict):
        shape = tuple(dict["shape"])
        ip = dict["initial_position"]
        initial_position = ip if ip is None else tuple(ip)
        placement = [TilePlacement.from_dict(i) for i in dict["placements"]]
        inventory = list(map(InventoryItemConfig.from_dict, dict["inventory"]))

        return MineWorldConfig(shape=shape, initial_position=initial_position, placements=placement,
                               inventory=inventory)


def n_hot_grid(shape: Tuple[int, int], grid_positions: Union[None, List[Tuple[float, float]]], grid=None):
    if grid is None:
        grid = np.zeros(shape, dtype=np.int8)

    if grid_positions is None:
        grid_positions = []

    for pos in grid_positions:
        # print(grid)
        grid[pos] = 1

    return grid


def const_plane(shape, val):
    result = np.full(shape, val)
    return result

# mu = b, sigma = a, x = distance from or time steps after
def mod_normal_distribution(x, mu=1, height=1, width=1):
    mod_sigma = (width - mu) / (2 * np.sqrt(2 * np.log(2)))
    exponent = -0.5 * ((x - mu) / (mod_sigma))**2
    return height * exp(exponent)


@functools.lru_cache(16384)
def obs_rewrite(shape, obs):
    position, tile_locs, inventories = obs
    # Convert to float?
    position_tile_layers = tuple(n_hot_grid(shape, layer) for layer in ((position,), *tile_locs))
    inventory_layers = tuple(np.full(shape, layer, dtype=np.int8) for layer in inventories)
    return np.stack((*position_tile_layers, *inventory_layers), axis=0)

@functools.lru_cache(16384)
def obs_rewrite_cont(obs):
    position, tile_locs, inventories = obs

    position = list(position)
    temp_tiles = []
    for i, tile in enumerate(tile_locs):
        try:
            temp_tiles.append(list(list(tile)[0]))
        except:
            temp_tiles.append([0,0])
    # tile_locs = [list(list(tile)[0]) for tile in tile_locs if list(tile)]
    tile_locs = [element for sublist in temp_tiles for element in sublist]
    # tile_locs = [temp_tiles]
    inventories = [inv_num for inv_num in inventories]
    inventories_one_hot = [0, 0] * len(inventories)
    for i in range(len(inventories)):
        inventories_one_hot[2 * i + inventories[i]] = 1

    # print(f"position: {position}")
    # print(f"tile locs: {tile_locs}")
    # print(f"inventories: {inventories_one_hot}")

    return_val = np.concatenate((position, tile_locs, inventories_one_hot), axis=0) 
    # print(f"return val: {return_val}")
    # assert False
    return return_val

class MineWorldEnvContinuous(GridEnv, SaveLoadEnv):


    """
    MineWorld: The continuous environment for the minecraft environment
    """

    @staticmethod
    def from_dict(dict):
        
        return MineWorldEnv(MineWorldConfig.from_dict(dict))
    
    def __init__(self,config:MineWorldConfig,*args,**kwargs):
        # super().__init__(shape=config.shape, *args, **kwargs)

        self.action_space = spaces.Box(low=np.array([-1,-1], dtype=np.float32),
                                           high=np.array([1,1],dtype=np.float32)) # positional change in the XY direction

        num_special_tiles = sum(tile_type.random_placements for tile_type in config.placements)
        num_special_tiles += sum(len(tile_type.fixed_placements) for tile_type in config.placements)
        # print(f"special tiles, position, len inv: {num_special_tiles}, 2, {len(config.inventory)}")

        
        self.observation_space = spaces.Box(low=0,
                                                high=1, shape=(num_special_tiles + 2 + len(config.inventory),), dtype=np.float32) # gym.spaces.Box(0.0, 1.0, (self.lidar_num_bins,), dtype=np.float32)
        self.metadata = {'render.modes': ['human']}
        self.reward_range = (0, 100)

        self.config = config

        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
        self.rand = Random()

        self.done = True
        self.prev_position: Tuple [float,float] = (0.0,0.0)
        self.new_position: Tuple [float,float] = (0.0,0.0)
        self.special_tiles: Dict[Tuple[float,float], MineWorldTileType] = dict()
        self.inventory = Counter()
        self.angle = 0
        self.tile_type = []

        self.range = self.config.shape[1]

        #variables

        self.dt = 0.05
        self.vx = 0
        self.vy = 0
        self.g = -0.05
  
    
    def step(self,action: np.ndarray):

        action = np.array(action)

        # print(f"action here {action}")

        # print (f"this is the action {action}")
       
        assert action.shape == (2,)
        assert not self.done

        action_names = set()


        self.prev_position = self.new_position

        dx = action[0]
        dy = action[1]
        
        # self.vx = self.vx + self.dt*dx
        # self.vy = self.vy + self.dt*(dy + self.g)

        new_posx = self.new_position[0] + 3.0*dx
        new_posy = self.new_position[1] + 3.0*dy

        self.new_position = (new_posx, new_posy)

        # print(f"the new position is {self.new_position}")

        # print (f"the old me {self.prev_position}")
        # print (f"the new mw {self.new_position}")

        reward = 0

        if (self.new_position[0] >= 0 and self.new_position[0] < self.config.shape[0] and self.new_position[1] >= 0 and self.new_position[1] < self.config.shape[1]):

            self.forward = True
        else:
            self.forward = False
            # print(f"the previous is {self.new_position}")
            self.new_position = self.prev_position
            # print(f"the new is {self.new_position}")
            reward += hit_wall_reward

        
        if self.forward:

            for special_tile in self.special_tiles:

                this_tile: MineWorldTileType = self.special_tiles[special_tile]

                if this_tile.meets_requirements(self.inventory):

                    distance = np.linalg.norm([self.new_position[0] - special_tile[0], self.new_position[1] - special_tile[1]])
                            
                    # dx = self.new_position[0] - special_tile[0]
                    # dy = self.new_position[1] - special_tile[1]
            
                    if distance < Goal_reached_Dist:

                        new_inv = this_tile.apply_inventory(self.inventory)

                        for inv_config in self.config.inventory:
                            if new_inv[inv_config.name] > inv_config.capacity:
                                new_inv[inv_config.name] = inv_config.capacity

                        self.inventory = new_inv
                        action_names.add(this_tile.action_name)

                        if this_tile.consumable:
                            del self.special_tiles[special_tile]

                        if this_tile.terminal:
                            self.done = True

                        reward += this_tile.reward

                        # print (f"here's the main reward {reward} for {action_names}")
                        # print(self.done)
                        break


        #reward-=0.01

        # print (reward)
        info = {
            'tile_action_names': action_names,
            'inventory': self.inventory.copy(),
            'position': self.new_position
            # 'position': self.tile_position
        }

        obs = self.obs()
        # print(info)

        # print (obs)

        return obs, reward, self.done, info # because it needed truncated


    def dist_xy(self,position):

        ''' Return the distance from the robot to an XY position '''

        dist_dict = {placement: (0.0) for placement in (self.tile_type)}

        robot_pos = np.array([position[0], position[1]])

        for special_tile in self.special_tiles:

            if special_tile in self.tile_type:

                dist = np.linalg.norm([special_tile[0] - robot_pos[0], special_tile[1] - robot_pos[1]])
                ob = dist
                dist_dict[special_tile] = ob

        all_dists = np.array([dist for dist in dist_dict.values()])
        # print(all_dists)

        return all_dists

    def obs(self):

        ''' Return the observation of our agent '''

        obz = np.array(([self.new_position[0], self.new_position[1]]))/self.range

        obsy = self.dist_xy(self.new_position).flatten()
        obsy = obsy/self.range
        inventories = tuple(self.inventory[inv_config.name] for inv_config in self.config.inventory)
        # print(f"special tiles, position, len inv: {len(obz)}, {len(obsy)}, {len(inventories)}")
        # assert False
        # assert self.observation_space.contains(obsy)
        ret = np.concatenate((obz,obsy, inventories))
        # print(f"Observation: {ret}")
        return ret

    def seed(self, seed=None):
        self.rand.seed(seed)
        
    def reset(self):

        self.done = False

        self.prev_position = self.config.initial_position
        if not self.prev_position:
            position = (self.config.shape[0] * self.rand.random(), self.config.shape[1] * self.rand.random())

            self.prev_position = position
            self.new_position = position

        self.inventory = self.default_inventory.copy()
        self.special_tiles = self._get_tile_positioning()
        self.tile_type = copy.deepcopy(self.special_tiles)
    
        return self.obs()


    def _get_tile_positioning(self) -> Dict[Tuple[float, float], MineWorldTileType]:

        tiles = {}

        for tile_type in self.config.placements:
            for fixed in tile_type.fixed_placements:
                tiles[fixed] = tile_type.tile

        # noinspection PyTypeChecker
        all_spaces = set(np.ndindex(self.config.shape))
        open_spaces = all_spaces.difference(tiles.keys())
       
        if (0, 0) in open_spaces:
            open_spaces.remove((0, 0)) #remove some other points like (0,7) (7,0),(7,7)

        for tile_type in self.config.placements:
            tile, num_placements = tile_type.tile, tile_type.random_placements
            
            spaces = self.rand.sample(sorted(open_spaces), num_placements) #used sorted to handle sequence in the data
            open_spaces.difference_update(spaces)
            # print (spaces)

            for space in spaces:
                tiles[space] = tile

        return tiles

    
    def save_state(self):
        return self.position, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.position, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()

    @property
    def unwrapped(self):
        return self
        
    # Probably wont work, doesnt translate well to continuous
    def render(self, mode='human'):
        def render_func(x, y):
            agent_str = "A" if self.tile_position == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
            return agent_str + tile_str, False, False

        print(self._render(render_func, 2), end="")
        print(dict(self.inventory))

    def save_state(self):
        return self.true_position, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.true_position, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()


class MineWorldEnv(GridEnv, SaveLoadEnv):
    """A basic minecraft-like environment, with a global view of the state space"""
    @staticmethod
    def from_dict(dict):
        return MineWorldEnv(MineWorldConfig.from_dict(dict))

    def __init__(self, config: MineWorldConfig, *args, **kwargs):
        super().__init__(shape=config.shape, *args, **kwargs)

        self.action_space = spaces.Discrete(6) # Input can only be 0, 1, 2, 3, 4, 5
        self.observation_space = spaces.Box(0, 1,
                                            shape=(1 + len(config.placements) + len(config.inventory), *config.shape),
                                            dtype=np.float32)
        self.config = config
        self.default_inventory = Counter(
            {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
        self.rand = Random()

        """
        Up: 0,
        Right: 1,
        Down: 2,
        Left: 3,
        No-op: 4,
        Tile action: 5"""

        self.done = True
        self.position: Tuple[int, int] = (0, 0)
        self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
        self.inventory = Counter()

    def step(self, action: int):
        assert self.action_space.contains(action)
        assert not self.done

        action_names = set()

        reward = 0

        if action < 5:
            # Movement or no-op
            action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]

            new_place = element_add(self.position, action_offsets[action])
            # print(f"Action: {action}, Action Offsets: {action_offsets[action]}")

            can_move = self._in_bounds(new_place)

            if new_place in self.special_tiles:
                tile = self.special_tiles[new_place]
                if tile.wall or not tile.move_requirements(self.inventory):
                    can_move = False

            if can_move:
                self.position = new_place
        else:
            if self.position in self.special_tiles:
                this_tile: MineWorldTileType = self.special_tiles[self.position]
                if this_tile.meets_requirements(self.inventory):
                    new_inv = this_tile.apply_inventory(self.inventory)
                    for inv_config in self.config.inventory:
                        if new_inv[inv_config.name] > inv_config.capacity:
                            new_inv[inv_config.name] = inv_config.capacity
                    self.inventory = new_inv
                    action_names.add(this_tile.action_name)
                    if this_tile.consumable:
                        del self.special_tiles[self.position]
                    if this_tile.terminal:
                        self.done = True
                    reward += this_tile.reward

        # print(f"Position: {self.position}")

        info = {
            'tile_action_names': action_names,
            'inventory': self.inventory.copy(),
            'position': self.position
        }

        obs = obs_rewrite(self.shape, self._get_observation())
        # print(f"Discrete Observation: {obs}")
        return obs, reward, self.done, info

    def seed(self, seed=None):
        self.rand.seed(seed)

    def reset(self):
        self.done = False
        self.position = self.config.initial_position
        if not self.position:
            self.position = self.rand.randrange(0, self.shape[0]), self.rand.randrange(0, self.shape[1])
        self.inventory = self.default_inventory.copy()
        self.special_tiles = self._get_tile_positioning()

        return obs_rewrite(self.shape, self._get_observation())

    def _get_tile_positioning(self) -> Dict[Tuple[int, int], MineWorldTileType]:

        tiles = {}

        for tile_type in self.config.placements:
            for fixed in tile_type.fixed_placements:
                tiles[fixed] = tile_type.tile

        # noinspection PyTypeChecker
        all_spaces = set(np.ndindex(self.config.shape))
        open_spaces = all_spaces.difference(tiles.keys())


        if (0, 0) in open_spaces:
            open_spaces.remove((0, 0))

        for tile_type in self.config.placements:
            tile, num_placements = tile_type.tile, tile_type.random_placements
            spaces = self.rand.sample(open_spaces, num_placements)
            open_spaces.difference_update(spaces)

            for space in spaces:
                tiles[space] = tile

        return tiles

    def _get_observation(self):

        tiles = tuple(
            frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
            placement in self.config.placements)

        inv = tuple(self.inventory[inv_config.name] for inv_config in self.config.inventory)

        return (
            self.position,
            tiles,
            inv
        )

    def render(self, mode='human'):
        def render_func(x, y):
            agent_str = "A" if self.position == (x, y) else " "
            tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
            return agent_str + tile_str, False, False

        print(self._render(render_func, 2), end="")
        print(dict(self.inventory))

    def save_state(self):
        return self.position, self.done, self.special_tiles.copy(), self.inventory.copy()

    def load_state(self, state):
        self.position, self.done, spec_tile, inv = state
        self.special_tiles = spec_tile.copy()
        self.inventory = inv.copy()




# class MineWorldEnvContinuousDiego(GridEnv, SaveLoadEnv):
#     """A basic minecraft-like environment, with a global view of the state space"""
#     @staticmethod
#     def from_dict(dict):
#         return MineWorldEnv(MineWorldConfig.from_dict(dict))

#     def __init__(self, config: MineWorldConfig, *args, **kwargs):
#         super().__init__(shape=config.shape, *args, **kwargs)

#         # self.action_space = spaces.Discrete(6) # Input can only be 0, 1, 2, 3, 4, 5
#         self.action_space = spaces.Box(low=0.0, high=5.0, shape=(1,), dtype=np.float32)
#         self.observation_space = spaces.Box(0, 1,
#                                             shape=(1 + len(config.placements) + len(config.inventory), *config.shape),
#                                             dtype=np.float32)
#         self.config = config
#         self.default_inventory = Counter(
#             {inv_type.name: inv_type.default_quantity for inv_type in self.config.inventory})
#         self.rand = Random()

#         # Position Update
#         # [0, 3):   Angle of step (map to degree values)
#         # [3, 4):   No Movement
#         # [4 , 5]:  Tile Action

#         self.done = True
#         self.true_position: Tuple[float, float] = (0.0, 0.0) ### x, y
#         self.tile_position: Tuple[int, int]     = (0, 0) ### Truncated version of self.true_position
#         self.special_tiles: Dict[Tuple[int, int], MineWorldTileType] = dict()
#         self.inventory = Counter()

#     def step(self, action: float):
#         # print(action)
#         # print(f"{self.action_space.low} {self.action_space.high}")
        
#         action = [action]
#         assert self.action_space.contains(action)
#         assert not self.done

#         action_names = set()

#         reward = 0

#         ###
#         action = action[0]

#         if action < 3:
#             # Movement or no-op
#             # action_offsets = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]
#             action_offsets = (cos((action/3)*6.28), sin((action/3)*6.28))

#             # new_place = element_add(self.position, action_offsets[action])
#             new_place = element_add(self.true_position, action_offsets)

#             can_move = self._in_bounds(new_place) # might fail

#             if new_place in self.special_tiles:
#                 tile = self.special_tiles[new_place]
#                 if tile.wall or not tile.move_requirements(self.inventory):
#                     can_move = False

#             if can_move:
#                 self.true_position = new_place
#                 self.tile_position = (int(self.true_position[0]), int(self.true_position[1]))
#         elif action > 4:
#             if self.tile_position in self.special_tiles:
#                 this_tile: MineWorldTileType = self.special_tiles[self.tile_position]
#                 if this_tile.meets_requirements(self.inventory):
#                     new_inv = this_tile.apply_inventory(self.inventory)
#                     for inv_config in self.config.inventory:
#                         if new_inv[inv_config.name] > inv_config.capacity:
#                             new_inv[inv_config.name] = inv_config.capacity
#                     self.inventory = new_inv
#                     action_names.add(this_tile.action_name)
#                     if this_tile.consumable:
#                         del self.special_tiles[self.tile_position]
#                     if this_tile.terminal:
#                         self.done = True
#                     reward += this_tile.reward

#         info = {
#             'tile_action_names': action_names,
#             'inventory': self.inventory.copy(),
#             'position': self.true_position
#         }

#         return obs_rewrite(self.shape, self._get_observation()), reward, self.done, info

#     def seed(self, seed=None):
#         self.rand.seed(seed)

#     def reset(self):
#         self.done = False
#         self.true_position = self.config.initial_position
#         if not self.true_position:
#             # self.true_position = self.rand.randrange(0, self.shape[0]), self.rand.randrange(0, self.shape[1])
#             self.true_position = (self.shape[0] * self.rand.random(), self.shape[1] * self.rand.random())
#         self.inventory = self.default_inventory.copy()
#         self.special_tiles = self._get_tile_positioning()

#         return obs_rewrite(self.shape, self._get_observation())

#     # Might not work
#     def _get_tile_positioning(self) -> Dict[Tuple[int, int], MineWorldTileType]:

#         tiles = {}

#         for tile_type in self.config.placements:
#             for fixed in tile_type.fixed_placements:
#                 tiles[fixed] = tile_type.tile

#         # noinspection PyTypeChecker
#         all_spaces = set(np.ndindex(self.config.shape))
#         open_spaces = all_spaces.difference(tiles.keys())
#         if (0, 0) in open_spaces:
#             open_spaces.remove((0, 0))

#         for tile_type in self.config.placements:
#             tile, num_placements = tile_type.tile, tile_type.random_placements
#             spaces = self.rand.sample(open_spaces, num_placements)
#             open_spaces.difference_update(spaces)

#             for space in spaces:
#                 tiles[space] = tile

#         return tiles

#     def _get_observation(self):

#         tiles = tuple(
#             frozenset(space for space, content in self.special_tiles.items() if content is placement.tile) for
#             placement in self.config.placements)

#         inv = tuple(self.inventory[inv_config.name] for inv_config in self.config.inventory)

#         return (
#             self.tile_position,
#             tiles,
#             inv
#         )

#     # Probably wont work, doesnt translate well to continuous
#     def render(self, mode='human'):
#         def render_func(x, y):
#             agent_str = "A" if self.tile_position == (x, y) else " "
#             tile_str = self.special_tiles[(x, y)].grid_letter if (x, y) in self.special_tiles else " "
#             return agent_str + tile_str, False, False

#         print(self._render(render_func, 2), end="")
#         print(dict(self.inventory))

#     def save_state(self):
#         return self.true_position, self.done, self.special_tiles.copy(), self.inventory.copy()

#     def load_state(self, state):
#         self.true_position, self.done, spec_tile, inv = state
#         self.special_tiles = spec_tile.copy()
#         self.inventory = inv.copy()
