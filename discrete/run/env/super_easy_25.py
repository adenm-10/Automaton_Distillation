from collections import Counter

from discrete.lib.config import EnvConfig
from discrete.lib.env.mineworldenv import MineWorldConfig, TilePlacement, MineWorldTileType
from discrete.lib.env.rew_every_step import RewEveryStep

super_easy_config_25 = MineWorldConfig(
    shape=(25, 25),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="a", consumable=True, grid_letter="A", inventory_modifier=Counter(), reward=+100,
                terminal=True
            ),
            fixed_placements=[],
            random_placements=1
        ),
    ],
    inventory=[
    ]
)

super_easy_env_config_25 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": super_easy_config_25},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -1})
