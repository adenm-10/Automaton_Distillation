from collections import Counter
import os

from discrete.lib.automaton.mine_aps import MineLocationAP, MineInventoryAP, MineInfoAutAP
from discrete.lib.automaton.mine_env_ap_extractor import AP
from discrete.lib.config import EnvConfig
from discrete.lib.env.mineworldenv import MineWorldConfig, TilePlacement, InventoryItemConfig, MineWorldTileType
from discrete.lib.env.rew_every_step import RewEveryStep
from discrete.lib.env.time_limit import TimeLimit

dragon_r, key_r, shield_r, sword_r = 100, 1, 1, 1
sequence_level = 2 # 0-3

try:
    dragon_r = int(os.environ.get("dragon_r"))
    key_r = int(os.environ.get("key_r"))
    shield_r = int(os.environ.get("shield_r"))
    sword_r = int(os.environ.get("sword_r"))
    sequence_level = int(os.environ.get("seq_level"))
except:
    print("Shell variables not detected for environment descriptions, using defaults...")

key_r, key_p, shield_r, shield_p, key_t, shield_t = 0, 0, 0, 0, False, False
sword_r, sword_p, dragon_r, dragon_p, sword_t, dragon_t = 0, 0, 0, 0, False, False

# key
if sequence_level == 0:
    key_p, key_t =  1, True
    shield_p, shield_t = 0, False
    sword_p, sword_t = 0, False
    dragon_p, dragon_t = 0, False

# key -> sword
elif sequence_level == 1:
    key_p, key_t = 1, False
    shield_p, shield_t = 0, False
    sword_p, sword_t = 1, True
    dragon_p, dragon_t = 0, False

# key -> sword -> dragon
elif sequence_level == 2:
    key_p, key_t = 1, False
    shield_p, shield_t = 1, False
    sword_p, sword_t = 0, False
    dragon_p, dragon_t = 1, True

# key -> sword V shield -> dragon
elif sequence_level == 3:
    key_p, key_t = 1, False
    shield_p, shield_t = 1, False
    sword_p, sword_t = 1, False
    dragon_p, dragon_t = 1, True


print("\n====================================")
print(f"Rewards for Automaton Steps (reward, placements)")
print(f"\tKey: {key_r}, {key_p}")
print(f"\tShield: {shield_r}, {shield_p}")
print(f"\tSword: {sword_r}, {sword_p}")
print(f"\tDragon: {dragon_r}, {dragon_p}")
print(f"Sequence Level: {sequence_level}")
print("====================================\n")

dungeon_quest_config_7 = MineWorldConfig(
    shape=(7, 7),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="key", consumable=True, grid_letter="K", inventory_modifier=Counter(key=+1), reward=+key_r, terminal=key_t
            ),
            fixed_placements=[],
            random_placements=key_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="shield", consumable=True, grid_letter="S", inventory_modifier=Counter(shield=+1),
                reward=+shield_r, terminal=shield_t
            ),
            fixed_placements=[],
            random_placements=shield_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="chest", consumable=False, grid_letter="C", inventory_modifier=Counter(key=-1, sword=+1),
                inventory_requirements=Counter(key=1), reward=+sword_r, terminal=sword_t
            ),
            fixed_placements=[],
            random_placements=sword_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="dragon", consumable=True, grid_letter="D", inventory_modifier=Counter(),
                inventory_requirements=Counter(sword=sword_p, shield=shield_p), reward=+100, terminal=dragon_t
            ),
            fixed_placements=[],
            random_placements=dragon_p
        )
    ],
    inventory=[
        InventoryItemConfig(name="key", default_quantity=0, capacity=1),
        InventoryItemConfig(name="sword", default_quantity=0, capacity=1),
        InventoryItemConfig(name="shield", default_quantity=0, capacity=1)
    ]
)

dungeon_quest_config_7_cont = MineWorldConfig(
    shape=(2),
    tile_shape=(7, 7),
    initial_position=None,
    placements=[
        TilePlacement(
            tile=MineWorldTileType(
                action_name="key", consumable=True, grid_letter="K", inventory_modifier=Counter(key=+1), reward=+key_r, terminal=key_t
            ),
            fixed_placements=[],
            random_placements=key_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="shield", consumable=True, grid_letter="S", inventory_modifier=Counter(shield=+1),
                reward=+shield_r, terminal=shield_t
            ),
            fixed_placements=[],
            random_placements=shield_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="chest", consumable=False, grid_letter="C", inventory_modifier=Counter(key=-1, sword=+1),
                inventory_requirements=Counter(key=1), reward=+sword_r, terminal=sword_t
            ),
            fixed_placements=[],
            random_placements=sword_p
        ),
        TilePlacement(
            tile=MineWorldTileType(
                action_name="dragon", consumable=True, grid_letter="D", inventory_modifier=Counter(),
                inventory_requirements=Counter(sword=sword_p, shield=shield_p), reward=+dragon_r, terminal=dragon_t
            ),
            fixed_placements=[],
            random_placements=dragon_p
        )
    ],
    inventory=[
        InventoryItemConfig(name="key", default_quantity=0, capacity=1),
        InventoryItemConfig(name="sword", default_quantity=0, capacity=1),
        InventoryItemConfig(name="shield", default_quantity=0, capacity=1)
    ]
)

dungeon_quest_rew_per_step_env_config_7_cont = EnvConfig(
    env_name="MineWorldEnv-v1",
    kwargs={"config": dungeon_quest_config_7_cont},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

dungeon_quest_rew_per_step_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7},
    wrapper_cls=RewEveryStep,
    wrapper_kwargs={"rew_per_step": -0.1}
)

dungeon_quest_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7}
)

dungeon_quest_exp_env_config_7 = EnvConfig(
    env_name="MineWorldEnv-v0",
    kwargs={"config": dungeon_quest_config_7},
    wrapper_cls=TimeLimit,
    wrapper_kwargs={"max_episode_steps": 999, "rew_on_expired": -1}
)

# APS AND LTLF BOTH CHANGED
# ORIGINAL
# dungeon_quest_aps = [
#     AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
#     AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
#     AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
#     AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
# ]

AP_list = [[
            AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
            # AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
            # AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
            # AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
            ],
            [
            AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
            AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
            # AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
            # AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
            ],
            [
            AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
            AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
            # AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
            AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
            ],
            [
            AP(name="key", func=MineInventoryAP(inventory_item="key", quantity=1)),
            AP(name="sword", func=MineInventoryAP(inventory_item="sword", quantity=1)),
            AP(name="shield", func=MineInventoryAP(inventory_item="shield", quantity=1)),
            AP(name="dragon", func=MineInfoAutAP(ap_name="dragon"))
            ],
            ]

LTLF_list = ["F(key)",
            "F(sword) & (!sword U key)",
            "F(dragon) & (!sword U key) & (!dragon U sword)",
            "F(dragon) & (!sword U key) & (!dragon U sword) & (!dragon U shield)"]

dungeon_quest_aps = AP_list[sequence_level]
dungeon_quest_ltlf = LTLF_list[sequence_level]

print(f"Dungeon Quest LTLF: {dungeon_quest_ltlf}")
print("============================================\n\n")
