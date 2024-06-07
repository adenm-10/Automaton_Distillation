import torch

from discrete.run.utils import construct_ap_extractor_automaton
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf

device = torch.device("cpu")

automaton, _ = construct_ap_extractor_automaton(dungeon_quest_aps, dungeon_quest_ltlf, device)
print([ap.name for ap in dungeon_quest_aps])
print(automaton.adj_mat)

reward_adj_list = torch.zeros_like(automaton.adj_mat)

# success = +100
reward_adj_list[automaton.adj_mat == 1] += 100

# no reward in terminal states
reward_adj_list[1, :] = 0
reward_adj_list[7, :] = 0

rm = RewardMachine(automaton, reward_adj_list, "dungeon_quest_machine", device)