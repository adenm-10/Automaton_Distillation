import torch

from discrete.run.utils import construct_ap_extractor_automaton
from discrete.lib.automaton.reward_machine import RewardMachine
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf, dragon_r

device = torch.device("cpu")

automaton, _ = construct_ap_extractor_automaton(dungeon_quest_aps, dungeon_quest_ltlf, device)
print([ap.name for ap in dungeon_quest_aps])
print(automaton.adj_mat)

# -0.1 per step
reward_adj_list = -0.1 * torch.ones_like(automaton.adj_mat)

# success = +100
# one less than terminal reward?
reward_adj_list[automaton.adj_mat == 7] += 9

# key/sword/shield/dragon = +1
for i in range(automaton.num_states):
    reward_adj_list[i, automaton.adj_mat[i] != i] += dragon_r - 1

# no reward in terminal states
# reward_adj_list[1, :] = 0
# reward_adj_list[7, :] = 0
terminal_states = torch.as_tensor([0,1,0,0,0,0,0,1], dtype=torch.float, device=device)

rm = RewardMachine(automaton, reward_adj_list, terminal_states, "dungeon_quest_machine_rew_per_step", device)