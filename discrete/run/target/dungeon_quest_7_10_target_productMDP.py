import torch

from discrete.lib.automaton.target_automaton import ExponentialAnnealTargetAutomaton
from discrete.lib.main import run_training
from discrete.run.env.dungeon_quest import dungeon_quest_env_config
from discrete.run.env.dungeon_quest import dungeon_quest_aps, dungeon_quest_ltlf
from discrete.run.utils import student_config_v1

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device('cuda')

config = student_config_v1(
    env_config=dungeon_quest_env_config,
    teacher_run_name="dungeon_quest_teacher_7_productMDP",
    student_run_name="dungeon_quest_7_10_target_productMDP",
    device=device,
    anneal_target_aut_class=ExponentialAnnealTargetAutomaton,
    anneal_target_aut_kwargs={
        "exponent_base": 0.999
    },
    aps=dungeon_quest_aps,
    ltlf=dungeon_quest_ltlf
)

if __name__ == '__main__':
    run_training(config)
