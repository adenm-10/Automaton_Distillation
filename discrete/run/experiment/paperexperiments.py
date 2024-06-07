from discrete.lib.main import run_training

# Blind Craftsman
from discrete.run.teacher import blind_craftsman_teacher_rew_per_step_7_productMDP
from discrete.run.target import blind_craftsman_7_10_target_rew_per_step_productMDP,\
                                blind_craftsman_target_machine_rew_per_step_productMDP,\
                                blind_craftsman_target_machine_rew_per_step_CRM
from discrete.run.teacher import blind_craftsman_teacher_rew_per_step,\
                                 blind_craftsman_teacher_rew_per_step_productMDP

# Blind Craftsman with Obstacles
from discrete.run.teacher import blind_craftsman_obstacles_teacher_rew_per_step,\
                                 blind_craftsman_obstacles_teacher_rew_per_step_productMDP
from discrete.run.target import blind_craftsman_7_obstacles_target_rew_per_step_productMDP,\
                                blind_craftsman_obstacles_target_machine_rew_per_step_productMDP,\
                                blind_craftsman_obstacles_target_machine_rew_per_step_CRM

# Dungeon quest
from discrete.run.teacher import dungeon_quest_teacher_rew_per_step_7_productMDP
from discrete.run.target import dungeon_quest_7_10_target_rew_per_step_productMDP,\
                                dungeon_quest_target_machine_q_rew_per_step_productMDP,\
                                dungeon_quest_target_machine_rew_per_step_CRM
from discrete.run.teacher import dungeon_quest_teacher_rew_per_step,\
                                 dungeon_quest_teacher_rew_per_step_productMDP

# Gold/Diamond Mine
from discrete.run.teacher import gold_mine_teacher_rew_per_step_7_productMDP
from discrete.run.target import gold_mine_7_10_target_rew_per_step_productMDP,\
                                gold_mine_target_machine_q_rew_per_step_productMDP,\
                                gold_mine_target_machine_rew_per_step_CRM
from discrete.run.teacher import gold_mine_teacher_rew_per_step,\
                                 gold_mine_teacher_rew_per_step_productMDP


def run_teachers():
    run_training(dungeon_quest_teacher_rew_per_step_7_productMDP.config)
    run_training(blind_craftsman_teacher_rew_per_step_7_productMDP.config)
    run_training(gold_mine_teacher_rew_per_step_7_productMDP.config)


def run_blind_craftsman(run_id=""):
    # Dynamic Automaton Distillation
    run_training(blind_craftsman_7_10_target_rew_per_step_productMDP.config,
                 "BlindCraftsman_DynamicAutomatonDistillation-{}".format(run_id))

    # Static Automaton Distillation - NOTE: this doesn't follow naming convention of dungeon quest/gold mine (missing "q" in name)
    run_training(blind_craftsman_target_machine_rew_per_step_productMDP.config,
                 "BlindCraftsman_StaticAutomatonDistillation-{}".format(run_id))

    # CRM
    run_training(blind_craftsman_obstacles_target_machine_rew_per_step_CRM.config,
                 "BlindCraftsman_CRM-{}".format(run_id))

    # Product MDP Q-Learning - NOTE: I had to create this script - I couldn't find anything similar int he codebase
    run_training(blind_craftsman_teacher_rew_per_step_productMDP.config,
                 "BlindCraftsman_ProductMDPQLearning-{}".format(run_id))

    # Vanilla Q-Learning
    run_training(blind_craftsman_teacher_rew_per_step.config,
                 "BlindCraftsman_VanillaQLearning-{}".format(run_id))

def run_blind_craftsman_obstacles(run_id=""):
    # Dynamic Automaton Distillation
    run_training(blind_craftsman_7_obstacles_target_rew_per_step_productMDP.config,
                 "BlindCraftsman_DynamicAutomatonDistillation-{}".format(run_id))

    # Static Automaton Distillation - NOTE: this doesn't follow naming convention of dungeon quest/gold mine (missing "q" in name)
    run_training(blind_craftsman_obstacles_target_machine_rew_per_step_productMDP.config,
                 "BlindCraftsman_StaticAutomatonDistillation-{}".format(run_id))

    # CRM
    run_training(blind_craftsman_obstacles_target_machine_rew_per_step_CRM.config,
                 "BlindCraftsman_CRM-{}".format(run_id))

    # Product MDP Q-Learning
    run_training(blind_craftsman_obstacles_teacher_rew_per_step_productMDP.config,
                 "BlindCraftsman_ProductMDPQLearning-{}".format(run_id))

    # Vanilla Q-Learning
    run_training(blind_craftsman_obstacles_teacher_rew_per_step.config,
                 "BlindCraftsman_VanillaQLearning-{}".format(run_id))

def run_dungeon_quest(run_id=""):
    # Dynamic Automaton Distillation
    run_training(dungeon_quest_7_10_target_rew_per_step_productMDP.config,
                 "DungeonQuest_DynamicAutomatonDistillation-{}".format(run_id))

    # Static Automaton Distillation
    run_training(dungeon_quest_target_machine_q_rew_per_step_productMDP.config,
                 "DungeonQuest_StaticAutomatonDistillation-{}".format(run_id))

    # CRM
    run_training(dungeon_quest_target_machine_rew_per_step_CRM.config,
                 "DungeonQuest_CRM-{}".format(run_id))

    # Product MDP Q-Learning
    run_training(dungeon_quest_teacher_rew_per_step_productMDP.config,
                 "DungeonQuest_ProductMDPQLearning-{}".format(run_id))

    # Vanilla Q-Learning
    run_training(dungeon_quest_teacher_rew_per_step.config,
                 "DungeonQuest_VanillaQLearning-{}".format(run_id))


def run_gold_mine(run_id=""):
    # Dynamic Automaton Distillation
    run_training(gold_mine_7_10_target_rew_per_step_productMDP.config,
                 "GoldMine_DynamicAutomatonDistillation-{}".format(run_id))

    # Static Automaton Distillation
    run_training(gold_mine_target_machine_q_rew_per_step_productMDP.config,
                 "GoldMine_StaticAutomatonDistillation-{}".format(run_id))

    # CRM
    run_training(gold_mine_target_machine_rew_per_step_CRM.config,
                 "GoldMine_CRM-{}".format(run_id))

    # Product MDP Q-Learning
    run_training(gold_mine_teacher_rew_per_step_productMDP.config,
                 "GoldMine_ProductMDPQLearning-{}".format(run_id))

    # Vanilla Q-Learning
    run_training(gold_mine_teacher_rew_per_step.config,
                 "GoldMine_VanillaQLearning-{}".format(run_id))


if __name__ == '__main__':
    run_teachers()
    run_blind_craftsman()
    run_blind_craftsman_obstacles()
    run_dungeon_quest()
    run_gold_mine()