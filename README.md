# Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning

## Source Code
Original source code can be found at: https://github.com/adenm-10/Automaton_Distillation

## Setup

Make sure to have git and conda installed before hand

1. Clone the repository using:
    
    `git clone https://github.com/adenm-10/Automaton_Distillation.git`
2. Create the necessary conda environment using:

    `conda env create -f  environment.yml`
3. Activate and finish the manual environment setup using:
    
    `conda activate autd`
    
    
    `pip install stable-baselines3`
    
    
    `pip install --no-deps stable-baselines3==1.5.0 gym==0.15.7 numpy==1.17.5`

4. Add source code root to PYTHONPATH

    `export PYTHONPATH="/path/to/your/repo:$PYTHONPATH"`
    `set PYTHONPATH="/path/to/your/repo:$PYTHONPATH`
    - This can be done permanently by appending the above line to your ~/.bashrc file 

5. See `discrete/requirements.txt` and `discrete/environment.yml` for exact environment specifications and package downloads

## For Artifact Verification

Go to root directory of repo (./Automaton_Distillation/)
Linux: `source ./scripts/artifact_verification.sh`
Windows: `.\scripts\artifact_verification.cmd`

## Experiment Description

The code tests five different reinforcement learning methods:
- Dynamic Automaton Distillation (proposed),
- Static Automaton Distillation (proposed)
- CRM
- DQN on Product MDP
- DQN without transfer learning (vanilla)

The overall organization:
- automaton_q: Cached automata and teacher Q values. This is extra information of the teacher that will be loaded later when running the student config.
- lib: the actual implementation of automaton transfer
- run: individual experiments. The files themselves should craft a config and call lib.main.run_training
  - env: different environments that will be reused across different runs. Their configurations live here.
  - utils: when two experiments require similar configurations, create a function in here to reduce repetition
  - teacher: the teacher configs that serve either as DQN without transfer learning or as "teacher"
  - target: the target configs that serve as the "student" in knowledge transfer
  - experiment: file used for generating results submitted to ICAPS 2025

## Experiment 

To train the teachers:
    
    1) Delete contents of automaton_q folder
        Linux: `rm -rf ./automaton_q/*`
        Windows: `del /s /q ".\automaton_q\*"`
    2) Run teachers:
       1) python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py
       2) python discrete/run/teacher/blind_craftsman_teacher_rew_per_step_7_productMDP.py
       3) python discrete/run/teacher/gold_mine_teacher_rew_per_step_7_productMDP.py

    3) Generate Q values from abstract MDPs:
       1) python discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
       2) python discrete/run/reward_machine/blind_craftsman_machine_rew_per_step.py
       3) python discrete/run/reward_machine/gold_mine_machine_rew_per_step.py

To train the students and related algorithms run the training for each environment/student:
    
    1) Dynamic Automaton Distillation:
        1) python discrete/run/target/dungeon_quest_7_10_target_rew_per_step_productMDP.py
        2) python discrete/run/target/blind_craftsman_7_10_target_rew_per_step_productMDP.py
        3) python discrete/run/target/blind_craftsman_7_obstacles_target_rew_per_step_productMDP.py
        4) python discrete/run/target/gold_mine_7_10_target_rew_per_step_productMDP.py

    2) Static Automaton Distillation:
        1) python discrete/run/target/dungeon_quest_target_machine_q_rew_per_step_productMDP.py
        2) python discrete/run/target/blind_craftsman_target_machine_rew_per_step_productMDP.py
        3) python discrete/run/target/blind_craftsman_obstacles_target_machine_rew_per_step_productMDP.py
        4) python discrete/run/target/gold_mine_target_machine_q_rew_per_step_productMDP.py

    3) CRM:
        1)  python discrete/run/target/dungeon_quest_target_machine_rew_per_step_CRM.py
        2) python discrete/run/target/blind_craftsman_target_machine_rew_per_step_CRM.py
        3) python discrete/run/target/blind_craftsman_obstacles_target_machine_rew_per_step_CRM.py
        4) python discrete/run/target/gold_mine_target_machine_rew_per_step_CRM.py

    4) Product MDP Q-Learning:
        1) python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_productMDP.py
        2) python discrete/run/teacher/blind_craftsman_teacher_rew_per_step_productMDP.py
        3) python discrete/run/teacher/blind_craftsman_obstacles_teacher_rew_per_step_productMDP.py
        4) python discrete/run/teacher/gold_mine_teacher_rew_per_step_productMDP.py

    5) Vanilla Q-Learning:
        1) python discrete/run/teacher/dungeon_quest_teacher_rew_per_step.py
        2) python discrete/run/teacher/blind_craftsman_teacher_rew_per_step.py
        3) python discrete/run/teacher/blind_craftsman_obstacles_teacher_rew_per_step.py
        4) python discrete/run/teacher/gold_mine_teacher_rew_per_step.py

    6) Dynamic Automaton Distillation to Continuous environments
        1) python discrete/run/teacher/dungeon_quest_7_target_rew_per_step_productMDP_TD3
        2) python discrete/run/teacher/blind_craftsman_7_target_rew_per_step_productMDP_TD3
        3) python discrete/run/teacher/gold_mine_7_target_rew_per_step_productMDP_TD3

     Note: The helper script discrete/run/experiment/paperexperiments.py can be used to run multiple experiment trials including a run ID to make the checkpoints/logs unique to each trial.

To extract CSV results from logs for plotting:
  python discrete/run/experiment/extract_events_to_csv.py LOGFOLDER


NOTES:
When running for the first time, training should print "NOT loading from the checkpoint"; otherwise, it will automatically load from the last checkpoint.
All the logs are in tensorboard. To view, run tensorboard --logdir logs

Running the code requires a GPU, although the code can be modified to run without a GPU (using CPU instead of CUDA).
It takes long to run each config. For my computer with a RTX 3060, it takes about five hours to run each config. and the running time depends on what machine you are running it on.

Note that the environment name Gold Mine in the code was renamed to Diamond Mine in the paper.

