Source code for paper "Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning".

The code tests five different reinforcement learning methods:
- dynamic automaton distillation (proposed),
- static automaton distillation (proposed)
- CRM
- DQN on Product MDP
- DQN without transfer learning (vanilla)

The overall organization:
    automaton_q: Cached automata and teacher Q values. This is extra information of the teacher that will be loaded later when running the student config.
	lib: the actual implementation of automaton transfer
	run: individual experiments. The files themselves should craft a config and call lib.main.run_training
		env: different environments that will be reused across different runs. Their configurations live here.
		utils: when two experiments require similar configurations, create a function in here to reduce repetition
		teacher: the teacher configs that serve either as DQN without transfer learning or as "teacher"
		target: the target configs that serve as the "student" in knowledge transfer
		experiment: file used for generating results submitted to AAAI 2023

Setup - install the required packages as shown in the environment.yml and requirement.txt file
        (also see the environment.yml and requirements.txt files for additional setup instructions):
  1) conda env create -f  environment.yml
  2) conda activate autd
  3) Add source code root to PYTHONPATH

To train the teachers:
  1) Delete contents of automaton_q folder

  2) Run teachers:
       python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py
       python discrete/run/teacher/blind_craftsman_teacher_rew_per_step_7_productMDP.py
       python discrete/run/teacher/gold_mine_teacher_rew_per_step_7_productMDP.py

  3) Generate Q values from abstract MDPs:
       python discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
       python discrete/run/reward_machine/blind_craftsman_machine_rew_per_step.py
       python discrete/run/reward_machine/gold_mine_machine_rew_per_step.py

To train the students and related algorithms:
  1) Run the training for each environment/student:
     A) Dynamic Automaton Distillation:
         python discrete/run/target/dungeon_quest_7_10_target_rew_per_step_productMDP.py
         python discrete/run/target/blind_craftsman_7_10_target_rew_per_step_productMDP.py
         python discrete/run/target/blind_craftsman_7_obstacles_target_rew_per_step_productMDP.py
         python discrete/run/target/gold_mine_7_10_target_rew_per_step_productMDP.py

     B) Static Automaton Distillation:
         python discrete/run/target/dungeon_quest_target_machine_q_rew_per_step_productMDP.py
         python discrete/run/target/blind_craftsman_target_machine_rew_per_step_productMDP.py
         python discrete/run/target/blind_craftsman_obstacles_target_machine_rew_per_step_productMDP.py
         python discrete/run/target/gold_mine_target_machine_q_rew_per_step_productMDP.py

     C) CRM:
         python discrete/run/target/dungeon_quest_target_machine_rew_per_step_CRM.py
         python discrete/run/target/blind_craftsman_target_machine_rew_per_step_CRM.py
         python discrete/run/target/blind_craftsman_obstacles_target_machine_rew_per_step_CRM.py
         python discrete/run/target/gold_mine_target_machine_rew_per_step_CRM.py

     D) Product MDP Q-Learning:
         python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_productMDP.py
         python discrete/run/teacher/blind_craftsman_teacher_rew_per_step_productMDP.py
         python discrete/run/teacher/blind_craftsman_obstacles_teacher_rew_per_step_productMDP.py
         python discrete/run/teacher/gold_mine_teacher_rew_per_step_productMDP.py

     E) Vanilla Q-Learning:
         python discrete/run/teacher/dungeon_quest_teacher_rew_per_step.py
         python discrete/run/teacher/blind_craftsman_teacher_rew_per_step.py
         python discrete/run/teacher/blind_craftsman_obstacles_teacher_rew_per_step.py
         python discrete/run/teacher/gold_mine_teacher_rew_per_step.py

     Note: The helper script discrete/run/experiment/paperexperiments.py can be used to run multiple experiment trials
           including a run ID to make the checkpoints/logs unique to each trial.

To extract CSV results from logs for plotting:
  python discrete/run/experiment/extract_events_to_csv.py LOGFOLDER


NOTES:
When running for the first time, training should print "NOT loading from the checkpoint"; otherwise, it will automatically load from the last checkpoint.
All the logs are in tensorboard. To view, run tensorboard --logdir logs

Running the code requires a GPU, although the code can be modified to run without a GPU (using CPU instead of CUDA).
It takes long to run each config. For my computer with a RTX 3060, it takes about five hours to run each config, and the running time depends on what machine you are running it on.

Note that the environment name Gold Mine in the code was renamed to Diamond Mine in the paper.

