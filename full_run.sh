rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py
python discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
python discrete/run/target/dungeon_quest_7_target_rew_per_step_productMDP_DDPG.py # Dynamic Distillation
# python discrete/run/target/dungeon_quest_target_machine_q_rew_per_step_productMDP.py # Static Distillation
# python discrete/run/experi-ment/extract_events_to_csv.py ./../

# tensorboard --logdir ./logs/