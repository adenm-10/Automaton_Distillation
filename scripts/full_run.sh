rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

# Hyperparameters
alr=0.0003
clr=0.0003
gamma=0.99
tau=0.005
total_steps=2500
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.2

export seq_level=2
python -u ./discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py --total-steps="$total_steps"
python -u ./discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
export seq_level=2
python -u ./discrete/run/target/dungeon_quest_7_target_rew_per_step_productMDP_TD3.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"