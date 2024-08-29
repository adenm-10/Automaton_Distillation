rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

export dragon_r=10
export bounding_persist="True"
export bounding_dist=3
export seq_level=2

356977/dragon-r_10_bound-persist_True_bounding-dist_3_seq-level_2

# Hyperparameters
alr=0.0001
clr=0.001
gamma=0.99
tau=0.001
total_steps=3000

python discrete/run/teacher/dungeon_quest_teacher_ddpg.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"