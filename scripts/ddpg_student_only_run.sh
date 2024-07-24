rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

# Hyperparameters
alr=0.0001
clr=0.001
gamma=0.99
tau=0.001
total_steps=2000

python discrete/run/teacher/dungeon_quest_teacher_ddpg.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"