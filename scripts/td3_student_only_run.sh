rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

export dragon_r=10
export bounding_persist="True"
export bounding_dist=3
export seq_level=2

# Hyperparameters
alr=0.0003
clr=0.0003
gamma=0.99
tau=0.005
total_steps=5000
export POLICY_FREQ=2
export NOISE_CLIP=0.5
export NOISE_STDDEV=0.2

python discrete/run/teacher/dungeon_quest_teacher_td3.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --total-steps="$total_steps"

unset POLICY_FREQ
unset NOISE_CLIP
unset NOISE_STDDEV