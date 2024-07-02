rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

alr=0.0001
clr=0.001
gamma=0.99
tau=1.0
dragon_r=1

# for dragon_r in 1 5 10 20 50 100
# do
# Should match the environment used when used as a student
python discrete/run/teacher/dungeon_quest_teacher_ddpg.py --alr="$alr" --clr="$clr" --gamma="$gamma" --tau="$tau" --dragon-reward="$dragon_r"
# done