rm -rf ./automaton_q/*
rm -rf ./logs/*
rm -rf ./checkpoints/*

# Should match the environment used when used as a student
python discrete/run/teacher/dungeon_quest_teacher_ddpg.py