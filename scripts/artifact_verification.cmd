del /s /q ".\automaton_q\*
python discrete/run/teacher/dungeon_quest_teacher_rew_per_step_7_productMDP.py --total-steps="1500"
python discrete/run/reward_machine/dungeon_quest_machine_rew_per_step.py
python discrete/run/target/dungeon_quest_7_10_target_rew_per_step_productMDP.py --total-steps="1500"