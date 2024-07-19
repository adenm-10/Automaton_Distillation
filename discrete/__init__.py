import gym

continuous_as_steps_to_term_state = 1000
discrete_as_steps_to_term_state = 1000

try:
    

print("\n\n============================================")
print(f"Discrete Steps until Terminal State: {discrete_as_steps_to_term_state}")
print(f"Continuous Steps until Terminal State: {continuous_as_steps_to_term_state}")

gym.register("MineWorldEnv-v0",
             entry_point="discrete.lib.env.mineworldenv:MineWorldEnv",
             nondeterministic=False,
             max_episode_steps=discrete_as_steps_to_term_state)
            #  max_episode_steps=1000)

gym.register("MineWorldEnv-v1",
             entry_point="discrete.lib.env.mineworldenv:MineWorldEnvContinuous",
             nondeterministic=False,
             max_episode_steps=continuous_as_steps_to_term_state)
            #  max_episode_steps=1000)


gym.register("ObtainDiamondGridworld-v0",
             entry_point="discrete.lib.env.obtaindiamondenv:ObtainDiamond",
             nondeterministic=False,
             max_episode_steps=10000
             )

gym.register("SpaceInvadersGridworld-v0",
             entry_point="discrete.lib.env.spaceinvadersenv:SpaceInvaders",
             nondeterministic=False,
             max_episode_steps=10000
             )

gym.register("DragonFightGridworld-v0",
             entry_point="discrete.lib.env.dragonfightenv:DragonFight",
             nondeterministic=False,
             max_episode_steps=10000
             )
