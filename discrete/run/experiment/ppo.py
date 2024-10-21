from stable_baselines3 import PPO
import gym
from discrete.run.env.dungeon_quest_7 import dungeon_quest_rew_per_step_env_config_7 as config
import discrete.lib.env.mineworldenv

if __name__ == "__main__":
        
        env = gym.make(config.env_name, **config.kwargs)

        model = PPO(
            "MlpPolicy",
            env=env,
            n_steps=512,
            n_epochs=20,
            seed=0,
            policy_kwargs=dict(net_arch=[16]),
            verbose=2,
            tensorboard_log="./test_output"
        )
        model.learn(total_timesteps=1000)