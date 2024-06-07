import gym

class TimeLimit(gym.Wrapper):
    """
    Modified from https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py to add rew_on_expired
    """

    def __init__(self, env, max_episode_steps=None, rew_on_expired: float = 0):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None
        self._rew_on_expired = rew_on_expired

    def step(self, action):
        assert (
                self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            if not done:
                reward += self._rew_on_expired
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
