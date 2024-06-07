from discrete.lib.config import EnvConfig
from discrete.lib.automaton.mine_env_ap_extractor import AP
from discrete.lib.automaton.space_invaders_ap import InPathBullet, BehindCover, EnemyRemaining, EnemyDestroyed

easy_config = ({'enemies': (4, 4), 'bunkers': 3, 'shape': (20, 20), 'alien_shape': (1, 1),
                     'player_shape': (2, 2), 'bunker_shape': (4, 4), 'max_time': 500})

space_invaders_config = EnvConfig(
    env_name='SpaceInvadersGridworld-v0',
    kwargs={'config': easy_config})

space_invaders_aps = [
    AP('in_path_bullet', InPathBullet()),
    AP('behind_cover', BehindCover()),
    AP('enemy_remaining', EnemyRemaining()),
    AP('enemy_destroyed', EnemyDestroyed())
]

space_invaders_ltlf = 'G((in_path_bullet & !behind_cover) -> F (!in_path_bullet | behind_cover)) & ' \
                      'G(enemy_remaining -> F enemy_destroyed)'
