from collections import deque
from typing import Tuple, Dict
import numpy as np
from gym import spaces
from math import ceil

from discrete.lib.env.saveloadenv import SaveLoadEnv


class Alien:
    # An individual alien
    def __init__(self, pos, shape, size=(1, 2)):
        self.shape = shape
        self.pos = pos
        self.size = size
        self.gun_pos = (pos[0], pos[1] + self.size[1])

    def step(self, dir_x, dir_y):
        # Just moves the alien and makes sure none are hitting a wall
        if dir_x < 0:
            self.pos = (max(self.pos[0] + dir_x, 0), self.pos[1] + dir_y)
        else:
            self.pos = (min(self.pos[0] + dir_x, self.shape[0] - 1), self.pos[1] + dir_y)
        self.gun_pos = (self.pos[0], self.pos[1] + 5)
        if self.pos[0] <= 0 or self.pos[0] >= self.shape[0] - 1:
            return True
        else:
            return False

    def fire(self):
        # If the alien is chosen this is the bullet it creates, only one fires at a time
        return AlienBullet(self.gun_pos)

    def distance(self, pos):
        return abs(self.gun_pos[0] - pos[0]) + abs(self.gun_pos[1] - pos[1])

    def contains(self, pos):
        return self.pos[0] <= pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= pos[1] < self.pos[1] + self.size[1]

    def positions(self):
        return [(self.pos[0] + i, self.pos[1] + j) for i in range(self.size[0]) for j in range(self.size[1])]


class AlienBullet:
    def __init__(self, pos):
        self.pos = pos

    def step(self):
        self.pos = (self.pos[0], self.pos[1] + 1)


class Player:
    def __init__(self, shape, size=(2, 2)):
        self.pos = (0, shape[1] - size[1])
        self.board_shape = shape
        self.size = size
        self.lives = 3
        self.invulnerable = False
        self.invuln_frame = 0

    def contains(self, pos):
        return self.pos[0] <= pos[0] < self.pos[0] + self.size[0] and self.pos[1] <= pos[1] < self.pos[1] + self.size[1]

    def hit(self):
        self.lives -= 1
        self.invulnerable = True
        self.invuln_frame = 8

    def step(self, action):
        """
        :param action: Move or Fire, 0: Left, 1: Right, 2: Fire, 3: Noop
        :return: PlayerBullet if fire
        """
        if self.invulnerable:
            self.invuln_frame -= 1
            if self.invuln_frame == 0:
                self.invulnerable = False

        if action == 0:
            self.pos = (max(self.pos[0] - 1, 0), self.pos[1])
            return None
        elif action == 1:
            self.pos = (min(self.pos[0] + 1, self.board_shape[0] - self.size[0]), self.pos[1])
            return None
        elif action == 2:
            return self.fire()
        elif action == 3:
            return None

    def fire(self):
        return PlayerBullet((self.pos[0] + 1, self.pos[1] - 1))

    def positions(self):
        return [(self.pos[0] + i, self.pos[1] + j) for i in range(self.size[0]) for j in range(self.size[1])]


class PlayerBullet:
    def __init__(self, pos):
        self.pos = pos

    def step(self):
        self.pos = (self.pos[0], self.pos[1] - 1)


class SpaceInvaders(SaveLoadEnv):

    def load_state(self, state):
        self.num_enemies, self.num_bunkers, self.shape, self.alien_shape, self.player_shape, self.bunker_shape, \
        self.max_time, self.player, self.obs, self.aliens, self.alien_projectiles, self.player_projectiles, \
        self.bunkers, self.num_destroyed, self.dir_x, self.dir_y, self.timestep, self.time_before_fire = state

    def save_state(self):
        return self.num_enemies, self.num_bunkers, self.shape, self.alien_shape, self.player_shape, self.bunker_shape, \
               self.max_time, self.player, self.obs, self.aliens, self.alien_projectiles, self.player_projectiles, \
               self.bunkers, self.num_destroyed, self.dir_x, self.dir_y, self.timestep, self.time_before_fire

    def handle_enemy_collisions(self):
        collided_proj = []
        collided_enemy = []
        out_of_bounds = []

        # Goes through every projectile, checking if it hit an alien or is out of bounds
        for proj in self.player_projectiles:
            if proj.pos[1] < 0:
                out_of_bounds.append(proj)
            for enemy in self.aliens:
                if enemy.contains(proj.pos):
                    collided_proj.append(proj)
                    collided_enemy.append(enemy)
                    break

        # Remove the things that have been hit or are out of bounds
        for proj in collided_proj:
            if proj in self.player_projectiles:
                self.player_projectiles.remove(proj)
        for enemy in collided_enemy:
            if enemy in self.aliens:
                self.aliens.remove(enemy)
        for proj in out_of_bounds:
            if proj in self.player_projectiles:
                self.player_projectiles.remove(proj)
        return len(collided_enemy)

    def handle_bunker_collisions(self):
        # Used for checking in case the bullet is more than just one tile, though currently they are only one tile large
        enemy_bullet_heads = {}
        player_bullet_heads = {}

        for proj in self.alien_projectiles:
            enemy_bullet_heads[proj.pos] = proj
        for proj in self.player_projectiles:
            player_bullet_heads[proj.pos] = proj

        # Checks each bunker and sees if any have been hit, removing the corresponding bullet
        destroyed_bunker = []
        for pos in self.bunkers:
            for enemy in self.aliens:
                if enemy.contains(pos):
                    destroyed_bunker.append(pos)
            if pos in enemy_bullet_heads.keys():
                destroyed_bunker.append(pos)
                if (pos[0], pos[1] + 1) in self.bunkers:
                    destroyed_bunker.append((pos[0], pos[1] + 1))
                if enemy_bullet_heads[pos] in self.alien_projectiles:
                    self.alien_projectiles.remove(enemy_bullet_heads[pos])
            elif pos in player_bullet_heads:
                destroyed_bunker.append(pos)
                if (pos[0], pos[1] + 1) in self.bunkers:
                    destroyed_bunker.append((pos[0], pos[1] + 1))
                if player_bullet_heads[pos] in self.player_projectiles:
                    self.player_projectiles.remove(player_bullet_heads[pos])

        for val in destroyed_bunker:
            if val in self.bunkers:
                self.bunkers.remove(val)

    def handle_player_collisions(self):
        # Similar to others, check if the player is hit by a bullet and removes the bullet/damages player if it is
        hit_proj = None
        out_of_bounds = []

        for proj in self.alien_projectiles:
            if proj.pos[1] > self.shape[1] - 1:
                out_of_bounds.append(proj)
            if self.player.contains(proj.pos):
                hit_proj = proj

        for proj in out_of_bounds:
            if proj in self.alien_projectiles:
                self.alien_projectiles.remove(proj)

        if hit_proj is not None and not self.player.invulnerable:
            if hit_proj in self.alien_projectiles:
                self.alien_projectiles.remove(hit_proj)
            self.player.hit()
            return True

        return False

    def step_aliens(self):
        # Moves all the aliens, if they collided with a wall then move them down one and
        # reverse their left/right direction.
        against_wall = False
        for enemy in self.aliens:
            if not against_wall:
                against_wall = enemy.step(self.dir_x, self.dir_y)
            else:
                enemy.step(self.dir_x, self.dir_y)

        if against_wall:
            self.dir_x *= -1
            self.dir_y = 1
        elif self.dir_y != 0:
            self.dir_y = 0

    def aliens_fire(self):
        # Causes the alien closer to the player to fire more often, and also handles the firing logic
        front_aliens = {}
        for enemy in self.aliens:
            if enemy.gun_pos[0] not in front_aliens.keys():
                front_aliens[enemy.gun_pos[0]] = enemy
            elif front_aliens[enemy.gun_pos[0]].gun_pos[1] < enemy.gun_pos[1]:
                front_aliens[enemy.gun_pos[0]] = enemy

        front_aliens = list(front_aliens.values())
        distances = np.array([front_alien.distance(self.player.pos) for front_alien in front_aliens])
        distances = 3 / (distances + np.ones(shape=distances.shape))
        distances = np.nan_to_num(distances)
        probs = np.abs(distances) / np.sum(distances)
        firing_alien = np.random.choice(front_aliens, p=probs)
        self.alien_projectiles.append(firing_alien.fire())

    def step_proj(self):
        # Moves all the projectiles
        for proj in self.alien_projectiles:
            proj.step()
        for proj in self.player_projectiles:
            proj.step()

    def frequency(self):
        # Speeds up the aliens as more are destroyed.
        max_enemies = self.num_enemies[0] * self.num_enemies[1]
        if len(self.aliens) > max_enemies // 2:
            return 5
        elif len(self.aliens) > max_enemies // 4:
            return 2
        elif len(self.aliens) > 1:
            self.dir_x = 2 * np.sign(self.dir_x)
            return 2
        else:
            self.dir_x = 3 * np.sign(self.dir_x)
            return 1

    def step(self, action):
        self.timestep += 1
        self.step_proj()

        # Only has the aliens move/fire every so often
        if self.timestep % self.frequency() == 0:
            self.step_aliens()
        if self.timestep % 2 == 0:
            self.aliens_fire()

        # Player movement/firing, and only allowing the player to fire so often
        player_bullet = self.player.step(action)
        if player_bullet is not None and self.time_before_fire == 0:
            self.player_projectiles.append(player_bullet)
            self.time_before_fire = 2
        elif self.time_before_fire > 0:
            self.time_before_fire -= 1

        # Deals with collisions, and tracks some things for reward handling
        num_destroyed = self.handle_enemy_collisions()
        self.num_destroyed += num_destroyed
        player_hit = self.handle_player_collisions()
        self.handle_bunker_collisions()
        self.obs.append(self.get_state())

        reward = 0
        reward += 20 / (self.num_enemies[0] * self.num_enemies[1]) * num_destroyed
        done = False
        if self.num_destroyed == (self.num_enemies[0] * self.num_enemies[1]):
            reward += 20
            done = True
        if player_hit:
            reward -= 5
        if self.player.lives == 0:
            reward -= 20
        for enemy in self.aliens:
            if enemy.contains((enemy.pos[0], self.shape[1] - 1)):
                done = True
                reward -= 20
                break

        return np.array(self.obs, dtype=np.uint8), reward, self.player.lives == 0 or self.timestep == self.max_time or \
               done, {'aliens': self.aliens, 'alien_proj': self.alien_projectiles, 'bunkers': self.bunkers,
                      'player': self.player}

    def reset_aliens(self):
        # Resets all the aliens and makes them evenly spaced
        self.aliens = []
        self.dir_x = -1
        self.dir_y = 0
        every_i = max((self.shape[0] - int(0.1 * self.shape[0]) - self.alien_shape[0] * self.num_enemies[0])
                      // (self.num_enemies[0] - 1), self.alien_shape[0] + 1)
        every_j = max((self.shape[1] - int(0.3 * self.shape[1]) - self.alien_shape[1] * self.num_enemies[1])
                      // (self.num_enemies[1] - 1), self.alien_shape[1] + 1)
        i = int(0.07 * self.shape[0])
        for _ in range(self.num_enemies[0]):
            j = int(0.05 * self.shape[1])
            for _ in range(self.num_enemies[1]):
                self.aliens.append(Alien((i, j), self.shape, size=self.alien_shape))
                j += every_j
            i += every_i

    def reset_bunkers(self):
        # Resets the bunkers in an even pattern left/right
        self.bunkers = []
        spacing = (self.shape[0] - self.bunker_shape[1] * self.num_bunkers) // (self.num_bunkers + 1)
        i = 0
        for _ in range(self.num_bunkers):
            i += spacing
            for x in range(self.bunker_shape[0]):
                for y in range(self.bunker_shape[1]):
                    self.bunkers.append((i + x, self.shape[1] - (2 + self.player_shape[1] + self.bunker_shape[1]) + y))
            i += self.bunker_shape[0]

    def reset(self):
        self.reset_aliens()
        self.reset_bunkers()
        self.player = Player(self.shape, self.player_shape)
        self.obs.append(self.get_state())
        self.alien_projectiles = []
        self.player_projectiles = []
        self.num_destroyed = 0
        self.dir_x = -1
        self.dir_y = 0
        self.timestep = 0
        self.time_before_fire = 0
        return np.array(self.obs, dtype=np.uint8)

    def get_state(self):
        obs = [[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])]
        for pos in self.player.positions():
            obs[pos[1]][pos[0]] = 1
        for pos in self.bunkers:
            obs[pos[1]][pos[0]] = 2
        for enemy in self.aliens:
            for pos in enemy.positions():
                obs[pos[1]][pos[0]] = 3
        for proj in self.alien_projectiles:
            obs[proj.pos[1]][proj.pos[0]] = 4
        for proj in self.player_projectiles:
            obs[proj.pos[1]][proj.pos[0]] = 5
        obs[0][0] = self.player.lives

        return obs

    def render(self, mode="human"):
        obs = self.get_state()
        for row in obs:
            temp_str = ''
            for val in row:
                temp_str += str(val)
            print(temp_str)
        print()

    def __init__(self, config: Dict):
        self.num_enemies = config['enemies']
        self.num_bunkers = config['bunkers']
        self.shape = config['shape']
        self.alien_shape = config['alien_shape']
        self.player_shape = config['player_shape']
        self.bunker_shape = config['bunker_shape']
        self.max_time = config['max_time']
        self.player = Player(self.shape, size=self.player_shape)
        self.observation_space = spaces.Box(0, 5, shape=(4, self.shape[1], self.shape[0]), dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        self.obs = deque(maxlen=4)
        for _ in range(3):
            self.obs.append([[0 for _ in range(self.shape[0])] for _ in range(self.shape[1])])
        self.aliens = []
        self.alien_projectiles = []
        self.player_projectiles = []
        self.bunkers = []
        self.num_destroyed = 0
        self.dir_x = -1
        self.dir_y = 0
        self.timestep = 0
        self.time_before_fire = 0


if __name__ == '__main__':
    # Just used for testing/debugging code
    env = SpaceInvaders({'enemies': (6, 5), 'bunkers': 3, 'shape': (20, 20), 'alien_shape': (1, 1),
                         'player_shape': (2, 2), 'bunker_shape': (4, 4), 'max_time': 500})
    for i in range(50):
        print(i)
        env.reset()
        for _ in range(1000):
            if i == 27:
                pass
            # print(_)
            env.render()
            # next_state, rew, done, info = env.step(np.random.choice([0, 1, 2]))
            next_state, rew, done, info = env.step(int(input()))
            if done:
                break
