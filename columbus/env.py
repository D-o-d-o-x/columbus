from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import pygame
import random as random_dont_use
from os import urandom
import math
from columbus import entities, observables


class ColumbusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observable=observables.Observable(), fps=60, env_seed=3.1):
        super(ColumbusEnv, self).__init__()
        self.action_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32)
        observable._set_env(self)
        self.observable = observable
        self.title = 'Untitled'
        self.fps = fps
        self.env_seed = env_seed
        self.joystick_offset = (10, 10)
        self.surface = None
        self.screen = None
        self.width = 720
        self.height = 720
        self.visible = False
        self.start_pos = (0.5, 0.5)
        self.speed_fac = 0.01/fps*60
        self.acc_fac = 0.03/fps*60
        self.die_on_zero = False
        self.return_on_score = -1  # -1 = never
        self.reward_mult = 1
        self.agent_drag = 0  # 0.01 is a good value
        self.controll_type = 'SPEED'  # one of SPEED, ACC
        self.limit_inp_to_unit_circle = True
        self.aux_reward_max = 0  # 0 = off
        self.aux_reward_discretize = 0  # 0 = dont discretize
        self.draw_observable = True
        self.draw_joystick = True
        self.draw_entities = True
        self.void_barrier = True
        self.void_damage = 100

        self.paused = False
        self.keypress_timeout = 0
        self.rng = random_dont_use.Random()
        self._seed(self.env_seed)

    @property
    def observation_space(self):
        return self.observable.get_observation_space()

    def _seed(self, seed):
        if seed == None:
            seed = urandom(12)
        self.rng.seed(seed)

    def random(self):
        return self.rng.random()

    def _ensure_surface(self):
        if not self.surface or self.visible and not self.screen:
            self.surface = pygame.Surface((self.width, self.height))
            if self.visible:
                self.screen = pygame.display.set_mode(
                    (self.width, self.height))
            pygame.display.set_caption(self.title)

    def _limit_to_unit_circle(self, coords):
        l_sq = coords[0]**2 + coords[1]**2
        if l_sq > 1:
            l = math.sqrt(l_sq)
            coords = coords[0] / l, coords[1] / l
        return coords

    def _step_entities(self):
        for entity in self.entities:
            entity.step()

    def _step_timers(self):
        new_timers = []
        for time_left, func, arg in self.timers:
            time_left -= 1/self.fps
            if time_left < 0:
                func(arg)
            else:
                new_timers.append((time_left, func, arg))
        self.timers = new_timers

    def sq_dist(self, pos1, pos2):
        return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2

    def dist(self, pos1, pos2):
        return math.sqrt(self.sq_dist(pos1, pos2))

    def _get_aux_reward(self):
        aux_reward = 0
        for entity in self.entities:
            if isinstance(entity, entities.Reward):
                if entity.avaible:
                    reward = self.aux_reward_max / \
                        (1 + self.sq_dist(entity.pos, self.agent.pos))

                    if self.aux_reward_discretize:
                        reward = int(reward*self.aux_reward_discretize*2) / \
                            self.aux_reward_discretize / 2

                    aux_reward += reward
        return aux_reward

    def step(self, action):
        inp = action[0], action[1]
        if self._disturb_next:
            inp = self._disturb_next
            self._disturb_next = False
        if self.limit_inp_to_unit_circle:
            inp = self._limit_to_unit_circle(((inp[0]-0.5)*2, (inp[1]-0.5)*2))
            inp = (inp[0]+1)/2, (inp[1]+1)/2
        self.inp = inp
        if not self.paused:
            self._step_timers()
            self._step_entities()
        observation = self.observable.get_observation()
        reward, self.new_reward, self.new_abs_reward = self.new_reward / \
            self.fps + self.new_abs_reward, 0, 0
        self.score += reward  # aux_reward does not count towards the score
        if self.agent.pos[0] < 0.001 or self.agent.pos[0] > 0.999 \
                or self.agent.pos[1] < 0.001 or self.agent.pos[1] > 0.999:
            reward -= self.void_damage/self.fps
        if self.aux_reward_max:
            reward += self._get_aux_reward()
        done = self.die_on_zero and self.score <= 0 or self.return_on_score != - \
            1 and self.score > self.return_on_score
        info = {'score': self.score, 'reward': reward}
        self._rendered = False
        return observation, reward*self.reward_mult, done, info

    def check_collisions_for(self, entity):
        for other in self.entities:
            if other != entity:
                if self._check_collision_between(entity, other):
                    entity.on_collision(other)
                    other.on_collision(entity)

    def _check_collision_between(self, e1, e2):
        shapes = [e1.shape, e2.shape]
        shapes.sort()
        if shapes == ['circle', 'circle']:
            sq_dist = ((e1.pos[0]-e2.pos[0])*self.width) ** 2 \
                + ((e1.pos[1]-e2.pos[1])*self.height)**2
            return sq_dist < (e1.radius + e2.radius)**2
        else:
            raise Exception(
                'Checking for collision between unsupported shapes: '+str(shapes))

    def kill_entity(self, target):
        newEntities = []
        for entity in self.entities:
            if target != entity:
                newEntities.append(entity)
            else:
                del target
                break
        self.entities = newEntities

    def setup(self):
        self.agent.pos = self.start_pos
        # Expand this function

    def reset(self):
        pygame.init()
        self._seed(self.env_seed)
        self._rendered = False
        self._disturb_next = False
        self.inp = (0.5, 0.5)
        # will get rescaled acording to fps (=reward per second)
        self.new_reward = 0
        self.new_abs_reward = 0  # will not get rescaled. should be used for one-time rewards
        self.score = 0
        self.entities = []
        self.timers = []
        self.agent = entities.Agent(self)
        self.setup()
        self.entities.append(self.agent)  # add it last, will be drawn on top
        self.observable._entities = None
        return self.observable.get_observation()

    def _draw_entities(self):
        for entity in self.entities:
            entity.draw()

    def _draw_observable(self, forceDraw=False):
        if (self.draw_observable or forceDraw) and self.visible:
            self.observable.draw()

    def _draw_joystick(self, forceDraw=False):
        if (self.draw_joystick or forceDraw) and self.visible:
            x, y = self.inp
            bigcol = (100, 100, 100)
            smolcol = (100, 100, 100)
            if self._disturb_next:
                smolcol = (255, 255, 255)
            pygame.draw.circle(self.screen, bigcol, (50 +
                                                     self.joystick_offset[0], 50+self.joystick_offset[1]), 50, width=1)
            pygame.draw.circle(self.screen, smolcol, (20+int(60*x) +
                                                      self.joystick_offset[0], 20+int(60*y)+self.joystick_offset[1]), 20, width=0)

    def _handle_user_input(self):
        for event in pygame.event.get():
            pass
        keys = pygame.key.get_pressed()
        if self.keypress_timeout == 0:
            self.keypress_timeout = int(self.fps/5)
            if keys[pygame.K_m]:
                self.draw_entities = not self.draw_entities
            elif keys[pygame.K_r]:
                self.reset()
            elif keys[pygame.K_p]:
                self.paused = not self.paused
            else:
                self.keypress_timeout = 0
        else:
            self.keypress_timeout -= 1

        # keys, that can be hold down to continously trigger them
        if keys[pygame.K_q]:
            self._disturb_next = (
                random_dont_use.random(), random_dont_use.random())
        elif keys[pygame.K_w]:
            self._disturb_next = (0.5, 0.0)
        elif keys[pygame.K_a]:
            self._disturb_next = (0.0, 0.5)
        elif keys[pygame.K_s]:
            self._disturb_next = (0.5, 1.0)
        elif keys[pygame.K_d]:
            self._disturb_next = (1.0, 0.5)

    def render(self, mode='human', dont_show=False):
        self._handle_user_input()
        self.visible = self.visible or not dont_show
        self._ensure_surface()
        pygame.draw.rect(self.surface, (0, 0, 0),
                         pygame.Rect(0, 0, self.width, self.height))
        if self.draw_entities:
            self._draw_entities()
        else:
            self.agent.draw()
        self._rendered = True
        if dont_show:
            return
        self.screen.blit(self.surface, (0, 0))
        self._draw_observable()
        self._draw_joystick()
        if self.visible:
            pygame.display.update()

    def close(self):
        pygame.display.quit()
        pygame.quit()


class ColumbusTest3_1(ColumbusEnv):
    def __init__(self, observable=observables.CnnObservable(out_width=48, out_height=48), fps=30):
        super(ColumbusTest3_1, self).__init__(
            observable=observable, fps=fps, env_seed=3.1)
        self.start_pos = [0.6, 0.3]
        self.score = 0
        self.aux_reward_max = 1

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(18):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*40+50
            self.entities.append(enemy)
        for i in range(3):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4*0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(0):
            reward = entities.TimeoutReward(self)
            self.entities.append(reward)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            self.entities.append(reward)


class ColumbusTestRay(ColumbusTest3_1):
    def __init__(self, observable=observables.RayObservable(), hide_map=False, fps=30):
        super(ColumbusTestRay, self).__init__(
            observable=observable, fps=fps)
        self.draw_entities = not hide_map


class ColumbusRayDrone(ColumbusTestRay):
    def __init__(self, observable=observables.RayObservable(), hide_map=False, fps=30):
        super(ColumbusRayDrone, self).__init__(
            observable=observable, hide_map=hide_map,  fps=fps)
        self.controll_type = 'ACC'
        self.agent_drag = 0.02


class ColumbusCandyland(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(chans=[entities.Reward, entities.Void], num_rays=16, include_rand=True), hide_map=False, fps=30, env_seed=None):
        super(ColumbusCandyland, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.draw_entities = not hide_map

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(0):
            reward = entities.TimeoutReward(self)
            reward.radius = 30
            self.entities.append(reward)
        for i in range(2):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            self.entities.append(reward)


class ColumbusEasyObstacles(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16), hide_map=False, fps=30, env_seed=None):
        super(ColumbusEasyObstacles, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.draw_entities = not hide_map
        self.aux_reward_max = 0.1

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(5):
            enemy = entities.CircleBarrier(self)
            enemy.radius = 30 + self.random()*70
            self.entities.append(enemy)
        for i in range(2):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            self.entities.append(reward)
        for i in range(1):
            enemy = entities.WalkingChaser(self)
            enemy.chase_speed = 0.20
            self.entities.append(enemy)


class ColumbusEasierObstacles(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16), hide_map=False, fps=30, env_seed=None):
        super(ColumbusEasierObstacles, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.draw_entities = not hide_map
        self.aux_reward_max = 0.5

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(5):
            enemy = entities.CircleBarrier(self)
            enemy.radius = 30 + self.random()*70
            self.entities.append(enemy)
        for i in range(3):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            reward.reward *= 2
            self.entities.append(reward)
        for i in range(1):
            enemy = entities.WalkingChaser(self)
            enemy.chase_speed = 0.20
            self.entities.append(enemy)


class ColumbusJustState(ColumbusEnv):
    def __init__(self, observable=observables.StateObservable(), fps=30, env_seed=None):
        super(ColumbusJustState, self).__init__(
            observable=observable,  fps=fps)
        self.aux_reward_max = 0.1

    def setup(self):
        self.agent.pos = self.start_pos
        # for i in range(2):
        #    enemy = entities.WalkingChaser(self)
        #    self.entities.append(enemy)
        for i in range(3):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4+0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            self.entities.append(reward)


class ColumbusStateWithBarriers(ColumbusEnv):
    def __init__(self, observable=observables.StateObservable(coordsAgent=True, speedAgent=False, coordsRelativeToAgent=False, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=True, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True), fps=30, env_seed=3.141, num_chasers=1):
        super(ColumbusStateWithBarriers, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.aux_reward_max = 10
        self.start_pos = (0.5, 0.5)
        self.num_chasers = num_chasers

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(3):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*25+75
            self.entities.append(enemy)
        for i in range(self.num_chasers):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = 0.55  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            self.entities.append(reward)


class ColumbusTrivialRay(ColumbusStateWithBarriers):
    def __init__(self, observable=observables.RayObservable(num_rays=8, ray_len=512), hide_map=False, fps=30):
        super(ColumbusTrivialRay, self).__init__(
            observable=observable, fps=fps, num_chasers=0)
        self.draw_entities = not hide_map


###
register(
    id='ColumbusTestCnn-v0',
    entry_point=ColumbusTest3_1,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusTestRay-v0',
    entry_point=ColumbusTestRay,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusRayDrone-v0',
    entry_point=ColumbusRayDrone,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusCandyland-v0',
    entry_point=ColumbusCandyland,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusEasyObstacles-v0',
    entry_point=ColumbusEasyObstacles,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusEasierObstacles-v0',
    entry_point=ColumbusEasyObstacles,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusStateWithBarriers-v0',
    entry_point=ColumbusStateWithBarriers,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusTrivialRay-v0',
    entry_point=ColumbusTrivialRay,
    max_episode_steps=30*60*2,
)
