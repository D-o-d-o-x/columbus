from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import pygame
import random as random_dont_use
from os import urandom
import math
from columbus import entities, observables
import torch as th


class ColumbusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observable=observables.Observable(), fps=60, env_seed=3.1):
        super(ColumbusEnv, self).__init__()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
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
        self.aux_penalty_max = 0  # 0 = off
        self.aux_reward_discretize = 0  # 0 = dont discretize
        self.draw_observable = True
        self.draw_joystick = True
        self.draw_entities = True
        self.draw_confidence_ellipse = True
        self.void_barrier = True
        self.void_damage = 100

        self.paused = False
        self.keypress_timeout = 0
        self.can_accept_chol = True
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
            elif isinstance(entity, entities.Enemy):
                if entity.radiateDamage:
                    penalty = self.aux_penalty_max / \
                        (1 + self.sq_dist(entity.pos, self.agent.pos))

                    if self.aux_reward_discretize:
                        penalty = int(penalty*self.aux_reward_discretize*2) / \
                            self.aux_reward_discretize / 2

                    aux_reward -= penalty
        return aux_reward/self.fps

    def step(self, action):
        # TODO: Just make the range consistent...
        inp = (action[0]+1)/2, (action[1]+1)/2
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
                depth = self._check_collision_between(entity, other)
                if depth > 0:
                    entity.on_collision(other, depth)
                    other.on_collision(entity, depth)

    def _check_collision_between(self, e1, e2):
        shapes = [e1.shape, e2.shape]
        shapes.sort()
        if shapes == ['circle', 'circle']:
            dist = math.sqrt(((e1.pos[0]-e2.pos[0])*self.width) ** 2
                             + ((e1.pos[1]-e2.pos[1])*self.height)**2)
            return max(0, e1.radius + e2.radius - dist)
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

    def _draw_confidence_ellipse(self, chol, forceDraw=False, seconds=1):
        if (self.draw_confidence_ellipse or forceDraw) and self.visible:
            col = (255, 255, 255)
            f = seconds/self.speed_fac

            while len(chol.shape) > 2:
                chol = chol[0]
            if chol.shape != (2, 2):
                chol = th.diag_embed(chol)
            if len(chol.shape) != 2:
                chol = chol[0]
            cov = chol.T @ chol

            L, V = th.linalg.eig(cov)
            L, V = L.real, V.real
            w, h = int(abs(L[0].item()*f))+1, int(abs(L[1].item()*f))+1
            # TODO: Is this correct? We try to solve for teh angle from this:
            # R = [[cos, -sin],[sin, cos]]
            # Via only the -sin term.
            #ang1 = int(math.acos(V[0, 0])/math.pi*360)
            ang2 = int(math.asin(-V[0, 1])/math.pi*360)
            #ang3 = int(math.asin(V[1, 0])/math.pi*360)
            ang = ang2

            # print(cov)
            #print(w, h, (ang1, ang2, ang3))

            x, y = self.agent.pos
            x, y = x*self.width, y*self.height
            rect = pygame.Rect((x-w/2, y-h/2, w, h))
            shape_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shape_surface, col,
                                (0, 0, *rect.size), 1)
            rotated_surf = pygame.transform.rotate(shape_surface, ang)
            self.screen.blit(rotated_surf, rotated_surf.get_rect(
                center=rect.center))

    def _handle_user_input(self):
        for event in pygame.event.get():
            pass
        keys = pygame.key.get_pressed()
        if self.keypress_timeout == 0:
            self.keypress_timeout = int(self.fps/5)
            if keys[pygame.K_m]:
                self.draw_entities = not self.draw_entities
            elif keys[pygame.K_c]:
                self.draw_confidence_ellipse = not self.draw_confidence_ellipse
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

    def render(self, mode='human', dont_show=False, chol=None):
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
        if chol != None:
            self._draw_confidence_ellipse(chol)
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


class ColumbusCandyland_Aux10(ColumbusCandyland):
    def __init__(self, fps=30):
        super(ColumbusCandyland_Aux10, self).__init__(fps=fps)
        self.aux_reward_max = 10


class ColumbusEasyObstacles(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16), hide_map=False, fps=30, env_seed=None):
        super(ColumbusEasyObstacles, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.draw_entities = not hide_map
        self.aux_reward_max = 10

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
        self.aux_reward_max = 10

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


class ColumbusComp(ColumbusEnv):
    def __init__(self, observable=observables.CompositionalObservable([observables.RayObservable(num_rays=6, chans=[entities.Enemy]), observables.StateObservable(coordsAgent=True, speedAgent=False, coordsRelativeToAgent=False, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=False, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True)]), hide_map=False, fps=30, env_seed=None):
        super().__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.draw_entities = not hide_map
        self.aux_reward_max = 10

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


class ColumbusJustState(ColumbusEnv):
    def __init__(self, observable=observables.StateObservable(), fps=30, num_enemies=0, num_rewards=1, env_seed=None):
        super(ColumbusJustState, self).__init__(
            observable=observable,  fps=fps)
        self.aux_reward_max = 10
        self.num_enemies = num_enemies
        self.num_rewards = num_rewards

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(self.num_enemies):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4+0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(self.num_rewards):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            self.entities.append(reward)


class ColumbusStateWithBarriers(ColumbusEnv):
    def __init__(self, observable=observables.StateObservable(coordsAgent=True, speedAgent=False, coordsRelativeToAgent=False, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=True, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True), fps=30, env_seed=3.141, num_enemys=0, num_barriers=3):
        super(ColumbusStateWithBarriers, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed)
        self.aux_reward_max = 10
        self.start_pos = (0.5, 0.5)
        self.num_barriers = num_barriers
        self.num_enemys = num_enemys

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(self.num_barriers):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*25+75
            self.entities.append(enemy)
        for i in range(self.num_enemys):
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


class ColumbusFootball(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16, chans=[entities.Goal, entities.Ball, entities.Barrier]), fps=30, walkingOpponent=0, flyingOpponent=0):
        super(ColumbusFootball, self).__init__(
            observable=observable, fps=fps, env_seed=None)
        self.start_pos = [0.5, 0.5]
        self.score = 0
        self.walkingOpponents = walkingOpponent
        self.flyingOpponents = flyingOpponent

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(8):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*40+50
            self.entities.append(enemy)
        ball = entities.Ball(self)
        self.entities.append(ball)
        self.entities.append(entities.TeleportingGoal(self))
        for i in range(self.walkingOpponents):
            self.entities.append(entities.WalkingFootballPlayer(self, ball))
        for i in range(self.flyingOpponents):
            self.entities.append(entities.FlyingFootballPlayer(self, ball))


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
    id='ColumbusCandyland_Aux10-v0',
    entry_point=ColumbusCandyland_Aux10,
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
    id='ColumbusJustState-v0',
    entry_point=ColumbusJustState,
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

register(
    id='ColumbusFootball-v0',
    entry_point=ColumbusFootball,
    max_episode_steps=30*60*2,
)

register(
    id='ColumbusComb-v0',
    entry_point=ColumbusComp,
    max_episode_steps=30*60*2,
)
