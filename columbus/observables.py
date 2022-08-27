from gym import spaces
import numpy as np
import pygame
import math
from columbus import entities
import torch as th


class Observable():
    def __init__(self):
        self.obs = None

    def _set_env(self, env):
        self.env = env

    def get_observation_space(self):
        print("[!] Using dummyObservable. Env won't output anything")
        return spaces.Box(low=0, high=1,
                          shape=(1,), dtype=np.float32)

    def get_observation(self):
        return np.array([0])

    def draw(self):
        pass

    def reset(self):
        pass


class CnnObservable(Observable):
    def __init__(self, in_width=256, in_height=256, out_width=32, out_height=32, draw_width=128, draw_height=128, smooth_scaling=True):
        super(CnnObservable, self).__init__()
        self.in_width = in_width
        self.in_height = in_height
        self.out_width = out_width
        self.out_height = out_height
        self.draw_width = draw_width
        self.draw_height = draw_height
        if smooth_scaling:
            self.scaler = pygame.transform.smoothscale
        else:
            self.scaler = pygame.transform.scale

    def get_observation_space(self):
        return spaces.Box(low=0, high=255,
                          shape=(self.out_width, self.out_height, 3), dtype=np.float32)

    def get_observation(self):
        if not self.env._rendered:
            self.env.render(dont_show=True)
        self.env._ensure_surface()
        x, y = self.env.agent.pos[0]*self.env.width - self.in_width / \
            2, self.env.agent.pos[1]*self.env.height - self.in_height/2
        w, h = self.in_width, self.in_height
        cx, cy = _clip(x, 0, self.env.width), _clip(
            y, 0, self.env.height)
        cw, ch = _clip(w, 0, self.env.width - cx), _clip(h,
                                                         0, self.env.height - cy)
        rect = pygame.Rect(cx, cy, cw, ch)
        snap = self.env.surface.subsurface(rect)
        self.snap = pygame.Surface((self.in_width, self.in_height))
        if self.env.void_barrier:
            col = (255, 0, 0)
        else:
            col = (50, 50, 50)
        pygame.draw.rect(self.snap, col,
                         pygame.Rect(0, 0, self.in_width, self.in_height))
        self.snap.blit(snap, (cx - x, cy - y))
        self.obs = self.scaler(
            self.snap, (self.out_width, self.out_height))
        arr = pygame.surfarray.array3d(self.obs)
        return arr

    def draw(self):
        if not self.obs:
            self.get_observation()
        big = pygame.transform.scale(
            self.obs, (self.draw_width, self.draw_height))
        x, y = self.env.width - self.draw_width - 10, 10
        pygame.draw.rect(self.env.screen, (50, 50, 50),
                         pygame.Rect(x - 1, y - 1, self.draw_width + 2, self.draw_height + 2))
        self.env.screen.blit(
            big, (x, y))


def _clip(num, lower, upper):
    return min(max(num, lower), upper)


class RayObservable(Observable):
    def __init__(self, num_rays=16, chans=[entities.Enemy, entities.Reward], ray_len=256, num_steps=64, include_rand=False):
        super(RayObservable, self).__init__()
        self.num_rays = num_rays
        self.chans = chans
        self.num_chans = len(chans)
        self.ray_len = ray_len
        self.num_steps = num_steps  # max = 255
        self.occlusion = True  # previous channels block view onto later channels
        self.include_rand = include_rand

    def get_observation_space(self):
        return spaces.Box(low=0, high=1,
                          shape=(self.num_rays+self.include_rand, self.num_chans), dtype=np.uint8)

    def _get_ray_heads(self):
        for i in range(self.num_rays):
            rad = 2*math.pi/self.num_rays*i
            yield self.ray_len*math.sin(rad), self.ray_len*math.cos(rad)

    def _check_collision(self, pos, entity_type, entities_l):
        for entity in entities_l:
            if isinstance(entity, entity_type) or (self.env.void_barrier and isinstance(entity, entities.Void) and entity_type == entities.Enemy):
                if isinstance(entity, entities.Void):
                    if not self.env.torus_topology and (0 >= pos[0] or pos[0] >= self.env.width or 0 >= pos[1] or pos[1] >= self.env.width):
                        return True
                else:
                    if entity.shape != 'circle':
                        raise Exception('Can only raycast circular entities!')
                    sq_dist = (pos[0]-entity.pos[0]*self.env.width) ** 2 \
                        + (pos[1]-entity.pos[1]*self.env.height)**2
                    if sq_dist < entity.radius**2:
                        return True
        return False

    def _get_possible_entities(self):
        entities_l = []
        if entities.Void in self.chans or self.env.void_barrier:
            entities_l.append(entities.Void(self.env))
        for entity in self.env.entities:
            sq_dist = ((self.env.agent.pos[0]-entity.pos[0])*self.env.width) ** 2 \
                + ((self.env.agent.pos[1]-entity.pos[1])*self.env.height) ** 2
            if sq_dist <= (entity.radius + self.env.agent.radius + self.ray_len)**2:
                entities_l.append(entity)  # cannot use yield here!
        return entities_l

    def get_observation(self):
        entities = self._get_possible_entities()
        self.rays = np.zeros((self.num_rays+self.include_rand, self.num_chans))
        if self.include_rand:
            for c in range(self.num_chans):
                self.rays[-1, c] = self.env.random()
        for r, (hx, hy) in enumerate(self._get_ray_heads()):
            occ_dist = self.num_steps
            for c, entity_type in enumerate(self.chans):
                for s in range(self.num_steps):
                    if s > occ_dist:
                        break
                    sx, sy = (s+1)*hx/self.num_steps, (s+1)*hy/self.num_steps
                    rx, ry = sx + \
                        self.env.agent.pos[0]*self.env.width, sy + \
                        self.env.agent.pos[1]*self.env.height
                    if self.env.torus_topology:
                        rx, ry = rx % self.env.width, ry % self.env.height
                    if self._check_collision((rx, ry), entity_type, entities):
                        self.rays[r, c] = (self.num_steps-s)/self.num_steps
                        if self.occlusion:
                            occ_dist = s
                        break
        return self.rays

    def draw(self):
        for c, entity_type in enumerate(self.chans):
            for r, (hx, hy) in enumerate(self._get_ray_heads()):
                s = self.num_steps - self.rays[r, c]*self.num_steps
                sx, sy = (s+1)*hx/self.num_steps, (s+1)*hy/self.num_steps
                rx, ry = sx + \
                    self.env.agent.pos[0]*self.env.width, sy + \
                    self.env.agent.pos[1]*self.env.height
                if self.env.torus_topology:
                    rx, ry = rx % self.env.width, ry % self.env.height
                # TODO: How stupid do I want to code?
                # This instanciates an Object for every Ray-hit,
                # just to get the color for the visual.
                # But since this Code will not be executed during training,
                # I don't think fixing this is an priority...
                col = entity_type(self.env).col
                col = int(col[0]/2), int(col[1]/2), int(col[2]/2)
                pygame.draw.circle(self.env.screen, col, (rx, ry), 3, width=0)


class StateObservable(Observable):
    def __init__(self, coordsAgent=False, speedAgent=False, coordsRelativeToAgent=True, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=True, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=True, include_rand=True):
        super(StateObservable, self).__init__()
        self._entities = None
        self._timeoutEntities = []
        self.coordsAgent = coordsAgent
        self.speedAgent = speedAgent
        self.coordsRelativeToAgent = coordsRelativeToAgent
        self.coordRewards = coordsRewards
        self.rewardsWhitelist = rewardsWhitelist
        self.coordsEnemys = coordsEnemys
        self.enemysWhitelist = enemysWhitelist
        self.enemysNoBarriers = enemysNoBarriers
        self.rewardsTimeouts = rewardsTimeouts
        self.include_rand = include_rand

    @property
    def entities(self):
        if not self._entities == None:
            return self._entities
        rewardsWhitelist = self.rewardsWhitelist or self.env.entities
        enemysWhitelist = self.enemysWhitelist or self.env.entities
        self._entities = []
        if self.coordsAgent:
            self._entities.append(self.env.agent)
        if self.coordRewards:
            for entity in rewardsWhitelist:
                if isinstance(entity, entities.Reward):
                    self._entities.append(entity)
        if self.coordsEnemys:
            for entity in enemysWhitelist:
                if isinstance(entity, entities.Enemy):
                    if not self.enemysNoBarriers or not isinstance(entity, entities.Barrier):
                        self._entities.append(entity)
        if self.rewardsTimeouts:
            for entity in enemysWhitelist:
                if isinstance(entity, entities.TimeoutReward):
                    self._timeoutEntities.append(entity)
        return self._entities

    def reset(self):
        self._entities = None

    def get_observation_space(self):
        self.reset()
        num = len(self.entities)*2+len(self._timeoutEntities) + \
            self.speedAgent*2 + self.include_rand
        return spaces.Box(low=0-1*self.coordsRelativeToAgent, high=1,
                          shape=(num,), dtype=np.float32)

    def get_observation(self):
        obs = []
        if self.coordsRelativeToAgent:
            for entity in self.entities:
                if not isinstance(entity, entities.Agent):
                    obs.append(entity.pos[0] - self.env.agent.pos[0])
                    obs.append(entity.pos[1] - self.env.agent.pos[1])
                else:
                    obs.append(entity.pos[0])
                    obs.append(entity.pos[1])
        else:
            for entity in self.entities:
                obs.append(entity.pos[0])
                obs.append(entity.pos[1])

        for entity in self._timeoutEntities:
            obs.append(entity.active)
        if self.speedAgent:
            obs.append(self.env.agent.speed[0])
            obs.append(self.env.agent.speed[1])
        if self.include_rand:
            obs.append(self.env.random())
        self.obs = obs
        return np.array(obs)

    def draw(self):
        ofs = (0 + self.env.height/2*self.coordsRelativeToAgent,
               0 + self.env.width/2*self.coordsRelativeToAgent)
        if self.coordsRelativeToAgent:
            pygame.draw.circle(self.env.screen, self.env.agent.col,
                               (0, self.env.height/2), 3, width=0)
            pygame.draw.circle(self.env.screen, self.env.agent.col,
                               (self.env.width/2, 0), 3, width=0)
        for i in range(int(len(self.obs)/2) - self.speedAgent):
            x, y = self.obs[i*2], self.obs[i*2+1]
            col = self.entities[i].col
            pygame.draw.circle(self.env.screen, col,
                               (0, y*self.env.height+ofs[0]), 1, width=0)
            pygame.draw.circle(self.env.screen, col,
                               (x*self.env.width+ofs[1], 0), 1, width=0)


class CompassObservable(Observable):
    def __init__(self, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=False, enemysWhitelist=None, enemysNoBarriers=True):
        super().__init__()
        self._entities = None
        self._timeoutEntities = []
        self.coordRewards = coordsRewards
        self.rewardsWhitelist = rewardsWhitelist
        self.coordsEnemys = coordsEnemys
        self.enemysWhitelist = enemysWhitelist
        self.enemysNoBarriers = enemysNoBarriers

    @property
    def entities(self):
        if not self._entities == None:
            return self._entities
        rewardsWhitelist = self.rewardsWhitelist or self.env.entities
        enemysWhitelist = self.enemysWhitelist or self.env.entities
        self._entities = []
        if self.coordRewards:
            for entity in rewardsWhitelist:
                if isinstance(entity, entities.Reward):
                    self._entities.append(entity)
        if self.coordsEnemys:
            for entity in enemysWhitelist:
                if isinstance(entity, entities.Enemy):
                    if not self.enemysNoBarriers or not isinstance(entity, entities.Barrier):
                        self._entities.append(entity)
        return self._entities

    def get_observation_space(self):
        self.reset()
        num = len(self.entities)*2
        return spaces.Box(low=-1, high=1,
                          shape=(num,), dtype=np.float32)

    def reset(self):
        self._entities = None

    def get_observation(self):
        obs = []
        for entity in self.entities:
            dx, dy = entity.pos[0] - \
                self.env.agent.pos[0], entity.pos[1] - self.env.agent.pos[1]
            l = math.sqrt(dx**2 + dy**2)*2
            x, y = math.tanh(dx/l), math.tanh(dy/l)
            obs.append(x)
            obs.append(y)

        self.obs = obs
        return np.array(obs)

    def draw(self):
        ofs = (0 + self.env.height/2,
               0 + self.env.width/2)
        if True:
            pygame.draw.circle(self.env.screen, self.env.agent.col,
                               (0, self.env.height/2), 3, width=0)
            pygame.draw.circle(self.env.screen, self.env.agent.col,
                               (self.env.width/2, 0), 3, width=0)
        for i in range(int(len(self.obs)/2)):
            x, y = self.obs[i*2], self.obs[i*2+1]
            col = self.entities[i].col
            pygame.draw.circle(self.env.screen, col,
                               (0, y*self.env.height+ofs[0]), 1, width=0)
            pygame.draw.circle(self.env.screen, col,
                               (x*self.env.width+ofs[1], 0), 1, width=0)


class CompositionalObservable(Observable):
    def __init__(self, observables):
        super().__init__()
        self.observables = observables

    def get_observation_space(self):
        num = 0
        for i, obs in enumerate(self.observables):
            space = obs.get_observation_space()
            num += math.prod(space.shape)
            if not i:
                low = space.low.reshape((-1))
                high = space.high.reshape((-1))
            else:
                low = np.hstack((low, space.low.reshape((-1))))
                high = np.hstack((high, space.high.reshape((-1))))
        return spaces.Box(low=low, high=high,
                          shape=(num,), dtype=np.float32)

    def get_observation(self):
        o = [obs.get_observation().reshape((-1))
             for obs in self.observables]
        o = np.hstack(o)
        return o

    def draw(self):
        for obs in self.observables:
            obs.draw()

    def _set_env(self, env):
        for obs in self.observables:
            obs._set_env(env)

    def reset(self):
        for obs in self.observables:
            obs.reset()
