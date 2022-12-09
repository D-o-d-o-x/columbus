from gym.envs.registration import register
import gym
from gym import spaces
import numpy as np
import pygame
import random as random_dont_use
from os import urandom
import math
import torch as th

from columbus import entities, observables
from columbus.utils import soft_int, parseObs


class ColumbusEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, observable=observables.Observable(), fps=60, env_seed=3.1, master_seed=None, start_pos=(0.5, 0.5), start_score=0, speed_fac=0.01, acc_fac=0.04, die_on_zero=False, return_on_score=-1, reward_mult=1, agent_drag=0, controll_type='SPEED', aux_reward_max=1, aux_penalty_max=0, aux_reward_discretize=0, void_is_type_barrier=True, void_damage=1, torus_topology=False, default_collision_elasticity=1, terminate_on_reward=False, agent_draw_path=False, clear_path_on_reset=True, max_steps=-1, value_color_mapper='tanh', width=720, height=720, agent_attrs={}):
        super(ColumbusEnv, self).__init__()
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32)
        if not isinstance(observable, observables.Observable):
            observable = parseObs(observable)
        observable._set_env(self)
        self.observable = observable
        self.title = 'Columbus Env'
        self.fps = fps
        self.env_seed = env_seed
        self.joystick_offset = (10, 10)
        self.surface = None
        self.screen = None
        self.width = width
        self.height = height
        self.visible = False
        self.start_pos = start_pos
        self.speed_fac = speed_fac/fps*60
        self.acc_fac = acc_fac/fps*60
        self.die_on_zero = die_on_zero  # return (/die) when score hist zero
        self.return_on_score = return_on_score  # -1 = Never
        self.reward_mult = reward_mult
        self.start_score = start_score
        # 0.01 is a good value, drag with the environment (air / ground)
        self.agent_drag = agent_drag
        assert controll_type == 'SPEED' or controll_type == 'ACC'
        self.limit_inp_to_unit_circle = True
        self.controll_type = controll_type  # one of SPEED, ACC
        self.aux_reward_max = aux_reward_max  # 0 = off
        self.aux_penalty_max = aux_penalty_max  # 0 = off
        self.aux_reward_discretize = aux_reward_discretize
        # 0 = dont discretize; how many steps (along diagonal)
        self.penalty_from_edges = True  # Don't change, only here to allow legacy behavior
        self.draw_observable = True
        self.draw_joystick = True
        self.draw_entities = True
        self.draw_confidence_ellipse = True
        # If the Void should be of type Barrier (else it is just of type Void and Entity)
        self.void_barrier = void_is_type_barrier
        self.void_damage = void_damage
        self.torus_topology = torus_topology
        self.default_collision_elasticity = default_collision_elasticity
        self.terminate_on_reward = terminate_on_reward
        self.agent_draw_path = agent_draw_path
        self.clear_path_on_reset = clear_path_on_reset
        self.path_decay = 0.1

        self.agent_attrs = agent_attrs

        if value_color_mapper == 'atan':
            def value_color_mapper(x): return th.atan(x*2)/0.786/2
        elif value_color_mapper == 'tanh':
            def value_color_mapper(x): return th.tanh(x*2)/0.762/2
        self.value_color_mapper = value_color_mapper

        self.max_steps = max_steps
        self._steps = 0
        self._has_value_map = False

        self.paused = False
        self.keypress_timeout = 0
        self.can_accept_chol = True
        self._master_rng = random_dont_use.Random()
        if master_seed == None:
            master_seed = urandom(12)
        if master_seed == 'numpy':
            master_seed = np.random.rand()
        self._master_rng.seed(master_seed)
        self.rng = random_dont_use.Random()
        self._seed(self.env_seed)

        self._init = False

    @property
    def observation_space(self):
        if not self._init:
            self.reset()
        return self.observable.get_observation_space()

    def _seed(self, seed):
        if seed == None:
            seed = self._master_rng.random()
        self.rng.seed(seed)

    def random(self):
        return self.rng.random()

    def _ensure_surface(self):
        if not self.surface or not self.screen:
            self.surface = pygame.Surface((self.width, self.height))
            self.path_overlay = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA, 32)
            self.value_overlay = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA, 32)
            if self.visible:
                self.screen = pygame.display.set_mode(
                    (self.width, self.height))
                pygame.display.set_caption(self.title)
            else:
                self.screen = pygame.Surface((self.width, self.height))

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
                    if self.penalty_from_edges:
                        if self.agent.shape != 'circle':
                            raise Exception(
                                'Radiating damage from edge for non-circle Agents not supported')
                        if entity.shape == 'circle':
                            penalty = self.aux_penalty_max / \
                                (1 + self.sq_dist(entity.pos,
                                                  self.agent.pos) - (entity.radius/max(self.height, self.width))**2 - (self.agent.radius/max(self.height, self.width))**2)
                        elif entity.shape == 'rect':
                            ax, ay = self.agent.pos
                            ex, ey, ex2, ey2 = entity.pos[0], entity.pos[1], entity.pos[0] + \
                                entity.width / \
                                self.width, entity.pos[1] + \
                                entity.height/self.height
                            lx, ly = ax, ay  # 'Lotpunkt'
                            if ax < ex:
                                lx = ex
                            elif ax > ex2:
                                lx = ex2
                            if ay < ey:
                                ly = ey
                            elif ay > ey2:
                                ly = ey2
                            penalty = self.aux_penalty_max / \
                                (1 + self.sq_dist((lx, ly),
                                                  (ax, ay)) - (self.agent.radius/max(self.height, self.width))**2)

                    else:
                        penalty = self.aux_penalty_max / \
                            (1 + self.sq_dist(entity.pos, self.agent.pos))

                    if self.aux_reward_discretize:
                        penalty = int(penalty*self.aux_reward_discretize*2) / \
                            self.aux_reward_discretize / 2

                    aux_reward -= penalty
        return aux_reward/self.fps

    def step(self, action):
        if not self._init:
            self.reset()
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
        gotRew = self.new_reward > 0 or self.new_abs_reward > 0
        self.gotHarm = self.new_reward < 0 or self.new_abs_reward < 0
        reward, self.new_reward, self.new_abs_reward = self.new_reward / \
            self.fps + self.new_abs_reward, 0, 0
        if not self.torus_topology:
            if self.agent.pos[0] < 0.001 or self.agent.pos[0] > 0.999 \
                    or self.agent.pos[1] < 0.001 or self.agent.pos[1] > 0.999:
                reward -= self.void_damage/self.fps
        self.score += reward  # aux_reward does not count towards the score
        if self.aux_reward_max or self.aux_penalty_max:
            reward += self._get_aux_reward()
        self._steps += 1
        done = (self.die_on_zero and self.score <= 0) or (self.return_on_score != -
                                                          1 and self.score > self.return_on_score) or (self._steps == self.max_steps) or (self.terminate_on_reward and gotRew)
        info = {'score': self.score, 'reward': reward}
        self._rendered = False
        if done:
            self.reset()
        return observation, reward*self.reward_mult, done, info

    def check_collisions_for(self, entity):
        for other in self.entities:
            if other != entity:
                depth = self._check_collision_between(entity, other)
                if depth > 0:
                    entity.on_collision(other, depth)
                    other.on_collision(entity, depth)

    def _check_collision_between(self, e1, e2):
        e = [e1, e2]
        e.sort(key=lambda x: x.shape)
        e1, e2 = e
        shapes = [e1.shape, e2.shape]
        if shapes == ['circle', 'circle']:
            dist = math.sqrt(((e1.pos[0]-e2.pos[0])*self.width) ** 2
                             + ((e1.pos[1]-e2.pos[1])*self.height)**2)
            return max(0, e1.radius + e2.radius - dist)
        elif shapes == ['circle', 'rect']:
            return sum([abs(d) for d in e1._get_crash_force_dir(e2)])
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

    def _spawnAgent(self):
        self.agent = entities.Agent(self)
        self.agent.draw_path = self.agent_draw_path
        for k, v in self.agent_attrs.items():
            setattr(self.agent, k, v)

    def reset(self, force_reset_path=False):
        pygame.init()
        self._init = True
        self._steps = 0
        self._has_value_map = False
        self._seed(self.env_seed)
        self._rendered = False
        self._disturb_next = False
        self.inp = (0.5, 0.5)
        # will get rescaled acording to fps (=reward per second)
        self.new_reward = 0
        self.new_abs_reward = 0  # will not get rescaled. should be used for one-time rewards
        self.gotHarm = False
        self.score = self.start_score
        self.entities = []
        self.timers = []
        self._spawnAgent()
        self.setup()
        self.entities.append(self.agent)  # add it last, will be drawn on top
        self.observable.reset()
        if self.clear_path_on_reset or force_reset_path:
            self._reset_paths()
        return self.observable.get_observation()

    def _reset_paths(self):
        self.path_overlay = pygame.Surface(
            (self.width, self.height), pygame.SRCALPHA, 32)

    def _draw_entities(self):
        for entity in self.entities:
            entity.draw()

    def _invalidate_value_map(self):
        self._has_value_map = False

    def _draw_values(self, value_func, static=True, resolution=64, color_depth=224, color_mapper=None):
        if (not (static and self._has_value_map)):
            agentpos = self.agent.pos
            agentspeed = self.agent.speed
            self.agent.speed = (0, 0)
            self.value_overlay = pygame.Surface(
                (self.width, self.height), pygame.SRCALPHA, 32)
            obs = []
            for i in range(resolution):
                for j in range(resolution):
                    x, y = (i+0.5)/resolution, (j+0.5)/resolution
                    self.agent.pos = x, y
                    ob = self.observable.get_observation()
                    obs.append(ob)
            self.agent.pos = agentpos
            self.agent.speed = agentspeed

            V = value_func(th.Tensor(np.array(obs)))
            V /= max(V.max(), -1*V.min())*2
            if color_mapper != None:
                V = color_mapper(V)
            V += 0.5

            c = 0
            for i in range(resolution):
                for j in range(resolution):
                    v = V[c].item()
                    c += 1
                    col = [int((1-v)*color_depth),
                           int(v*color_depth), 0, color_depth]
                    x, y = i*(self.width/resolution), j * \
                        (self.height/resolution)
                    rect = pygame.Rect(x, y, int(self.width/resolution)+1,
                                       int(self.height/resolution)+1)
                    pygame.draw.rect(self.value_overlay, col,
                                     rect, width=0)
        self.surface.blit(self.value_overlay, (0, 0))
        self._has_value_map = True

    def _draw_observable(self, forceDraw=False):
        if self.draw_observable and (self.visible or forceDraw):
            self.observable.draw()

    def _draw_joystick(self, forceDraw=False):
        if self.draw_joystick and (self.visible or forceDraw):
            x, y = self.inp
            bigcol = (100, 100, 100)
            smolcol = (100, 100, 100)
            if self._disturb_next:
                smolcol = (255, 255, 255)
            pygame.draw.circle(self.screen, bigcol, (50 +
                                                     self.joystick_offset[0], 50+self.joystick_offset[1]), 50, width=1)
            pygame.draw.circle(self.screen, smolcol, (20+int(60*x) +
                                                      self.joystick_offset[0], 20+int(60*y)+self.joystick_offset[1]), 20, width=0)

    def _draw_confidence_ellipse(self, chol, forceDraw=False, seconds=0.1):
        # The 'seconds'-parameter only really makes sense, when using control_type='SPEED',
        # you can still use it to scale the cov-ellipse when using control_type='ACC',
        # but it's relation to 'seconds' is no longer there...
        if self.draw_confidence_ellipse and (self.visible or forceDraw):
            col = (255, 255, 255)
            f = seconds*self.speed_fac*self.fps*max(self.height, self.width)

            while len(chol.shape) > 2:
                chol = chol[0]
            if chol.shape != (2, 2):
                chol = th.diag_embed(chol)
            if len(chol.shape) != 2:
                chol = chol[0]
            cov = chol.T @ chol

            L, V = th.linalg.eig(cov)
            L, V = L.real, V.real
            l1, l2 = int(abs(math.sqrt(L[0].item())*f)) + \
                1, int(abs(math.sqrt(L[1].item())*f))+1

            if l1 >= l2:
                w, h = l1, l2
                run, rise = V[0][0], V[0][1]
            else:
                w, h = l2, l1
                run, rise = V[1][0], V[1][1]

            ang = (math.atan(rise/run))/(2*math.pi)*360

            # print(w, h, (run, rise, ang))

            x, y = self.agent.pos
            x, y = x*self.width, y*self.height
            rect = pygame.Rect((x-w/2, y-h/2, w, h))
            shape_surface = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.ellipse(shape_surface, col,
                                (0, 0, *rect.size), 1)
            rotated_surf = pygame.transform.rotate(shape_surface, ang)
            self.screen.blit(rotated_surf, rotated_surf.get_rect(
                center=rect.center))

    def _draw_paths(self):
        if self.path_decay != 0.0:
            s = pygame.Surface((self.width, self.height))
            s.set_alpha(soft_int(255*self.path_decay/self.fps))
            s.fill((0, 0, 0))
            self.path_overlay.blit(s, (0, 0))
        self.surface.blit(self.path_overlay, (0, 0))

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
            elif keys[pygame.K_t]:
                self._reset_paths()
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

    def render(self, mode='human', dont_show=False, chol=None, value_func=None, values_static=True):
        if mode == 'human':
            self._handle_user_input()
        self.visible = self.visible or not dont_show
        self._ensure_surface()
        pygame.draw.rect(self.surface, (0, 0, 0),
                         pygame.Rect(0, 0, self.width, self.height))
        if value_func != None:
            self._draw_values(value_func, values_static,
                              color_mapper=self.value_color_mapper)
        self._draw_paths()
        if self.draw_entities:
            self._draw_entities()
        else:
            self.agent.draw()
        self._rendered = True
        if mode == 'human' and dont_show:
            return
        self.screen.blit(self.surface, (0, 0))
        self._draw_observable(forceDraw=mode != 'human')
        self._draw_joystick(forceDraw=mode != 'human')
        if chol != None:
            self._draw_confidence_ellipse(chol, forceDraw=mode != 'human')
        if self.visible and mode == 'human':
            pygame.display.update()
        if mode != 'human':
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.display.quit()
        pygame.quit()


class ColumbusConfigDefined(ColumbusEnv):
    # Allows defining Columbus Environments using dicts.
    # Intended to be used in combination with cw2 configuration.
    # Look into humanPlayer to see how this is supposed to be interfaced with.

    def __init__(self, observable={}, env_seed=None, entities=[], fps=30, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=env_seed, **kw)
        self.entities_definitions = entities
        self.start_pos = self.conv_unit(self.start_pos[0], target='em', axis='x'), self.conv_unit(
            self.start_pos[1], target='em', axis='y')

    def is_unit(self, s):
        if type(s) in [int, float]:
            return True
        if s.replace('.', '', 1).isdigit():
            return True
        num, unit = s[:-2], s[-2:]
        if unit in ['px', 'em', 'rx', 'ry', 'ct', 'au']:
            if num.replace('.', '', 1).isdigit():
                return True
        return False

    def conv_unit(self, s, target='px', axis='x'):
        assert self.is_unit(s)
        if type(s) in [int, float]:
            return s
        if s.replace('.', '', 1).isdigit():
            if target == 'px':
                return int(s)
            return float(s)
        num, unit = s[:-2], s[-2:]
        num = float(num)
        if unit == 'rx':
            unit = 'px'
            axis = 'x'
        elif unit == 'ry':
            unit = 'px'
            axis = 'y'
        if unit == 'em':
            em = num
        elif unit == 'px':
            em = num / ({'x': self.width, 'y': self.height}[axis])
        elif unit == 'au':
            em = num * 36 / ({'x': self.width, 'y': self.height}[axis])
        elif unit == 'ct':
            em = num / 100
        else:
            raise Exception('Conversion not implemented')

        if target == 'em':
            return em
        elif target == 'px':
            return int(em * ({'x': self.width, 'y': self.height}[axis]))

    def setup(self):
        self.agent.pos = self.start_pos
        for i, e in enumerate(self.entities_definitions):
            Entity = getattr(entities, e['type'])
            for i in range(e.get('num', 1) + int(self.random()*(0.99+e.get('num_rand', 0)))):
                entity = Entity(self)
                conf = {k: v for k, v in e.items() if str(
                    k) not in ['num', 'num_rand', 'type']}

                for k, v_raw in conf.items():
                    if k == 'pos':
                        v = self.conv_unit(v_raw[0], target='em', axis='x'), self.conv_unit(
                            v_raw[1], target='em', axis='y')
                    elif k in ['width', 'height', 'radius']:
                        v = self.conv_unit(
                            v_raw, target='px', axis='y' if k == 'height' else 'x')
                    else:
                        v = v_raw
                    if k.endswith('_rand'):
                        n = k.replace('_rand', '')
                        cur = getattr(
                            entity, n)
                        inc = int((v+0.99)*self.random())
                        setattr(entity, n, cur + inc)
                    elif k.endswith('_randf'):
                        n = k.replace('_randf', '')
                        cur = getattr(
                            entity, n)
                        inc = v*self.random()
                        setattr(entity, n, cur + inc)
                    else:
                        setattr(entity, k, v)

                self.entities.append(entity)

###
# Custom Env Definitions


class ColumbusTest3_1(ColumbusEnv):
    def __init__(self, observable=observables.CnnObservable(out_width=48, out_height=48), fps=30, aux_reward_max=1, **kw):
        super(ColumbusTest3_1, self).__init__(
            observable=observable, fps=fps, env_seed=3.1, aux_reward_max=aux_reward_max, **kw)
        self.start_pos = [0.6, 0.3]
        self.score = 0

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


class ColumbusTestRect(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(), fps=30, aux_reward_max=1, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=3.3, aux_reward_max=aux_reward_max, controll_type='ACC', **kw)
        self.start_pos = [0.5, 0.5]
        self.score = 0

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(1):
            enemy = entities.RectBarrier(self)
            enemy.width = self.random()*40+50
            enemy.height = self.random()*40+50
            self.entities.append(enemy)
        for i in range(1):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*40+50
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            self.entities.append(reward)


class ColumbusTestRay(ColumbusTest3_1):
    def __init__(self, observable=observables.RayObservable(), hide_map=False, fps=30, **kw):
        super(ColumbusTestRay, self).__init__(
            observable=observable, fps=fps, **kw)
        self.draw_entities = not hide_map


class ColumbusRayDrone(ColumbusTestRay):
    def __init__(self, observable=observables.RayObservable(), hide_map=False, fps=30, **kw):
        super(ColumbusRayDrone, self).__init__(
            observable=observable, hide_map=hide_map,  fps=fps, **kw)
        self.controll_type = 'ACC'
        self.agent_drag = 0.02


class ColumbusDemoEnv3_1(ColumbusEnv):
    def __init__(self, observable=observables.Observable(), fps=30, aux_reward_max=1, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=3.1, aux_reward_max=aux_reward_max, controll_type='ACC', agent_drag=0.05, **kw)
        self.start_pos = [0.6, 0.3]
        self.score = 0

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(18):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*40+50
            self.entities.append(enemy)
        for i in range(0):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4*0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            self.entities.append(reward)


class ColumbusDemoEnv2_7(ColumbusEnv):
    def __init__(self, observable=observables.Observable(), fps=30, aux_reward_max=1, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=2.7, aux_reward_max=aux_reward_max, controll_type='ACC', agent_drag=0.05, **kw)
        self.start_pos = [0.6, 0.3]
        self.score = 0

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(12):
            enemy = entities.CircleBarrier(self)
            enemy.radius = self.random()*30+40
            self.entities.append(enemy)
        for i in range(3):
            enemy = entities.FlyingChaser(self)
            enemy.chase_acc = self.random()*0.4*0.3  # *0.6+0.5
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            self.entities.append(reward)


class ColumbusDemoEnvFootball(ColumbusEnv):
    def __init__(self, observable=observables.Observable(), fps=30, walkingOpponent=0, flyingOpponent=0, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=1.23, **kw)
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


class ColumbusCandyland(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(chans=[entities.Reward, entities.Void], num_rays=16, include_rand=True), hide_map=False, fps=30, env_seed=None, **kw):
        super(ColumbusCandyland, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed, **kw)
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
    def __init__(self, fps=30, aux_reward_max=10, **kw):
        super(ColumbusCandyland_Aux10, self).__init__(
            fps=fps, aux_reward_max=aux_reward_max, **kw)


class ColumbusEasyObstacles(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16), hide_map=False, fps=30, env_seed=None, aux_reward_max=10, **kw):
        super(ColumbusEasyObstacles, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
        self.draw_entities = not hide_map

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
    def __init__(self, observable=observables.RayObservable(num_rays=16), hide_map=False, fps=30, env_seed=None, aux_reward_max=10, **kw):
        super(ColumbusEasierObstacles, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
        self.draw_entities = not hide_map

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
    def __init__(self, observable=observables.CompositionalObservable([observables.RayObservable(num_rays=6, chans=[entities.Enemy]), observables.StateObservable(coordsAgent=True, speedAgent=False, coordsRelativeToAgent=False, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=False, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True)]), hide_map=False, fps=30, env_seed=None, aux_reward_max=10, **kw):
        super().__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
        self.draw_entities = not hide_map

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


class ColumbusSingle(ColumbusEnv):
    def __init__(self, observable=observables.CompositionalObservable([observables.RayObservable(num_rays=6, chans=[entities.Enemy]), observables.StateObservable(coordsAgent=False, speedAgent=False, coordsRelativeToAgent=True, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=False, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True)]), hide_map=False, fps=30, env_seed=None, aux_reward_max=1, enemy_damage=1, reward_reward=25, void_damage=1, **kw):
        super().__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, void_damage=void_damage, **kw)
        self.draw_entities = not hide_map
        self._enemy_damage = enemy_damage
        self._reward_reward = reward_reward

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(4 + math.floor(self.random()*4)):
            enemy = entities.CircleBarrier(self)
            enemy.radius = 30 + self.random()*70
            enemy.damage = self._enemy_damage
            self.entities.append(enemy)
        for i in range(1):
            reward = entities.TeleportingReward(self)
            reward.radius = 30
            reward.reward = self._reward_reward
            self.entities.append(reward)


class ColumbusJustState(ColumbusEnv):
    def __init__(self, observable=observables.StateObservable(), fps=30, num_enemies=0, num_rewards=1, env_seed=None, aux_reward_max=10, **kw):
        super(ColumbusJustState, self).__init__(
            observable=observable, fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
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
    def __init__(self, observable=observables.StateObservable(coordsAgent=True, speedAgent=False, coordsRelativeToAgent=False, coordsRewards=True, rewardsWhitelist=None, coordsEnemys=True, enemysWhitelist=None, enemysNoBarriers=True, rewardsTimeouts=False, include_rand=True), fps=30, env_seed=3.141, num_enemys=0, num_barriers=3, aux_reward_max=10, **kw):
        super(ColumbusStateWithBarriers, self).__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
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


class ColumbusCompassWithBarriers(ColumbusEnv):
    def __init__(self, observable=observables.CompassObservable(coordsRewards=True), fps=30, env_seed=3.141, num_enemys=0, num_barriers=3, aux_reward_max=10, **kw):
        super().__init__(
            observable=observable,  fps=fps, env_seed=env_seed, aux_reward_max=aux_reward_max, **kw)
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
    def __init__(self, observable=observables.RayObservable(num_rays=8, ray_len=512), hide_map=False, fps=30, **kw):
        super(ColumbusTrivialRay, self).__init__(
            observable=observable, fps=fps, num_chasers=0, **kw)
        self.draw_entities = not hide_map


class ColumbusFootball(ColumbusEnv):
    def __init__(self, observable=observables.RayObservable(num_rays=16, chans=[entities.Goal, entities.Ball, entities.Barrier]), fps=30, walkingOpponent=0, flyingOpponent=0, **kw):
        super(ColumbusFootball, self).__init__(
            observable=observable, fps=fps, env_seed=None, **kw)
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


class ColumbusBlub(ColumbusEnv):
    def __init__(self, observable=observables.CompositionalObservable([observables.StateObservable(), observables.RayObservable(num_rays=6, chans=[entities.Enemy])]), env_seed=None, entities=[], fps=30, **kw):
        super().__init__(
            observable=observable, fps=fps, env_seed=env_seed, default_collision_elasticity=0.8, speed_fac=0.01, acc_fac=0.1, agent_drag=0.06, controll_type='ACC', aux_penalty_max=1)

    def setup(self):
        self.agent.pos = self.start_pos
        for i in range(1):
            enemy = entities.RectBarrier(self)
            enemy.radius = 100
            enemy.width, enemy.height = 200, 75
            self.entities.append(enemy)


###
# Registering Envs fro Gym
register(  # Legacy
    id='ColumbusConfigDefined-v0',
    entry_point=ColumbusConfigDefined,
    max_episode_steps=30*60*2,  # 2 min at default (30) fps
)

register(
    id='Columbus-v1',
    entry_point=ColumbusConfigDefined
)

###

# register(
#    id='ColumbusBlub-v0',
#    entry_point=ColumbusBlub,
#    max_episode_steps=30*60*2,
# )


# register(
#    id='ColumbusTestCnn-v0',
#    entry_point=ColumbusTest3_1,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusTestRay-v0',
#    entry_point=ColumbusTestRay,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusRayDrone-v0',
#    entry_point=ColumbusRayDrone,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusCandyland-v0',
#    entry_point=ColumbusCandyland,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusCandyland_Aux10-v0',
#    entry_point=ColumbusCandyland_Aux10,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusEasyObstacles-v0',
#    entry_point=ColumbusEasyObstacles,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusEasierObstacles-v0',
#    entry_point=ColumbusEasyObstacles,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusJustState-v0',
#    entry_point=ColumbusJustState,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusStateWithBarriers-v0',
#    entry_point=ColumbusStateWithBarriers,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusCompassWithBarriers-v0',
#    entry_point=ColumbusCompassWithBarriers,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusTrivialRay-v0',
#    entry_point=ColumbusTrivialRay,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusFootball-v0',
#    entry_point=ColumbusFootball,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusComb-v0',
#    entry_point=ColumbusComp,
#    max_episode_steps=30*60*2,
# )

# register(
#    id='ColumbusSingle-v0',
#    entry_point=ColumbusSingle,
#    max_episode_steps=30*60*2,
# )

register(
    id='ColumbusDemoEnvFootball-v0',
    entry_point=ColumbusDemoEnvFootball,
    max_episode_steps=30*60*2,
)
register(
    id='ColumbusDemoEnv3_1-v0',
    entry_point=ColumbusDemoEnv3_1,
    max_episode_steps=30*60*2,
)
register(
    id='ColumbusDemoEnv2_7-v0',
    entry_point=ColumbusDemoEnv2_7,
    max_episode_steps=30*60*2,
)
