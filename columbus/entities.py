import pygame
import math


class Entity(object):
    def __init__(self, env):
        self.env = env
        self.pos = (env.random(), env.random())
        self.speed = (0, 0)
        self.acc = (0, 0)
        self.drag = 0
        self.radius = 10
        self.col = (255, 255, 255)
        self.shape = 'circle'
        self.solid = False
        self.movable = False  # False = Non movable, True = Movable, x>1: lighter movable
        self.elasticity = 1
        self.collision_changes_speed = True
        self._crash_list = []
        self._coll_add_pushback = 0

    def physics_step(self):
        x, y = self.pos
        vx, vy = self.speed
        ax, ay = self.acc
        vx, vy = vx+ax*self.env.acc_fac,  vy+ay*self.env.acc_fac
        x, y = x+vx*self.env.speed_fac, y+vy*self.env.speed_fac
        if not self.env.torus_topology:
            if x > 1 or x < 0:
                x, y, vx, vy = self.calc_void_collision(x < 0, x, y, vx, vy)
            if y > 1 or y < 0:
                x, y, vx, vy = self.calc_void_collision(
                    2 + (x < 0), x, y, vx, vy)
        else:
            x = x % 1
            y = y % 1
        self.speed = vx/(1+self.drag), vy/(1+self.drag)
        self.pos = x, y

    def controll_step(self):
        pass

    def step(self):
        self.controll_step()
        self.physics_step()
        self._crash_list = []

    def draw(self):
        x, y = self.pos
        pygame.draw.circle(self.env.surface, self.col,
                           (x*self.env.width, y*self.env.height), self.radius, width=0)

    def on_collision(self, other, depth):
        if self.solid and other.solid:
            if self.movable:
                self.on_crash(other, depth)

    def on_crash(self, other, depth):
        if other in self._crash_list:
            return
        self._crash_list.append(other)
        force_dir = self.pos[0] - other.pos[0], self.pos[1] - other.pos[1]
        force_dir_len = math.sqrt(force_dir[0]**2+force_dir[1]**2)
        if force_dir_len == 0:
            return
        force_dir = force_dir[0]/force_dir_len, force_dir[1]/force_dir_len
        if not self.env.torus_topology:
            if self.env.agent.pos[0] > 0.99 or self.env.agent.pos[0] < 0.01:
                force_dir = force_dir[0], force_dir[1] * 2
            if self.env.agent.pos[1] > 0.99 or self.env.agent.pos[1] < 0.01:
                force_dir = force_dir[0] * 2, force_dir[1]
        depth *= 1.0*self.movable/(self.movable + other.movable)/2
        depth /= other.elasticity
        force_vec = force_dir[0]*depth/self.env.width, \
            force_dir[1]*depth/self.env.height
        self.pos = self.pos[0] + force_vec[0], self.pos[1] + force_vec[1]
        if self._coll_add_pushback:
            self.pos = self.pos[0] - self.env.inp[0]*self._coll_add_pushback * \
                self.env.speed_fac, self.pos[1] - self.env.inp[1] * \
                self._coll_add_pushback*self.env.speed_fac
        if self.collision_changes_speed:
            self.speed = self.speed[0] + \
                force_vec[0]/self.env.speed_fac, self.speed[1] + \
                force_vec[1]/self.env.speed_fac

    def on_collect(self, other):
        pass

    def on_collected(self):
        pass

    def calc_void_collision(self, dir, x, y, vx, vy):
        if dir < 2:
            x = min(max(x, 0), 1)
            vx = 0
        else:
            y = min(max(y, 0), 1)
            vy = 0
        return x, y, vx, vy

    def kill(self):
        self.env.kill_entity(self)


class Agent(Entity):
    def __init__(self, env):
        super(Agent, self).__init__(env)
        self.pos = (0.5, 0.5)
        self.col = (0, 0, 255)
        self.drag = self.env.agent_drag
        self.controll_type = self.env.controll_type
        self.solid = True
        self.movable = True

    def controll_step(self):
        self._read_input()
        self.env.check_collisions_for(self)

    def _read_input(self):
        if self.controll_type == 'SPEED':
            self.speed = self.env.inp[0] - 0.5, self.env.inp[1] - 0.5
        elif self.controll_type == 'ACC':
            self.acc = self.env.inp[0] - 0.5, self.env.inp[1] - 0.5
        else:
            raise Exception('Unsupported controll_type')


class Enemy(Entity):
    def __init__(self, env):
        super(Enemy, self).__init__(env)
        self.col = (255, 0, 0)
        self.damage = 100
        self.radiateDamage = True

    def on_collision(self, other, depth):
        super().on_collision(other, depth)
        if isinstance(other, Agent):
            self.env.new_reward -= self.damage


class Barrier(Enemy):
    def __init__(self, env):
        super(Barrier, self).__init__(env)
        self.solid = True
        self.movable = False


class CircleBarrier(Barrier):
    def __init__(self, env):
        super(CircleBarrier, self).__init__(env)


class Chaser(Enemy):
    def __init__(self, env):
        super(Chaser, self).__init__(env)
        self.target = self.env.agent
        self.arrow_fak = 100
        self.lookahead = 0

    def _get_arrow(self):
        tx, ty = self.target.pos
        x, y = self.pos
        fx, fy = x + self.speed[0]*self.lookahead*self.env.speed_fac, y + \
            self.speed[1]*self.lookahead*self.env.speed_fac
        dx, dy = (tx-fx)*self.arrow_fak, (ty-fy)*self.arrow_fak
        return self.env._limit_to_unit_circle((dx, dy))


class WalkingChaser(Chaser):
    def __init__(self, env):
        super(WalkingChaser, self).__init__(env)
        self.col = (255, 0, 0)
        self.chase_speed = 0.45

    def controll_step(self):
        arrow = self._get_arrow()
        self.speed = arrow[0] * self.chase_speed, arrow[1] * self.chase_speed


class FlyingChaser(Chaser):
    def __init__(self, env):
        super(FlyingChaser, self).__init__(env)
        self.col = (255, 0, 0)
        self.chase_acc = 0.5
        self.arrow_fak = 5
        self.lookahead = 8 + env.random()*2

    def controll_step(self):
        arrow = self._get_arrow()
        self.acc = arrow[0] * self.chase_acc, arrow[1] * self.chase_acc


class Collectable(Entity):
    def __init__(self, env):
        super(Collectable, self).__init__(env)
        self.avaible = True
        self.enforce_not_on_barrier = False
        self.reward = 10
        self.collectors = []

    def on_collision(self, other, depth):
        super().on_collision(other, depth)
        if isinstance(other, Barrier):
            self.on_barrier_collision()
        else:
            for Col in self.collectors:
                if isinstance(other, Col):
                    other.on_collect(self)
                    self.on_collected()

    def on_collected(self):
        self.env.new_reward += self.reward

    def on_barrier_collision(self):
        if self.enforce_not_on_barrier:
            self.pos = (self.env.random(), self.env.random())
            self.env.check_collisions_for(self)


class Reward(Collectable):
    def __init__(self, env):
        super(Reward, self).__init__(env)
        self.col = (0, 255, 0)
        self.reward = 10
        self.collectors = [Agent]


class OnceReward(Reward):
    def __init__(self, env):
        super(OnceReward, self).__init__(env)
        self.reward = 500

    def on_collected(self):
        self.env.new_abs_reward += self.reward
        self.kill()


class TeleportingReward(OnceReward):
    def __init__(self, env):
        super(TeleportingReward, self).__init__(env)
        self.enforce_not_on_barrier = True
        self.env.check_collisions_for(self)

    def on_collected(self):
        self.env.new_abs_reward += self.reward
        self.pos = (self.env.random(), self.env.random())
        self.env.check_collisions_for(self)


class TimeoutReward(OnceReward):
    def __init__(self, env):
        super(TimeoutReward, self).__init__(env)
        self.enforce_not_on_barrier = True
        self.env.check_collisions_for(self)
        self.timeout = 10

    def set_avaible(self, value):
        self.avaible = value
        if self.avaible:
            self.col = (0, 255, 0)
        else:
            self.col = (50, 100, 50)

    def on_collected(self):
        if self.avaible:
            self.env.new_abs_reward += self.reward
            self.set_avaible(False)
            self.env.timers.append((self.timeout, self.set_avaible, True))


class Ball(Entity):
    def __init__(self, env):
        super(Ball, self).__init__(env)
        self.col = (255, 128, 0)
        self.drag = 0.0025
        self.solid = True
        self.movable = 10
        self.elasticity = 1
        self.collision_changes_speed = True
        self.wall_reflect_damping = 0.1

    def calc_void_collision(self, dir, x, y, vx, vy):
        if dir < 2:
            x = min(max(x, 0), 1)
            vx = -(vx/(1+self.wall_reflect_damping))
        else:
            y = min(max(y, 0), 1)
            vy = -(vy/(1+self.wall_reflect_damping))
        return x, y, vx, vy

    def physics_step(self):
        self.env.check_collisions_for(self)
        super().physics_step()


class Goal(Collectable):
    def __init__(self, env):
        super(Goal, self).__init__(env)
        self.col = (0, 200, 0)
        self.reward = 500
        self.radius = 20
        self.collectors = [Ball]


class TeleportingGoal(Goal):
    def __init__(self, env):
        super(TeleportingGoal, self).__init__(env)
        self.enforce_not_on_barrier = True
        self.env.check_collisions_for(self)

    def on_collected(self):
        self.env.new_abs_reward += self.reward
        self.pos = (self.env.random(), self.env.random())
        self.env.check_collisions_for(self)


class FootballPlayer():
    def __init__(self, env, target):
        super(FootballPlayer, self).__init__(env)
        self.col = (200, 0, 100)
        self.target = target
        self.solid = True
        self.movable = 1
        self.elasticity = 1


class WalkingFootballPlayer(FootballPlayer, WalkingChaser):
    def __init__(self, env, target):
        super(WalkingFootballPlayer, self).__init__(env, target)
        self.target = target


class FlyingFootballPlayer(FootballPlayer, FlyingChaser):
    def __init__(self, env, target):
        super(FlyingFootballPlayer, self).__init__(env, target)

# Not a real entity. Is used in the config of RayObserver to reference the outer boundary of the environment.


class Void():
    def __init__(self, env):
        self.col = (50, 50, 50)
    pass
