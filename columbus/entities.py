import pygame
import math


class Entity(object):
    def __call__(cls, *args, **kwargs):
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj

    def __init__(self, env):
        self.shape = None
        self.env = env
        self.pos = (env.random(), env.random())
        self.last_pos = None
        self.speed = (0, 0)
        self.acc = (0, 0)
        self.drag = 0
        self.col = (255, 255, 255)
        self.solid = False
        self.movable = False  # False = Non movable, True = Movable, x>1: lighter movable
        self.elasticity = 1
        self.collision_changes_speed = self.env.controll_type == 'ACC'
        self.collision_elasticity = self.env.default_collision_elasticity
        self._crash_list = []
        self._coll_add_pushback = 0
        self.crash_conservation_of_energy = True
        self.draw_path = False
        self.draw_path_col = (55, 55, 55)
        self.draw_path_width = 2

    def __post_init__(self):
        pass

    def physics_step(self):
        self.last_pos = self.pos[0], self.pos[1]
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
        if self.draw_path and self.last_pos:
            pygame.draw.line(self.env.path_overlay, self.draw_path_col,
                             (self.last_pos[0]*self.env.width, self.last_pos[1]*self.env.height), (self.pos[0]*self.env.width, self.pos[1]*self.env.height), self.draw_path_width)

    def on_collision(self, other, depth):
        if self.solid and other.solid:
            if self.movable:
                self.on_crash(other, depth)

    def on_crash(self, other, depth):
        if other in self._crash_list:
            return
        self._crash_list.append(other)
        force_dir = self._get_crash_force_dir(other)
        #print(force_dir, depth)
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
        if sum([abs(f) for f in force_vec]) > 0.005:
            self.pos = self.pos[0] + force_vec[0], self.pos[1] + force_vec[1]
            if self._coll_add_pushback:
                self.pos = self.pos[0] - self.env.inp[0]*self._coll_add_pushback * \
                    self.env.speed_fac, self.pos[1] - self.env.inp[1] * \
                    self._coll_add_pushback*self.env.speed_fac
        if self.collision_changes_speed:
            oldspeed = math.sqrt(self.speed[0]**2+self.speed[1]**2)
            self.speed = self.speed[0] + \
                force_vec[0]*self.collision_elasticity/self.env.speed_fac, self.speed[1] + \
                force_vec[1]*self.collision_elasticity/self.env.speed_fac
            newspeed = math.sqrt(self.speed[0]**2+self.speed[1]**2)
            if self.crash_conservation_of_energy and newspeed > oldspeed*1.1:
                self.speed = self.speed[0]/newspeed*1.1 * \
                    oldspeed, self.speed[1]/newspeed*oldspeed*1.1

    def _get_crash_force_dir(self, other):
        if 1 == 1:  # linter hack
            raise Exception(
                '[!] No collision-logic implemented for shape"'+str(self.shape)+'"')

    def on_collect(self, other):
        pass

    def on_collected(self):
        pass

    def calc_void_collision(self, dir, x, y, vx, vy):
        if dir < 2:
            x = min(max(x, 0), 1)
            vx = -vx*self.collision_elasticity*0.5*self.collision_changes_speed
        else:
            y = min(max(y, 0), 1)
            vy = -vy*self.collision_elasticity*0.5*self.collision_changes_speed
        return x, y, vx, vy

    def kill(self):
        self.env.kill_entity(self)


class CircularEntity(Entity):
    def __init__(self, env):
        super().__init__(env)
        self.shape = 'circle'
        self.radius = 10

    def draw(self):
        super().draw()
        x, y = self.pos
        pygame.draw.circle(self.env.surface, self.col,
                           (x*self.env.width, y*self.env.height), self.radius, width=0)

    def _get_crash_force_dir(self, other):
        if other.shape == 'circle':
            return self.pos[0] - other.pos[0], self.pos[1] - other.pos[1]
        elif other.shape == 'rect':
            pad = 0
            edge_size = min(self.radius, min(
                other.width/3, other.height/3)) + 1

            x, y = self.pos
            x, y = x*self.env.height, y*self.env.width
            left, top = x - self.radius + pad, y - self.radius + pad
            right, bottom = x + self.radius - pad, y + self.radius - pad
            lrcenter, tbcenter = x, y

            ox, oy = other.pos
            ox, oy = ox*self.env.height, oy*self.env.width
            oleft, otop = ox + pad, oy + pad
            oright, obottom = ox + other.width - pad, oy + other.height - pad
            olrcenter, otbcenter = ox + other.width/2, oy + other.height/2

            lr, tb = 0, 0

            if otop < bottom and obottom > bottom:
                # col from top
                tb = otop - bottom
                #print('t', tb)
            elif top < obottom and top > otop:
                # col from bottom
                tb = - top + obottom
                #print('b', tb)

            if right > oleft and right < oright:
                # col from left
                lr = oleft - right
                #print('l', lr)
            elif left < oright and left > oleft:
                # col from right
                lr = - left + oright
                #print('r', lr)

            if lr != 0 and tb != 0:
                if abs(abs(tb) - abs(lr)) < edge_size:
                    if abs(tb) < abs(lr):
                        return lr/5, tb
                    else:
                        return lr, tb/5
                if abs(tb) < abs(lr):
                    return 0, tb
                else:
                    return lr, 0

            return 0, 0
        else:
            raise Exception(
                '[!] Shape "circle" does not know how to collide with shape "'+str(other.shape)+'"')


class RectangularEntity(Entity):
    def __init__(self, env):
        super().__init__(env)
        self.shape = 'rect'
        self.width = 10
        self.height = 10

    def draw(self):
        super().draw()
        x, y = self.pos
        rect = pygame.Rect(x*self.env.width, y *
                           self.env.width, self.width, self.height)
        pygame.draw.rect(self.env.surface, self.col,
                         rect, width=0)

    def _get_crash_force_dir(self, other):
        raise Exception(
            '[!] Collisions in this direction not implemented for shape "rectangle"')


class Agent(CircularEntity):
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


class CircleBarrier(Barrier, CircularEntity):
    def __init__(self, env):
        super(CircleBarrier, self).__init__(env)


class RectBarrier(Barrier, RectangularEntity):
    def __init__(self, env):
        super().__init__(env)


class Chaser(Enemy, CircularEntity):
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


class Collectable(CircularEntity):
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


class LoopReward(OnceReward):
    def __init__(self, env):
        super().__init__(env)
        self.loop = [[0.25, 0.5], [0.75, 0.5]]
        self.state = 0
        self.jump_to_state()
        self.barrier_physics = False

    def jump_to_state(self):
        pos_vec = [v for v in self.loop[self.state]]
        if len(pos_vec) == 4:
            pos_vec = pos_vec[0] + pos_vec[2] * \
                (self.env.random()-0.5), pos_vec[1] + \
                pos_vec[3]*(self.env.random()-0.5)
        self.pos = pos_vec

    def next_state(self):
        self.state = (self.state + 1) % len(self.loop)

    def jump_next(self):
        self.next_state()
        self.jump_to_state()

    def on_collected(self):
        self.env.new_abs_reward += self.reward
        self.jump_next()

    def physics_step(self):
        if self.barrier_physics:
            self.env.check_collisions_for(self)
        super().physics_step()


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


class Ball(CircularEntity):
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
