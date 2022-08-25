from time import sleep, time
import numpy as np
import pygame

from columbus import env
from columbus.observables import Observable, CnnObservable


def main():
    Env = chooseEnv()
    env = Env(fps=30)
    env.start_pos = [0.6, 0.3]
    playEnv(env)
    env.close()


def getAvaibleEnvs():
    # kinda hacky... idk
    strs = dir(env)
    for s in strs:
        if s.startswith('Columbus') and s != 'ColumbusEnv':
            yield getattr(env, s)


def chooseEnv():
    envs = list(getAvaibleEnvs())
    for i, Env in enumerate(envs):
        print('['+str(i)+'] '+Env.__name__)
    while True:
        inp = input('[#> ')
        try:
            i = int(inp)
        except:
            print('[!] You have to enter the number...')
        if i < 0 or i >= len(envs):
            print(
                '[!] That is a number, but not one that makes sense in this context...')
        return envs[i]


def playEnv(env):
    done = False
    env.reset()
    while not done:
        t1 = time()
        env.render()
        pos = (0.5, 0.5)
        pos = pygame.mouse.get_pos()
        pos = (min(max((pos[0]-env.joystick_offset[0]-20)/60, 0), 1),
               min(max((pos[1]-env.joystick_offset[1]-20)/60, 0), 1))
        pos = pos[0]*2-1, pos[1]*2-1
        obs, rew, done, info = env.step(np.array(pos, dtype=np.float32))
        print('Reward: '+str(rew))
        print('Score: '+str(info))
        t2 = time()
        dt = t2 - t1
        delay = (1/env.fps - dt)
        if delay < 0:
            print("[!] Can't keep framerate!")
        else:
            sleep(delay)


if __name__ == '__main__':
    main()
