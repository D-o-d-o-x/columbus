from time import sleep, time
import numpy as np
import pygame
import yaml

from columbus import env
from columbus.observables import Observable, CnnObservable


def main():
    env = chooseEnv()
    while True:
        playEnv(env)
        input('<again?>')
    env.close()


def getAvaibleEnvs():
    # kinda hacky... idk
    strs = dir(env)
    for s in strs:
        if s.startswith('Columbus') and s != 'ColumbusEnv':
            yield getattr(env, s)


def loadConfigDefinedEnv(EnvClass):
    p = input('[Path to config> ')
    with open(p, 'r') as f:
        docs = list([d for d in yaml.safe_load_all(
            f) if 'name' in d and d['name'] not in ['SLURM']])
    for i, doc in enumerate(docs):
        name = doc['name']
        print('['+str(i)+'] '+name)
    ds = int(input('[0]> ') or '0')
    doc = docs[ds]
    cur = doc
    path = 'params.task.env_args'
    p = path.split('.')
    while True:
        try:
            if len(p) == 0:
                break
            key = p.pop(0)
            print(key)
            cur = cur[key]
        except Exception as e:
            print('Unable to find key "'+key+'"')
            path = input('[Path> ')
    print(cur)
    return EnvClass(fps=30, **cur)


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
        if envs[i] in [env.ColumbusConfigDefined]:
            return loadConfigDefinedEnv(envs[i])
        Env = envs[i]
        return Env(fps=30)


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
