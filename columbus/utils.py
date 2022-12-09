from columbus import entities, observables

import random as random_dont_use


def parseObs(obsConf):
    # Parsing Observable Definitions
    if type(obsConf) == list:
        obs = []
        for i, c in enumerate(obsConf):
            obs.append(parseObs(c))
        if len(obs) == 1:
            return obs[0]
        else:
            return observables.CompositionalObservable(obs)

    if obsConf['type'] == 'State':
        conf = {k: v for k, v in obsConf.items() if k not in ['type']}
        return observables.StateObservable(**conf)
    elif obsConf['type'] == 'Compass':
        conf = {k: v for k, v in obsConf.items() if k not in ['type']}
        return observables.CompassObservable(**conf)
    elif obsConf['type'] == 'RayCast':
        chans = []
        for chan in obsConf.get('chans', []):
            chans.append(getattr(entities, chan))
        conf = {k: v for k, v in obsConf.items() if k not in ['type', 'chans']}
        return observables.RayObservable(chans=chans, **conf)
    elif obsConf['type'] == 'CNN':
        conf = {k: v for k, v in obsConf.items() if k not in ['type']}
        return observables.CnnObservable(**conf)
    elif obsConf['type'] == 'Dummy':
        conf = {k: v for k, v in obsConf.items() if k not in ['type']}
        return observables.Observable(**conf)
    else:
        raise Exception('Unknown Observable selected')


def soft_int(num):
    i = int(num)
    r = num - i
    return i + int(random_dont_use.random() < r)
