import math
import numpy as np

from typing import List

from knowledge import Knowledge
from events import Events, Action
from stats import Stats

from random import choice, random
from copy import deepcopy


events = Events.from_paths(['data/events/events_example.yaml'])

stats = Stats.from_paths(['data/stats/stats.yaml'])

knowledge = Knowledge(stats.stats)
knowledge.from_paths(['data/knowledge/world1.yaml'])

# Paramètres pour la fonction epsilon 
A = 0.5
B = 0.1 
C =  0.1
EPISODES =10

def reward(stats):
    #use function to manage reward
    reward = stats['life'] + stats['pleasure'] - stats['disgust'] + stats['food']
    return reward

def forecast_reward(event,action:Action): 
    print("stats ", [stat.current_value for stat in stats.stats])
    stats_copy = deepcopy(stats)
    knowledge_copy = deepcopy(knowledge)
    knowledge_copy.change_stats(stats_copy)
    for stat in stats_copy.stats:
        stat.label = stat.label + '_copy'
    event.run(knowledge_copy, lambda _: action)
    
    print("for ", action.action_id)
    for stat in stats_copy.stats:
        stat.label = stat.label[:-len('_copy')]
    r = reward(stats_copy)
    print("reward : ", r)
    print("copy ",[stat.current_value for stat in stats_copy.stats])
    return r

def epsilon(time):
    standardized_time=(time-A*EPISODES)/(B*EPISODES)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=1.1-(1/cosh+(time*C/EPISODES))
    return epsilon    

def action_choice(event,time):
    def ff(actions: List[Action]) -> Action:
        print("Rencontre : ", event.id)
        eps = epsilon(time)
        print("epsilon :", eps)
        p = random()
        if p < eps : 
            a = random_policy(actions)
        else : 
            a = best_policy(event, actions)
        print("Chosen action : ", a.action_id)
        return a
    return ff

def best_policy(event, actions: List[Action]) -> Action:
    return max(actions,key=lambda action: forecast_reward(event,action))

def random_policy(actions: List[Action]) -> Action:
    return choice(actions)

knowledge.show()

next_event = None
i = 0
end = False

while i <= EPISODES:
    event = \
        events.get_random_event() if not next_event else list(filter(lambda e: e.id == next_event, events.events))[0]
    next_event = event.run(knowledge, action_choice(event, i))
    for stat in stats.stats:
        if stat.decrease_period != 0 and i % stat.decrease_period == 0:
            stat.update(-1)
        if stat.increase_period != 0 and i % stat.increase_period == 0:
            stat.update(1)

        if stat.deadly and stat.current_value == stat.min:
            stats.reload()
    print([stat.current_value for stat in stats.stats])
    i += 1

knowledge.show()
