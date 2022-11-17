import math
import numpy as np

from typing import List

from knowledge import Knowledge
from events import Events, Action
from stats import Stats

from random import choice


events = Events.from_paths(['data/events/events_example.yaml'])

stats = Stats.from_paths(['data/stats/stats.yaml'])

knowledge = Knowledge(stats.stats)
knowledge.from_paths(['data/knowledge/world1.yaml'])

# Paramètres pour la fonction epsilon 
A = 0.5
B = 0.1 
C =  0.1
EPISODES =10000

def reward(s,a): 
    life = stats['life']
    pleasure = stats['pleasure']
    disgust = stats['disgust']
    food = stats['food']
    #Call event to retrieve all stats changes after interaction between event and actions
    #change stats with the change retrieved
    #Update reward with real function
    reward = life + pleasure - disgust + food
    return reward


def epsilon(time):
    standardized_time=(time-A*EPISODES)/(B*EPISODES)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=1.1-(1/cosh+(time*C/EPISODES))
    return epsilon    

def action_choice(s, time):
    eps = eps(time)
    p = np.random()
    if p < eps : 
        a = random_policy()
    else : 
        a = best_policy()

def random_policy(actions: List[Action]) -> Action:
    return choice(actions)

### pas encore complet
#Call actual event
def best_policy(actions: List[Action]) -> Action:
    best_reward =  -math.inf
    for action in actions:
        reward = reward(s,action)
        if reward > best_reward :
            best_reward = reward
    return choice(actions)


def random_policy(actions: List[Action]) -> Action:
    return choice(actions)

knowledge.show()

next_event = None
max_iteration = 1000
i = 0
end = False

while not end and i <= max_iteration:
    event = \
        events.get_random_event() if not next_event else list(filter(lambda e: e.id == next_event, events.events))[0]
    next_event = event.run(knowledge, random_policy)
    print(event.id)
    for stat in stats.stats:
        if stat.decrease_period != 0 and i % stat.decrease_period == 0:
            stat.update(-1)
        if stat.increase_period != 0 and i % stat.increase_period == 0:
            stat.update(1)

        if stat.deadly and stat.current_value == stat.min:
            end = True
    print([stat.current_value for stat in stats.stats])
    i += 1

knowledge.show()
