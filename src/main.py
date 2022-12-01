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
EPISODES =1000

def transform_stat_value(value, A , B ,C , cst):
    standardized_value=(value-A*10)/(B*10)
    cosh=np.cosh(np.exp(-standardized_value))
    transformed_value=1.1-(1/cosh+(value*C/10))
    return cst*transformed_value

def real_value_life(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 10) #varie entre 0 et 10
    return real_stat_value

def real_value_pleasure(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 5) #varie entre 0 et 5
    return real_stat_value

def real_value_disgust(stat_value) :
    real_stat_value = stat_value**2 / 20   #varie entre 0 et 5
    return real_stat_value

def real_value_food(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 10) #varie entre 0 et 10
    return real_stat_value

def reward(stats):
    reward = real_value_life(stats['life']) + real_value_pleasure(stats['pleasure']) + real_value_disgust(stats['disgust']) + real_value_food(stats['food'])
    return reward

def forecast_reward(event,action:Action): 
    stats_copy = deepcopy(stats)
    knowledge_copy = deepcopy(knowledge)
    knowledge_copy.change_stats(stats_copy.stats)
    for stat in stats_copy.stats:
        stat.label = stat.label + '_copy'
    event.run(knowledge_copy, lambda _: action)
    #print("for action : ", action.action_id)
    for stat in stats_copy.stats:
        stat.label = stat.label[:-len('_copy')]
    r = reward(stats_copy)
    #print("reward : ", r)
    #print("copy ",[stat.current_value for stat in stats_copy.stats])
    return r

def epsilon(time):
    standardized_time=(time-A*EPISODES)/(B*EPISODES)
    cosh=np.cosh(math.exp(-standardized_time))
    epsilon=1.1-(1/cosh+(time*C/EPISODES))
    return epsilon    

def action_choice(event,time):
    def ff(actions: List[Action]) -> Action:
        print("---")
        print("Encounter with : ", event.id)
        eps = epsilon(time)
        
        p = random()
        if p < eps : 
            print("random : ", eps)
            a = random_policy(actions)
        else : 
            print("best : ", eps)
            a = best_policy(event, actions)
        print("Chosen action : ", a.action_id)
        return a
    return ff

def best_policy(event, actions: List[Action]) -> Action:
    return min(actions,key=lambda action: np.mean([forecast_reward(event,action) for _ in range(100)]))

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
    print("stats finale : ", [stat.current_value for stat in stats.stats])
    print("reward finale : ", reward(stats))
    i += 1

knowledge.show()
