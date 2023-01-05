import math
import numpy as np

from typing import List
from matplotlib import pyplot as plt 

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
EPISODES =50

def transform_stat_value(value, A , B ,C , cst):
    standardized_value=(value-A*10)/(B*10)
    cosh=np.cosh(np.exp(-standardized_value))
    transformed_value=1.1-(1/cosh+(value*C/10))
    return cst*transformed_value

def real_value_life(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 15) #varie entre 0 et 15
    return real_stat_value

def real_value_pleasure(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 5) #varie entre 0 et 5
    return real_stat_value

def real_value_disgust(stat_value) :
    real_stat_value = stat_value**2 / 20   #varie entre 0 et 5
    return real_stat_value

def real_value_food(stat_value) :
    real_stat_value = transform_stat_value(stat_value,0.17,0.18,0.1, 18) #varie entre 0 et 18
    return real_stat_value

def reward(stats):
    reward = real_value_life(stats['life']) + real_value_pleasure(stats['pleasure']) + real_value_disgust(stats['disgust']) + real_value_food(stats['food'])
    print('stat life', stats['life'])
    print(real_value_life(stats['life']))
    print('stat pleasure', stats['pleasure'])

    print(real_value_pleasure(stats['pleasure']))
    print('stat disgust', stats['disgust'])

    print(real_value_disgust(stats['disgust']))
    print('stat food', stats['food'])

    print(real_value_food(stats['food']))
    print(reward)
    print('...')
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


stats_list = []
for stat in stats.stats:
        stats_list.append([])

number_of_death = 0
reward_list=[]

while i < EPISODES:
    
    
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
            number_of_death+= 1
    print("stats finale : ", [stat.current_value for stat in stats.stats])
    print("reward finale : ", reward(stats))
    reward_list.append(reward(stats))
    for j in range(len(stats.stats)):
        stats_list[j].append(stats.stats[j].current_value)
    i += 1

knowledge.show()

# stats number of deaths : 
if number_of_death != 0 :
    print(f"Number of deaths : {number_of_death}")
    print(f"Mean survival time : {EPISODES/number_of_death} episodes")
else : 
    print('No deaths during the training')


stats_names = ['Food', 'Life', 'Pleasure','Disgust']
x = np.linspace(start=0, stop= EPISODES, num=EPISODES)
fig = plt.figure()
for i in range(len(stats.stats)):
    plt.subplot(2, 2, i+1)
    plt.ylim(0,10)
    plt.plot(x, stats_list[i])
    plt.title(stats_names[i])  
#plt.show()
'''
plt.figure()
for i in range(len(stats.stats)):
    plt.plot(x, stats_list[i], label= stats_names[i])
plt.legend()
plt.title('Evolution of stats during the time')
'''


plt.figure()
plt.plot(x,reward_list)
plt.title("Reward (we try to minimize it)")
#plt.show()

# Evolution of reward for each stat
possible_stats=list(range(0,10))
ylife=[real_value_life(stat) for stat in possible_stats]
yfood=[real_value_food(stat) for stat in possible_stats]
ydisgust=[real_value_disgust(stat) for stat in possible_stats]
ypleasure=[real_value_pleasure(stat) for stat in possible_stats]
plt.figure()
plt.plot(possible_stats,ylife, c = 'red', label = 'life', alpha = 0.8)
plt.plot(possible_stats,yfood, c = 'blue', label = 'food', alpha = 0.8)
plt.plot(possible_stats,ydisgust, c = 'green', label = 'disgust', alpha = 0.8)
plt.plot(possible_stats,ypleasure, c = 'yellow', label = 'pleasure', alpha = 0.8)
plt.legend()
plt.xlabel('Stat value')
plt.ylabel('Associated reward')
plt.title('Reward for each stat according to its value')
plt.show()
