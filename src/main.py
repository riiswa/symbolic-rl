from typing import List

from knowledge import Knowledge
from events import Events, Action
from stats import Stats

from random import choice


events = Events.from_paths(['data/events/events_example.yaml'])

stats = Stats.from_paths(['data/stats/stats.yaml'])

knowledge = Knowledge(stats.stats)
knowledge.from_paths(['data/knowledge/world1.yaml'])

def reward():
    return

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
