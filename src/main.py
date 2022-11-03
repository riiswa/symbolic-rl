from typing import List

from knowledge import Knowledge
from events import Events, Action

from random import choice

knowledge = Knowledge()
knowledge.from_paths(['data/knowledge/world1.yaml'])

events = Events.from_paths(['data/events/events_example.yaml'])


def random_policy(actions: List[Action]) -> Action:
    return choice(actions)


knowledge.show()

next_event = None
for i in range(1000):
    event = events.get_random_event() if not next_event else list(filter(lambda e: e.id == next_event, events.events))[
        0]
    next_event = event.run(knowledge, random_policy)

knowledge.show()
