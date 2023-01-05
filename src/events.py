from typing import List, Optional, Any, TypeVar, Callable, Type, cast

from knowledge import Knowledge, KnowledgeNode
from utils import *
import itertools
from random import choice
import re


def extract_weight(element: str):
    return re.search(r"([a-zA-Z]*)(\(\d*\))?", element).groups()


class Consequence:
    description: str
    probability: int
    action_links: List[str]
    event_links: List[str]
    stats_changes: List[str]
    next: Optional[str]
    next_period: bool

    def __init__(self, description: str, probability: int, action_links: List[str], event_links: List[str],
                 stats_changes: List[str], next: Optional[str], next_period: bool) -> None:
        self.description = description
        self.probability = probability
        self.action_links = action_links
        self.event_links = event_links
        self.stats_changes = stats_changes
        self.next = next
        self.next_period = next_period

    @staticmethod
    def from_dict(obj: Any) -> 'Consequence':
        assert isinstance(obj, dict)
        description = from_str(obj.get("description"))
        probability = from_int(obj.get("probability"))
        action_links = from_list(from_str, obj.get("action_links"))
        event_links = from_list(from_str, obj.get("event_links"))
        stats_changes = from_list(from_str, obj.get("stats_changes"))
        next = from_union([from_str, from_none], obj.get("next"))
        next_period = from_bool(obj.get("next_period"))
        return Consequence(description, probability, action_links, event_links, stats_changes, next, next_period)

    def to_dict(self) -> dict:
        result: dict = {"description": from_str(self.description), "probability": from_int(self.probability),
                        "action_links": from_list(from_str, self.action_links),
                        "event_links": from_list(from_str, self.event_links),
                        "stats_changes": from_list(from_str, self.stats_changes),
                        "next": from_union([from_str, from_none], self.next),
                        "next_period": from_bool(self.next_period)}
        return result

    def run(self, action: KnowledgeNode, events: List[KnowledgeNode], knowledge: Knowledge) -> Optional[str]:
        for link in self.action_links:
            l, w = extract_weight(link)

            if w is None:
                w = 1
            else:
                w = int(w[1:-1])
            for _ in range(w):
                action.link(knowledge[l], "gives")

        for link in self.event_links:
            for event in events:
                l, w = extract_weight(link)
                if w is None:
                    w = 1
                else:
                    w = int(w[1:-1])
                for _ in range(int(w)):
                    event.link(knowledge[l], "gives")
        return self.next


class Action:
    action_id: str
    consequences: List[Consequence]

    def __init__(self, action_id: str, consequences: List[Consequence]) -> None:
        self.action_id = action_id
        self.consequences = consequences

    @staticmethod
    def from_dict(obj: Any) -> 'Action':
        assert isinstance(obj, dict)
        action_id = from_str(obj.get("action_id"))
        consequences = from_list(Consequence.from_dict, obj.get("consequences"))
        return Action(action_id, consequences)

    def to_dict(self) -> dict:
        result: dict = {"action_id": from_str(self.action_id),
                        "consequences": from_list(lambda x: to_class(Consequence, x), self.consequences)}
        return result


class Event:
    id: str
    is_a: List[str]
    description: str
    probability: int
    actions: List[Action]

    def __init__(self, id: str, is_a: List[str], description: str, probability: int, actions: List[Action]) -> None:
        self.id = id
        self.is_a = is_a
        self.description = description
        self.probability = probability
        self.actions = actions

    @staticmethod
    def from_dict(obj: Any) -> 'Event':
        assert isinstance(obj, dict)
        id = from_str(obj.get("id"))
        is_a = from_list(from_str, obj.get("is_a"))
        description = from_str(obj.get("description"))
        probability = from_int(obj.get("probability"))
        actions = from_list(Action.from_dict, obj.get("actions"))
        return Event(id, is_a, description, probability, actions)

    def to_dict(self) -> dict:
        result: dict = {"id": from_str(self.id), "is_a": from_list(from_str, self.is_a),
                        "description": from_str(self.description), "probability": from_int(self.probability),
                        "actions": from_list(lambda x: to_class(Action, x), self.actions)}
        return result

    def node_is_concerned(self, node: KnowledgeNode) -> bool:
        link_to_check = [('is_a', is_a) for is_a in self.is_a]
        node_links = [(link.link_name, link.to.id) for link in node.links]
        return all(link in node_links for link in link_to_check)
    
    def random_concerned_node(self, knowledge: Knowledge):
        concerned_nodes = [node for node in knowledge.nodes.values() if self.node_is_concerned(node)]
        if not concerned_nodes and len(self.is_a) == 1:
            concerned_nodes.append(knowledge[self.is_a[0]])
        node = choice(concerned_nodes)
        return node

    def run(self, knowledge: Knowledge, policy: Callable[[List[Action]], Action], concerned_node: Optional[KnowledgeNode] = None) -> Optional[str]:
        action = policy(self.actions)
        action_knowledge = knowledge[action.action_id]
        consequence = choice(flatten([[consequence] * consequence.probability for consequence in action.consequences]))
        if concerned_node is None:
            concerned_nodes = [node for node in knowledge.nodes.values() if self.node_is_concerned(node)]
            if not concerned_nodes and len(self.is_a) == 1:
                concerned_nodes.append(knowledge[self.is_a[0]])
            concerned_node = choice(concerned_nodes)
        return consequence.run(action_knowledge, [concerned_node], knowledge), action

class Events:
    events: List[Event]

    def __init__(self, events: List[Event]) -> None:
        self.events = events

    @staticmethod
    def from_dict(obj: Any) -> 'Events':
        assert isinstance(obj, dict)
        events = from_list(Event.from_dict, obj.get("events"))
        return Events(events)

    @staticmethod
    def from_paths(filepaths: List[str]) -> 'Events':
        return Events.from_dict(merge_yaml_data(filepaths))

    def get_random_event(self, knowledge: Knowledge):
        event = choice(flatten([[event] * event.probability for event in self.events]))
        node = event.random_concerned_node(knowledge)
        return event, node

    def to_dict(self) -> dict:
        result: dict = {"events": from_list(lambda x: to_class(Event, x), self.events)}
        return result


if __name__ == "__main__":
    from sys import argv

    if len(argv) > 1:
        print(Events.from_paths(argv[1:]))
    else:
        print("Please provide at least one file")
