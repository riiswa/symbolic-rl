# To use this code, make sure you
#
#     import json
#
# and then, to convert JSON from a string, do
#
#     result = welcome5_from_dict(json.loads(json_string))

from typing import List, Optional, Any, TypeVar, Callable, Type, cast

from utils import merge_yaml_data

T = TypeVar("T")


def from_str(x: Any) -> str:
    assert isinstance(x, str)
    return x


def from_int(x: Any) -> int:
    assert isinstance(x, int) and not isinstance(x, bool)
    return x


def from_list(f: Callable[[Any], T], x: Any) -> List[T]:
    assert isinstance(x, list)
    return [f(y) for y in x]


def from_none(x: Any) -> Any:
    assert x is None
    return x


def from_union(fs, x):
    for f in fs:
        try:
            return f(x)
        except:
            pass
    assert False


def from_bool(x: Any) -> bool:
    assert isinstance(x, bool)
    return x


def to_class(c: Type[T], x: Any) -> dict:
    assert isinstance(x, c)
    return cast(Any, x).to_dict()


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

    def to_dict(self) -> dict:
        result: dict = {"events": from_list(lambda x: to_class(Event, x), self.events)}
        return result


if __name__ == "__main__":
    import yaml
    from sys import argv

    if len(argv) > 1:
        print(Events.from_paths(argv[1:]))
    else:
        print("Please provide at least one file")