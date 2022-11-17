from utils import *


class Effect:
    node: str
    value: int

    def __init__(self, node: str, value: int) -> None:
        self.node = node
        self.value = value

    @staticmethod
    def from_dict(obj: Any) -> 'Effect':
        assert isinstance(obj, dict)
        node = from_str(obj.get("node"))
        value = from_int(obj.get("value"))
        return Effect(node, value)

    def to_dict(self) -> dict:
        result: dict = {}
        result["node"] = from_str(self.node)
        result["value"] = from_int(self.value)
        return result


class Stat:
    label: str
    base: int
    min: int
    max: int
    deadly: bool
    increase_period: int
    decrease_period: int
    effects: List[Effect]

    def __init__(self, label: str, base: int, min: int, max: int, deadly: bool, increase_period: int, decrease_period: int, effects: List[Effect]) -> None:
        self.label = label
        self.base = base
        self.min = min
        self.max = max
        self.deadly = deadly
        self.increase_period = increase_period
        self.decrease_period = decrease_period
        self.effects = effects
        self.current_value = base

    def update(self, value):
        v = self.current_value + value
        if v < self.min:
            self.current_value = self.min
        elif v > self.max:
            self.current_value = self.max
        else:
            self.current_value = v



    @staticmethod
    def from_dict(obj: Any) -> 'Stat':
        assert isinstance(obj, dict)
        label = from_str(obj.get("label"))
        base = from_int(obj.get("base"))
        min = from_int(obj.get("min"))
        max = from_int(obj.get("max"))
        deadly = from_bool(obj.get("deadly"))
        increase_period = from_int(obj.get("increase_period"))
        decrease_period = from_int(obj.get("decrease_period"))
        effects = from_list(Effect.from_dict, obj.get("effects"))
        return Stat(label, base, min, max, deadly, increase_period, decrease_period, effects)

    def to_dict(self) -> dict:
        result: dict = {}
        result["label"] = from_str(self.label)
        result["base"] = from_int(self.base)
        result["min"] = from_int(self.min)
        result["max"] = from_int(self.max)
        result["deadly"] = from_bool(self.deadly)
        result["increase_period"] = from_int(self.increase_period)
        result["decrease_period"] = from_int(self.decrease_period)
        result["effects"] = from_list(lambda x: to_class(Effect, x), self.effects)
        return result


class Stats:
    stats: List[Stat]

    def __init__(self, stats: List[Stat]) -> None:
        self.stats = stats

    @staticmethod
    def from_dict(obj: Any) -> 'Stats':
        assert isinstance(obj, dict)
        stats = from_list(Stat.from_dict, obj.get("stats"))
        return Stats(stats)

    @staticmethod
    def from_paths(filepaths: List[str]) -> 'Stats':
        return Stats.from_dict(merge_yaml_data(filepaths))

    def to_dict(self) -> dict:
        result: dict = {}
        result["stats"] = from_list(lambda x: to_class(Stat, x), self.stats)
        return result
