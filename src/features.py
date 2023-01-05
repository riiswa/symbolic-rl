from utils import *
from copy import deepcopy

class Feature:
    def __init__(self, label: str, type: str) -> None:
        self.label = label
        self.type = type
        self.score = {}

    def add_occurence(self, action_id:str, reward:float):
        if action_id in self.score:
            self.score[action_id] = ((self.score[action_id][0]+reward)/self.score[action_id][1]+1,self.score[action_id][1]+1)
        else:
            self.score.update({action_id: (reward,1)})
    
    @staticmethod
    def from_dict(obj: Any) -> 'Feature':
        assert isinstance(obj, dict)
        label = from_str(obj.get("label"))
        type = from_str(obj.get("type"))
        return Feature(label, type)

    def to_dict(self) -> dict:
        result: dict = {}
        result["label"] = from_str(self.label)
        result["type"] = from_str(self.type)
        return result

class Features:

    def __init__(self, features: List[Feature]) -> None:
        self.features = features
        self.default = deepcopy(features)

    @staticmethod
    def from_dict(obj: Any) -> 'Features':
        assert isinstance(obj, dict)
        features = from_list(Feature.from_dict, obj.get("features"))
        return Features(features)

    @staticmethod
    def from_paths(filepaths: List[str]) -> 'Features':
        return Features.from_dict(merge_yaml_data(filepaths))

    def to_dict(self) -> dict:
        result: dict = {}
        result["features"] = from_list(lambda x: to_class(Feature, x), self.features)
        return result
    
    def __getitem__(self, label):
        return  [s for s in self.features if s.label==label][0]

    def reload(self):
        self.features = deepcopy(self.default)