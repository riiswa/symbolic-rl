import os
import webbrowser
from dataclasses import dataclass
from typing import Dict, List
from urllib.parse import urljoin
from urllib.request import pathname2url
from pyvis.network import Network

from utils import merge_yaml_data


class KnowledgeNode:
    @dataclass
    class Link:
        to: "KnowledgeNode"
        link_name: str
        weight: int

    def __init__(self, id: str, label: str):
        self.id = id
        self.label = label
        self.links: List[KnowledgeNode.Link] = []
        self.weight = 1

    def link(self, other: "KnowledgeNode", link_name: str):
        may_link = list(filter(lambda link: link.to == other and link.link_name == link_name, self.links))
        if may_link:
            may_link[0].weight += 1
        else:
            self.links.append(self.Link(other, link_name, 1))


class Knowledge:
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}

    def add(self, node: KnowledgeNode):
        self.nodes[node.id] = node

    def __getitem__(self, id: str) -> KnowledgeNode:
        return self.nodes[id]

    def from_paths(self, filepaths: List[str]):
        knowledge = merge_yaml_data(filepaths)

        for node in knowledge["nodes"]:
            self.add(KnowledgeNode(node["id"], node["lbl"]))
        for edge in knowledge["edges"]:
            self[edge["subj"]].link(self[edge["obj"]], edge["pred"])

    def show(self):
        net = Network(directed=True)
        for node in self.nodes.values():
            net.add_node(node.id, label=node.label, shape='box')
            for link in node.links:
                net.add_edge(node.id, link.to.id, label=link.link_name, weight=link.weight)

        net.toggle_physics(True)

        output_file = os.path.join("output", "ontology.html")

        if not os.path.exists("output"):
            os.makedirs("output")
        net.show(os.path.join("output", "ontology.html"))
        webbrowser.open(urljoin('file:', pathname2url(os.path.abspath(output_file))), new=2)


if __name__ == "__main__":
    from sys import argv

    if len(argv) > 1:
        knowledge = Knowledge()
        knowledge.from_paths(argv[1:])
        knowledge.show()
    else:
        print("Please provide at least one file")
