from sys import argv
from pyvis.network import Network
import os
from ontology_loader import load_ontology
import webbrowser
from urllib.parse import urljoin
from urllib.request import pathname2url


def show_ontology(args):
    ontology = load_ontology(args)
    net = Network(directed=True)
    for node in ontology["nodes"]:
        net.add_node(node["id"], label=node["lbl"], shape='box')
    for edge in ontology["edges"]:
        net.add_edge(edge["subj"], edge["obj"], label=edge["pred"])
    net.toggle_physics(True)

    output_file = os.path.join("output", "ontology.html")

    if not os.path.exists("output"):
        os.makedirs("output")
    net.show(os.path.join("output", "ontology.html"))
    webbrowser.open(urljoin('file:', pathname2url(os.path.abspath(output_file))), new=2)


if __name__ == "__main__":
    if len(argv) > 1:
        show_ontology(argv[1:])
    else:
        print("Please provide at least one file")
