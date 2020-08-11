import json
import requests

def get_nodes(k_graph):
    return json.loads(json.dumps(k_graph.build_knowledge_graph()["nodes"]))
    # return [n[1]["attr_dict"] for n in k_graph.net.nodes(data=True)]


def format0(k_graph):
    nodes = get_nodes(k_graph)
    for node in nodes:
        """ Reformat attributes to be meaningful data """
        ontology = node["id"].split(":")[0]
        # While the id of a node alone isn't useful whatsoever because it is unique to only said node,
        # the ontology of the node may offer some value
        # node["ontology"] = ontology

        """ Filter attributes """
        for attr in list(node.keys()):
            if attr in ["name", "id", "omnicorp_article_count", "equivalent_identifiers", "reasoner"]:
                # Also remove select predetermined attributes that have no significance and may mislead the model
                del node[attr]
        for attr in list(node.keys()):
            if isinstance(node[attr], list):
                # If attribute is a list, convert it to be vectorizable
                for x in node[attr]:
                    node[attr+"="+str(x)] = True
                del node[attr]

    return nodes


def format1(k_graph):
    nodes = get_nodes(k_graph)
    for node in nodes:
        ontology = node["id"].split(":")[0]
        for attr in list(node.keys()):
            if attr == "type":
                for x in node[attr]:
                    node[attr + "=" + str(x)] = True
                del node[attr]
            else: del node[attr]
        """
            elif not isinstance(node[attr], (bool)):
                del node[attr]
        node["ontology"] = ontology
        """
    return nodes


def format2(k_graph):
    """ Try to incorporate a good portion of the node's attributes as its feature vector """
    nodes = get_nodes(k_graph)
    for node in nodes:
        ontology = node["id"].split(":")[0]
        # node["ontology"] = ontology
        for attr in list(node.keys()):
            if attr in ["id", "name", "omnicorp_article_count", "reasoner", "equivalent_identifiers", "molecule_properties"]:
                del node[attr]
            elif isinstance(node[attr], list):
                if attr == "type":
                    for x in node[attr]:
                        node[attr + "=" + str(x)] = True
                del node[attr]
            elif isinstance(node[attr], (dict, str)):
                del node[attr]

    return nodes


def format3(k_graph):
    """ Just use the node's types as its feature vector """
    nodes = get_nodes(k_graph)
    for node in nodes:
        type = node["type"]
        for i in list(node.keys()):
            del node[i]
        for n_type in type:
            node[n_type] = True

    return nodes

def neighborhood_format(k_graph):
    """ Encode information about a node's neighborhood as its feature vector """
    USE_CONNECTED_NODES = True
    USE_TYPE_COUNT = False
    USE_NODE_TYPE = True
    USE_ONTOLOGY = True

    USE_ONTO_PARENTS = True
    USE_ONTO_SIBLINGS = False
    USE_ONTO_CHILDREN = False
    USE_ONTO_ANCESTORS = False

    USE_NODE_ATTRIBUTES = True

    nodes = get_nodes(k_graph)
    for i, node in enumerate(nodes):
        id = node["id"]
        attributes = {
            "connected_nodes": {},
            "type_count": {}, # keep track of the connected nodes' types
            "node_type": node["type"], # keep track of node's type,
            "ontology": id.split(":")[0], # keep track of the node's ontology,
        }
        connected_to = k_graph.net[id]
        for connected_node_id in connected_to:
            edges = connected_to[connected_node_id]
            attributes["connected_nodes"][connected_node_id] = len(edges)

            for type in k_graph.net.nodes[connected_node_id]["attr_dict"]["type"]:
                if type not in attributes["type_count"]: attributes["type_count"][type] = 0
                attributes["type_count"][type] += 1

        if USE_ONTO_PARENTS:
            res = requests.get(f"https://onto.renci.org/parents/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_parents"] = res.json().get("parents", [])
        if USE_ONTO_SIBLINGS:
            res = requests.get(f"https://onto.renci.org/siblings/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_siblings"] = res.json().get("siblings", [])
        if USE_ONTO_CHILDREN:
            res = requests.get(f"https://onto.renci.org/children/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_children"] = res.json()
        if USE_ONTO_ANCESTORS:
            res = requests.get(f"https://onto.renci.org/ancestors/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_ancestors"] = res.json()

        if USE_NODE_ATTRIBUTES and "gene" in node["type"]:
            attributes["node_attr"] = {}
            for attr in node:
                val = node[attr]
                if isinstance(val, str):
                    attributes["node_attr"][attr] = val

        """
        To avoid large chunks of sporadic commenting, let's just always create the
        non-performance-heavy attributes and delete them here if they're disabled.
        ONTO requests take a lot of time, so they have to be made conditionally.
        """
        if not USE_CONNECTED_NODES:
            del attributes["connected_nodes"]
        if not USE_TYPE_COUNT:
            del attributes["type_count"]
        if not USE_NODE_TYPE:
            del attributes["node_type"]
        if not USE_ONTOLOGY:
            del attributes["ontology"]

        """ Go through the attributes dict and convert the dictionary/list values into a vectorizable form """
        for attr in list(attributes.keys()):
            if isinstance(attributes[attr], dict):
                for key in attributes[attr]:
                    attributes[attr + "=" + key] = attributes[attr][key]
                del attributes[attr]
            elif isinstance(attributes[attr], list):
                for x in attributes[attr]:
                    attributes[attr + "=" + x] = True
                del attributes[attr]

        nodes[i] = attributes

    return nodes