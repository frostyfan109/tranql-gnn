import json

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
    nodes = get_nodes(k_graph)
    for node in nodes:
        type = node["type"]
        for i in list(node.keys()):
            del node[i]
        for n_type in type:
            node[n_type] = True

    return nodes