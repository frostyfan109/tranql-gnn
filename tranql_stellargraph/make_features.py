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


""" For `neighborhood_format`, a variety of features can incorporated that each offer their own benefits.
For the following explanations, `source_node` is the node that the attributes are being generated for. """

"""
This creates a dict of {target_node: multiplicity},
where multiplicity is the number of edges going from source_node->target_node.
E.g. given a graph CHEBI:X-[targets]->HGNC:Y,
                   CHEBI:X-[related_to]->HGNC:Y,
                   CHEBI:X-[literature_co-occurrence]->MONDO:Z
The connected_nodes attribute for CHEBI:X will be {"HGNC:Y": 2, "MONDO:Z": 1}
"""
USE_CONNECTED_NODES = True

""" This attribute simply keeps track of the types of nodes that source_node is connected to.
Remember that none of the models consider node types inherently; they have to be encoded into
the feature vector (StellarGraph does inherently support node types with certain algorithms,
however it does not support a node of multiple types (e.g. [disease, named_thing, genetic_condition])
So, for example, given the aforementioned graph:
    CHEBI:X-[targets]->HGNC:Y<gene>,
    CHEBI:X-[related_to]->HGNC:Y<gene>,
    CHEBI:X-[literature_co-occurrence]->MONDO:Z<disease, genetic_condition>
this attribute would be {"gene": 2, "disease": 1, "genetic_condition": 1}
"""
USE_TYPE_COUNT = False

""" This attribute incorporates source_node's type attribute specifically.
Given source_node is CHEBI:X, type=["chemical_substance"],
this attribute would be ["chemical_substance"]
"""
USE_NODE_TYPE = True

""" This attribute extracts the ontology from the node's id. It doesn't seem
to do much and will probably go unnoticed.
Given source_node is CHEBI:X, this attribute would be just "CHEBI"
"""
USE_ONTOLOGY = True

""" ONTO section. Makes requests to the ONTO API for node information. Keep in mind
that for each node in a graph, an ONTO request will be made for every attribute enabled.
This means that 8k requests will be made for a graph of 4000 nodes and siblings/parents turned on.

Not going to describe each ONTO attribute in detail, since they all do the same thing and are self-explanatory.

It's important to note that some node types don't work with ONTO, so the feature vector should not be made up of just
ONTO attributes. For example, an ontology for genes, such as HGNC, can't be structured as a descendant tree like ONTO
requires for a lot of its methods. Genes simply don't have parent genes. Some things still may work though; for example,
ONTO can still return the siblings of an HGNC gene.
"""
USE_ONTO_PARENTS = True
USE_ONTO_SIBLINGS = False
USE_ONTO_CHILDREN = False
USE_ONTO_ANCESTORS = False

""" As mentioned above, a problem with ONTO is that nodes of certain types will often end up with basically no 
information for their feature vector. This was an experiment to try to fix that. It tries to incorporate the node's
actual attributes if ONTO fails to return suitable data. """
NEVER = 0 # Never use the node's actual attributes
FAILED_ONTO = 1 # If ONTO fails, add the node's attributes to its feature vector
ALWAYS = 2 # Always add the node's attributes, even if ONTO succeeds (add them to every node in the graph)
USE_NODE_ATTRIBUTES = FAILED_ONTO # Set the behavior

""" This whole thing is a mess and deserves a cleanup """
def neighborhood_format(k_graph):
    """ Encode information about a node's neighborhood as its feature vector """
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

        onto_did_fail = False
        if USE_ONTO_PARENTS:
            res = requests.get(f"https://onto.renci.org/parents/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_parents"] = res.json().get("parents", [])
            if len(attributes.get("onto_parents", [])) == 0: onto_did_fail = True
        if USE_ONTO_SIBLINGS:
            res = requests.get(f"https://onto.renci.org/siblings/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_siblings"] = res.json().get("siblings", [])
            if len(attributes.get("onto_siblings", [])) == 0: onto_did_fail = True
        if USE_ONTO_CHILDREN:
            res = requests.get(f"https://onto.renci.org/children/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_children"] = res.json()
            if len(attributes.get("onto_children", [])) == 0: onto_did_fail = True
        if USE_ONTO_ANCESTORS:
            res = requests.get(f"https://onto.renci.org/ancestors/{id}", headers={"accept": "application/json"})
            if res.ok:
                attributes["onto_ancestors"] = res.json()
            if len(attributes.get("onto_ancestors", [])) == 0: onto_did_fail = True

        if USE_NODE_ATTRIBUTES == ALWAYS or (USE_NODE_ATTRIBUTES == FAILED_ONTO and onto_did_fail):
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