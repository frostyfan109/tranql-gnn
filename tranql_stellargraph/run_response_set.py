from functools import reduce
from tranql_jupyter import KnowledgeGraph
import os
import json
import yaml
import networkx as nx
from tensorflow.keras.models import load_model

class ModelSerializer:
    FEATURE_NAME = "feature"

    @staticmethod
    def _gpickle_path(model_name):
        return os.path.join(model_name, "dataset.pickle")

    @staticmethod
    def load(model_name):
        from stellargraph import StellarGraph

        model = load_model(model_name)

        path = ModelSerializer._gpickle_path(model_name)
        nx_instance = nx.read_gpickle(path)
        dataset = StellarGraph.from_networkx(nx_instance, node_features=ModelSerializer.FEATURE_NAME)

        return model, dataset

    @staticmethod
    def save(model, model_name, dataset):
        # If model.save is throwing a ValueError "If specifying TensorSpec names for nested structures, either zero or all names have to be specified."
        # Then TensorFlow needs to be upgraded to >=2.3.0, where this bug was fixed
        # For explanation, see this issue: https://github.com/stellargraph/stellargraph/issues/1251
        model.save(model_name)

        # We're going to save the dataset in the same directory as the Keras model
        # This is a bit intrusive, but Keras doesn't seem to care.
        # The process of serializing a StellarGraph instance is as follows:
        #   1) Convert dataset to networkx instance with StellarGraph.to_networkx
        #   2) Pickle the networkx instance with networkx.write_gpickle
        # We can then unpickle it later. As a side note, this whole process is not
        # well-tested/definitively reliable. If worst comes to worst, I would recommend
        # simply rebuilding the dataset yourself like so:
        #   >>> from run_response_set import union_response_set
        #   >>> from whatever_model_was_used import get_dataset
        #   >>> import yaml
        #   >>> import os
        #   >>> file = open(os.path.join("data", "response_sets", "whatever_response_set.yaml"), "r")
        #   >>> response_set = yaml.safe_load(file)
        #   >>> dataset = get_dataset(union_response_set(response_set))
        path = ModelSerializer._gpickle_path(model_name)
        nx_instance = dataset.to_networkx(feature_attr=ModelSerializer.FEATURE_NAME)
        nx.write_gpickle(nx_instance, path)


def union_response_set(response_set):
    return union_response_sets([response_set])


def union_response_sets(response_sets):
    knowledge_graphs = []
    for response_set in response_sets:
        knowledge_graphs += [KnowledgeGraph(response["response"]["knowledge_graph"]) for response in response_set]

    union_graph = reduce(lambda a, b: a + b, knowledge_graphs)
    return union_graph


def create_model(response_sets, get_dataset, make_model, save=None):
    """ Creates a union graph, dataset, and model from single/list of response sets.
    Can also save the model/dataset if desired.

    :param response_sets: A file path or list of file paths to response sets
        These paths are relative to the current path. They are NOT relative to data/response_sets.
    :type response_sets: str, list
    :param get_dataset: A model's standard get_dataset function which returns a StellarGraph instance
    :type get_dataset: function
    :param make_model: A model's standard make_model function which returns a tensorflow.keras.Model instance
    :type make_model: function
    :param save: The model's name to be used in `Model.save`, or None
    :type save: str, None

    :return: The Keras model, StellarGraph dataset, and union of all knowledge graphs in the repsonse sets
    :rtype: tuple
    """
    # If a single response set is provided, turn it into a list for convenience
    if not isinstance(response_sets, list):
        response_sets = [response_sets]

    loaded_response_sets = []
    for response_set in response_sets:
        with open(response_set, "r") as file:
            loaded_response_sets.append(yaml.safe_load(file))


    union_graph = union_response_sets(loaded_response_sets)

    dataset = get_dataset(union_graph)
    model = make_model(dataset)

    if save != None:
        ModelSerializer.save(model, save, dataset)

    return model, dataset, union_graph

if __name__ == "__main__":
    import argparse
    from importlib.machinery import SourceFileLoader

    parser = argparse.ArgumentParser(
        description="Takes a response set of queries and runs a model using all of the responses",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--source",
        help="Response set file (relative to data/response_sets). May also pass in multiple response sets at once",
        nargs="+",
        required=True
    )
    parser.add_argument(
        "-m",
        "--model",
        help="""File path to the model module (relative to ./).
A model should implement the following top-level functions:
    get_dataset(kg: tranql_jupyter.KnowledgeGraph) -> stellargraph.StellarGraph
    make_model(dataset: stellargraph.StellarGraph) -> keras.Model""",
        required=True)
    parser.add_argument("-o", "--output", help="Save the Keras model in the current working directory", required=False)

    args = parser.parse_args()

    response_sets = args.source
    model_file = args.model
    output = args.output

    loader = SourceFileLoader(model_file, model_file)
    model_module = loader.load_module()

    response_sets = [os.path.join("data", "response_sets", resp) for resp in response_sets]

    model, dataset, union_graph = create_model(response_sets, model_module.get_dataset, model_module.make_model, save=output)