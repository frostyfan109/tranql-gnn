from functools import reduce
from tranql_jupyter import KnowledgeGraph

def create_model(response_sets, get_dataset, make_model):
    knowledge_graphs = []
    for response_set in response_sets:
        knowledge_graphs += [KnowledgeGraph(response["response"]["knowledge_graph"]) for response in response_set]

    union_graph = reduce(lambda a, b: a+b, knowledge_graphs)

    dataset = get_dataset(union_graph)
    model = make_model(dataset)

    return model

if __name__ == "__main__":
    import argparse
    import yaml
    import os
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

    response_set_names = args.source
    model_file = args.model
    output = args.output

    response_sets = []
    for response_set_name in response_set_names:
        with open(os.path.join("data", "response_sets", response_set_name), "r") as file:
            response_sets.append(yaml.safe_load(file))

    loader = SourceFileLoader(model_file, model_file)
    model_module = loader.load_module()

    model = create_model(response_sets, model_module.get_dataset, model_module.make_model)

    if output is not None:
        model.save(output)