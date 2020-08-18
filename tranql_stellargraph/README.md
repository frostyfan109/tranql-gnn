# TranQL StellarGraph Models
This is a collection of models created using TranQL knowledge graphs as datasets.
The two models of interest are `build_complex.py` and `build_gcn.py`. The others
did not end up going anywhere.

## Structure of a model
`build_complex.py` and `build_gcn.py` offer the following functions:
- `get_dataset(knowledge_graph: KnowledgeGraph) -> StellarGraph`
- `make_model(dataset: StellarGraph) -> Model`
- `make_prediction(model: Model, dataset: StellarGraph, edge: tuple)`
### Making a prediction
```
python
from build_complex import get_dataset, make_model, make_prediction
from tranql_jupyter import KnowledgeGraph

kg = KnowledgeGraph.mock1()
dataset = get_dataset(kg)
model = make_model(dataset)

edge = ("HGNC:9591", "directly_interacts_with", "CHEBI:36314")
print(make_prediction(model, dataset, edge)
```

## Running through CLI
These models can be run in the same manner as described in the main README.
They do not take any arguments.

For example:
```
python build_complex.py
python build_gcn.py
```
However, this will only build the model and output its metrics. For using the
model to make predictions and such, refer to the section "Making predictions"

## Notebooks
This directory also contains various notebooks detailing usage of the models,
which can be run like so:
```
export PYTHONPATH=$PWD
jupyter notebook
```

## Graph data collection
There are a couple utilities within this directory for collecting sets of graphs.
First is `data/data_collector.py`. This takes query sets in `data/query_sets`,
runs them, and turns them into response sets in `data/response_sets`. A query
set is a YAML file of this structure:
```
type: object
properties:
  info:
    type: object
    optional: true
    properties:
      name:
        type: string
      description:
        type: string
  queries:
    type: list
    items:
      type: string
```
Hopefully this schema definition is understandable. If still confused about
how they work and how to make one, take a quick look at one of them; they aren't
very large.

To run `data/data_collector.py`, do `python data/data_collector.py`.
There are two optional arguments:
- `-a`/`--api` specifies the URL to the TranQL API, which the script uses to
  make queries. This defaults to `http://localhost:8001`.
- `-r`/`--remake` specifies whether or not to rerun all the query sets
  (that have already been built). If you just want to rerun one query set
  because of a small change, it's probably best to just delete it in
  `data/response_sets`. This argument defaults to `False`.
  
Then, to use a response set created by `data_collector.py`, you can use
`run_response_set.py`. This script takes a response set and model as input
and trains said model using the union of the graphs in the response set.
To run it, do `python run_response_set.py --source {file_name} --model {file_name}`.
There are three arguments overall:
- `-s`/`--source` specifies the response set file path (relative to `data/response_sets`)
- `-m`/`--model` specifies the file path to the python module which builds the model.
- `-o`/`--output` specifies the file path to save the model and data set. This argument is optional.

In order to ease the process of loading the model and data set at the same time,
`run_response_set.py` has the class `ModelSerializer`. You can use it as follows:
```
python run_response_set.py --source airway_lung_disease.yaml --model build_complex.py --output my_model
python
from run_response_set import ModelSerializer
model, dataset = ModelSerializer.load("my_model")
```