# Various testing GNN models

#### In a virtual environment:

Building a model:
```
cd {network}
pip install -r requirements.txt
python {model}.py
```
Running a notebook (make sure that ipykernel is set up if using a virtual
environment):
```
cd {network}
export PYTHONPATH=$PWD
jupyter notebook
```
It is important that the PYTHONPATH environment variable is correctly set to the
location of the network or else notebooks won't function. Jupyter notebook
does not have to be run within the network directory.
