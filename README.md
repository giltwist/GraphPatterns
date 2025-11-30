Basic exploration of copurchasing patterns using Amazon data ( From https://snap.stanford.edu/data/amazon-meta.html ) and a GNN.  This branch utilizes HinSAGE

For parallelization run with:

export NETWORKX_AUTOMATIC_BACKENDS="parallel" && python graph_patterns.py

Note: Runs on Python 3.8