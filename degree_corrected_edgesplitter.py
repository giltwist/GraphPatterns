from stellargraph.data import EdgeSplitter
import numpy as np

# Identical to original StellarGraph EdgeSplitter except in how it samples negative edges
# Key difference is that nodes appear in the possible source list degree(node) times instead of just once.
# Inspired by https://arxiv.org/abs/2405.14985
# More info https://skojaku.github.io/posts/degree-bias-in-link-prediction/ 
class DegreeCorrectedEdgeSplitter(EdgeSplitter):

    	# Constructor 
    def __init__(self, g, g_master=None):
        super().__init__(g, g_master)

    def _sample_negative_examples_by_edge_type_global(
            self, edges, edge_label, p=0.5, limit_samples=None
        ):
            """
            This method produces a list of edges that don't exist in graph self.g (negative examples). The number of
            negative edges produced is equal to the number of edges with label edge_label in the graph times p (that should
            be in the range (0,1] or limited to maximum limit_samples if the latter is not None. The negative samples are
            between node types as inferred from the edge type of the positive examples previously removed from the graph
            and given in edges_positive.

            The source graph is not modified.

            Args:
                edges (list): The positive edge examples that have previously been removed from the graph
                edge_label (str): The edge type to sample negative examples of
                p (float): Factor that multiplies the number of edges in the graph and determines the number of negative
                edges to be sampled.
                limit_samples (int, optional): It limits the maximum number of samples to the given number, if not None

            Returns:
                (list) A list of 2-tuples that are pairs of node IDs that don't have an edge between them in the graph.
            """
            self.negative_edge_node_distances = []
            # determine the number of edges in the graph that have edge_label type
            # Multiply this number by p to determine the number of positive edge examples to sample
            all_edges = self._get_edges(edge_label=edge_label)
            num_edges_total = len(all_edges)
            print("Network has {} edges of type {}".format(num_edges_total, edge_label))
            #
            num_edges_to_sample = int(num_edges_total * p)

            if limit_samples is not None:
                if num_edges_to_sample > limit_samples:
                    num_edges_to_sample = limit_samples

            edge_source_target_node_types = self._get_edge_source_and_target_node_types(
                edges=edges
            )

            # to speed up lookup of edges in edges list, create a set the values stored are the concatenation of
            # the source and target node ids.
            edges_set = set(edges)
            edges_set.update({(u[1], u[0]) for u in edges})
            sampled_edges_set = set()

            # CHANGED CODE HERE
            #start_nodes = list(self.g.nodes(data=True))
            
            start_nodes = []
            for node in self.g.nodes(data=True):
                for i in range(len(list(self.g.neighbors(node[0])))):
                    start_nodes.append(node)

            # END CHANGED CODE

            end_nodes = list(self.g.nodes(data=True))

            count = 0
            sampled_edges = []

            # This line was also changed from start_nodes to end_nodes because of the above change.
            num_iter = int(np.ceil(num_edges_to_sample / (1.0 * len(end_nodes)))) + 1

            for _ in np.arange(0, num_iter):
                self._random.shuffle(start_nodes)
                self._random.shuffle(end_nodes)
                for u, v in zip(start_nodes, end_nodes):
                    u_v_edge_type = (u[1]["label"], v[1]["label"])
                    if (
                        (u_v_edge_type in edge_source_target_node_types)
                        and (u != v)
                        and ((u[0], v[0]) not in edges_set)
                        and ((u[0], v[0]) not in sampled_edges_set)
                    ):
                        sampled_edges.append(
                            (u[0], v[0], 0)
                        )  # the last entry is the class label
                        sampled_edges_set.update({(u[0], v[0]), (v[0], u[0])})
                        count += 1

                        if count == num_edges_to_sample:
                            return sampled_edges

            if len(sampled_edges) != num_edges_to_sample:
                raise ValueError(
                    "Unable to sample {} negative edges. Consider using smaller value for p.".format(
                        num_edges_to_sample
                    )
                )