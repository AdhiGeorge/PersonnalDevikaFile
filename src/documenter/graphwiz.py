import graphviz
import os
from typing import Dict, List, Any

class GraphVisualizer:
    def __init__(self, name: str):
        self.graph = graphviz.Digraph(name)
        self.graph.attr(rankdir='TB')
        
    def add_node(self, node_id: str, label: str, **attrs):
        """Add a node to the graph."""
        self.graph.node(node_id, label, **attrs)
        
    def add_edge(self, from_node: str, to_node: str, label: str = None, **attrs):
        """Add an edge between nodes."""
        self.graph.edge(from_node, to_node, label, **attrs)
        
    def add_subgraph(self, name: str, nodes: List[str], **attrs):
        """Add a subgraph containing the specified nodes."""
        with self.graph.subgraph(name=name) as s:
            s.attr(**attrs)
            for node in nodes:
                s.node(node)
                
    def save(self, filename: str = None, format: str = 'png'):
        """Save the graph to a file."""
        if filename is None:
            filename = f"graph_{name}"
            
        # Ensure the graphs directory exists
        graphs_dir = os.path.join(os.getcwd(), "data", "graphs")
        os.makedirs(graphs_dir, exist_ok=True)
        
        # Save the graph
        filepath = os.path.join(graphs_dir, filename)
        self.graph.render(filepath, format=format, cleanup=True)
        return f"{filepath}.{format}" 