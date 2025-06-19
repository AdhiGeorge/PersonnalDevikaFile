from graphviz import Digraph
import os
from typing import Dict, List, Any

class UMLGenerator:
    def __init__(self, name: str):
        self.graph = Digraph(name)
        self.graph.attr(rankdir='TB')
        self.graph.attr('node', shape='record')
        
    def add_class(self, class_name: str, attributes: List[str] = None, methods: List[str] = None):
        """Add a class to the UML diagram."""
        label = f"{class_name}"
        if attributes or methods:
            label += "\\n"
            if attributes:
                label += "\\n".join(f"+ {attr}" for attr in attributes)
            if methods:
                label += "\\n" + "\\n".join(f"+ {method}()" for method in methods)
        self.graph.node(class_name, label)
        
    def add_inheritance(self, child: str, parent: str):
        """Add an inheritance relationship."""
        self.graph.edge(child, parent, "inherits")
        
    def add_composition(self, container: str, component: str):
        """Add a composition relationship."""
        self.graph.edge(container, component, "contains")
        
    def add_association(self, source: str, target: str, label: str = None):
        """Add an association relationship."""
        self.graph.edge(source, target, label)
        
    def save(self, filename: str = None, format: str = 'png'):
        """Save the UML diagram to a file."""
        if filename is None:
            filename = f"uml_{name}"
            
        # Ensure the diagrams directory exists
        diagrams_dir = os.path.join(os.getcwd(), "data", "diagrams")
        os.makedirs(diagrams_dir, exist_ok=True)
        
        # Save the diagram
        filepath = os.path.join(diagrams_dir, filename)
        self.graph.render(filepath, format=format, cleanup=True)
        return f"{filepath}.{format}" 