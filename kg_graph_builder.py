"""
Graph Builder Module
Builds and visualizes knowledge graphs from triples
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict
import io


class GraphBuilder:
    def __init__(self):
        """Initialize the graph builder"""
        self.graph = None
    
    def build_graph(self, triples: List[Dict[str, str]]) -> nx.DiGraph:
        """
        Build a directed graph from knowledge triples
        
        Args:
            triples: List of dictionaries with 'subject', 'predicate', 'object' keys
            
        Returns:
            NetworkX directed graph
        """
        self.graph = nx.DiGraph()
        
        for triple in triples:
            subject = triple['subject']
            predicate = triple['predicate']
            obj = triple['object']
            
            # Add edge with predicate as label
            self.graph.add_edge(subject, obj, label=predicate)
        
        return self.graph
    
    def visualize(self, figsize=(14, 10), save_path=None):
        """
        Visualize the knowledge graph
        
        Args:
            figsize: Figure size as tuple (width, height)
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if self.graph is None or len(self.graph.nodes()) == 0:
            print("No graph to visualize. Build a graph first.")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate layout
        if len(self.graph.nodes()) <= 10:
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        else:
            pos = nx.spring_layout(self.graph, k=1, iterations=50, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, 
            pos,
            node_color='lightblue',
            node_size=3000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            width=2,
            alpha=0.6,
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=10,
            font_weight='bold',
            font_family='sans-serif',
            ax=ax
        )
        
        # Draw edge labels (predicates)
        edge_labels = nx.get_edge_attributes(self.graph, 'label')
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels,
            font_size=8,
            font_color='red',
            ax=ax
        )
        
        ax.set_title("Knowledge Graph Visualization", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def get_graph_stats(self) -> Dict:
        """
        Get statistics about the graph
        
        Returns:
            Dictionary with graph statistics
        """
        if self.graph is None:
            return {}
        
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_connected_components': nx.number_weakly_connected_components(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            'density': nx.density(self.graph)
        }
    
    def get_central_nodes(self, top_n: int = 5) -> List[tuple]:
        """
        Get the most central nodes in the graph
        
        Args:
            top_n: Number of top nodes to return
            
        Returns:
            List of (node, centrality_score) tuples
        """
        if self.graph is None or len(self.graph.nodes()) == 0:
            return []
        
        # Calculate degree centrality
        centrality = nx.degree_centrality(self.graph)
        
        # Sort by centrality score
        sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_nodes[:top_n]
    
    def find_paths(self, source: str, target: str) -> List[List[str]]:
        """
        Find all simple paths between two nodes
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            List of paths (each path is a list of nodes)
        """
        if self.graph is None:
            return []
        
        try:
            paths = list(nx.all_simple_paths(self.graph, source, target))
            return paths
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            return []
    
    def export_to_dict(self) -> Dict:
        """
        Export graph to dictionary format
        
        Returns:
            Dictionary representation of the graph
        """
        if self.graph is None:
            return {'nodes': [], 'edges': []}
        
        nodes = [{'id': node} for node in self.graph.nodes()]
        edges = [
            {
                'source': edge[0],
                'target': edge[1],
                'label': self.graph.edges[edge].get('label', '')
            }
            for edge in self.graph.edges()
        ]
        
        return {'nodes': nodes, 'edges': edges}


if __name__ == "__main__":
    # Test the graph builder
    test_triples = [
        {'subject': 'Einstein', 'predicate': 'developed', 'object': 'relativity theory'},
        {'subject': 'Einstein', 'predicate': 'won', 'object': 'Nobel Prize'},
        {'subject': 'Einstein', 'predicate': 'was born in', 'object': 'Germany'},
        {'subject': 'Nobel Prize', 'predicate': 'awarded in', 'object': '1921'},
    ]
    
    builder = GraphBuilder()
    graph = builder.build_graph(test_triples)
    
    print("Graph Statistics:")
    stats = builder.get_graph_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nMost Central Nodes:")
    central = builder.get_central_nodes(3)
    for node, score in central:
        print(f"  {node}: {score:.3f}")
    
    # Visualize
    fig = builder.visualize()
    plt.savefig('/home/claude/test_graph.png', dpi=300, bbox_inches='tight')
    print("\nGraph saved to test_graph.png")