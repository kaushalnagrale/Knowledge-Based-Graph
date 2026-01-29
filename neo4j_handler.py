"""
Neo4j Handler Module
Handles all Neo4j database operations for knowledge graph storage
"""

from typing import List, Dict, Optional
from py2neo import Graph, Node, Relationship
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jHandler:
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j database connection
        
        Args:
            uri: Neo4j database URI (e.g., 'bolt://localhost:7687')
            user: Database username
            password: Database password
        """
        try:
            self.graph = Graph(uri, auth=(user, password))
            logger.info(f"Successfully connected to Neo4j at {uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def clear_database(self):
        """Clear all nodes and relationships from the database"""
        try:
            self.graph.delete_all()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise
    
    def store_triples(self, triples: List[Dict[str, str]], graph_name: str = "default"):
        """
        Store knowledge triples in Neo4j
        
        Args:
            triples: List of dictionaries with 'subject', 'predicate', 'object' keys
            graph_name: Optional name to tag the graph
        """
        if not triples:
            logger.warning("No triples to store")
            return
        
        try:
            for triple in triples:
                subject = triple['subject']
                predicate = triple['predicate']
                obj = triple['object']
                
                # Create Cypher query to merge nodes and relationships
                query = """
                MERGE (s:Entity {name: $subject})
                SET s.graph_name = $graph_name
                MERGE (o:Entity {name: $object})
                SET o.graph_name = $graph_name
                MERGE (s)-[r:RELATION {type: $predicate, graph_name: $graph_name}]->(o)
                RETURN s, r, o
                """
                
                self.graph.run(
                    query,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    graph_name=graph_name
                )
            
            logger.info(f"Successfully stored {len(triples)} triples in graph '{graph_name}'")
            
        except Exception as e:
            logger.error(f"Error storing triples: {e}")
            raise
    
    def query_entity_relationships(self, entity_name: str, graph_name: str = None) -> List[Dict]:
        """
        Query all relationships for a specific entity
        
        Args:
            entity_name: Name of the entity to query
            graph_name: Optional graph name filter
            
        Returns:
            List of relationship dictionaries
        """
        try:
            if graph_name:
                query = """
                MATCH (e:Entity {name: $name, graph_name: $graph_name})-[r]->(o)
                RETURN e.name as subject, r.type as predicate, o.name as object
                UNION
                MATCH (s)-[r]->(e:Entity {name: $name, graph_name: $graph_name})
                RETURN s.name as subject, r.type as predicate, e.name as object
                """
                results = self.graph.run(query, name=entity_name, graph_name=graph_name).data()
            else:
                query = """
                MATCH (e:Entity {name: $name})-[r]->(o)
                RETURN e.name as subject, r.type as predicate, o.name as object
                UNION
                MATCH (s)-[r]->(e:Entity {name: $name})
                RETURN s.name as subject, r.type as predicate, e.name as object
                """
                results = self.graph.run(query, name=entity_name).data()
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying entity relationships: {e}")
            return []
    
    def get_all_entities(self, graph_name: str = None) -> List[str]:
        """
        Get all entity names from the database
        
        Args:
            graph_name: Optional graph name filter
            
        Returns:
            List of entity names
        """
        try:
            if graph_name:
                query = "MATCH (e:Entity {graph_name: $graph_name}) RETURN e.name as name"
                results = self.graph.run(query, graph_name=graph_name).data()
            else:
                query = "MATCH (e:Entity) RETURN e.name as name"
                results = self.graph.run(query).data()
            
            return [result['name'] for result in results]
            
        except Exception as e:
            logger.error(f"Error getting entities: {e}")
            return []
    
    def get_all_triples(self, graph_name: str = None) -> List[Dict[str, str]]:
        """
        Get all triples from the database
        
        Args:
            graph_name: Optional graph name filter
            
        Returns:
            List of triple dictionaries
        """
        try:
            if graph_name:
                query = """
                MATCH (s:Entity {graph_name: $graph_name})-[r:RELATION]->(o:Entity)
                RETURN s.name as subject, r.type as predicate, o.name as object
                """
                results = self.graph.run(query, graph_name=graph_name).data()
            else:
                query = """
                MATCH (s:Entity)-[r:RELATION]->(o:Entity)
                RETURN s.name as subject, r.type as predicate, o.name as object
                """
                results = self.graph.run(query).data()
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting triples: {e}")
            return []
    
    def find_shortest_path(self, start_entity: str, end_entity: str, graph_name: str = None) -> List[Dict]:
        """
        Find shortest path between two entities
        
        Args:
            start_entity: Starting entity name
            end_entity: Ending entity name
            graph_name: Optional graph name filter
            
        Returns:
            List of nodes and relationships in the path
        """
        try:
            if graph_name:
                query = """
                MATCH path = shortestPath(
                    (start:Entity {name: $start, graph_name: $graph_name})-[*]->
                    (end:Entity {name: $end, graph_name: $graph_name})
                )
                RETURN [node in nodes(path) | node.name] as nodes,
                       [rel in relationships(path) | rel.type] as relationships
                """
                result = self.graph.run(
                    query,
                    start=start_entity,
                    end=end_entity,
                    graph_name=graph_name
                ).data()
            else:
                query = """
                MATCH path = shortestPath(
                    (start:Entity {name: $start})-[*]->(end:Entity {name: $end})
                )
                RETURN [node in nodes(path) | node.name] as nodes,
                       [rel in relationships(path) | rel.type] as relationships
                """
                result = self.graph.run(query, start=start_entity, end=end_entity).data()
            
            return result
            
        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            return []
    
    def get_graph_stats(self, graph_name: str = None) -> Dict:
        """
        Get statistics about the knowledge graph
        
        Args:
            graph_name: Optional graph name filter
            
        Returns:
            Dictionary with graph statistics
        """
        try:
            stats = {}
            
            # Count nodes
            if graph_name:
                node_query = "MATCH (e:Entity {graph_name: $graph_name}) RETURN count(e) as count"
                stats['num_entities'] = self.graph.run(node_query, graph_name=graph_name).data()[0]['count']
                
                # Count relationships
                rel_query = "MATCH ()-[r:RELATION {graph_name: $graph_name}]->() RETURN count(r) as count"
                stats['num_relationships'] = self.graph.run(rel_query, graph_name=graph_name).data()[0]['count']
            else:
                node_query = "MATCH (e:Entity) RETURN count(e) as count"
                stats['num_entities'] = self.graph.run(node_query).data()[0]['count']
                
                rel_query = "MATCH ()-[r:RELATION]->() RETURN count(r) as count"
                stats['num_relationships'] = self.graph.run(rel_query).data()[0]['count']
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {}
    
    def delete_graph(self, graph_name: str):
        """
        Delete all nodes and relationships for a specific graph
        
        Args:
            graph_name: Name of the graph to delete
        """
        try:
            query = """
            MATCH (e:Entity {graph_name: $graph_name})
            DETACH DELETE e
            """
            self.graph.run(query, graph_name=graph_name)
            logger.info(f"Successfully deleted graph '{graph_name}'")
            
        except Exception as e:
            logger.error(f"Error deleting graph: {e}")
            raise


if __name__ == "__main__":
    # Test the Neo4j handler
    # Note: Requires Neo4j to be running
    try:
        handler = Neo4jHandler(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password"
        )
        
        # Test data
        test_triples = [
            {'subject': 'Einstein', 'predicate': 'developed', 'object': 'relativity'},
            {'subject': 'Einstein', 'predicate': 'won', 'object': 'Nobel Prize'},
            {'subject': 'Einstein', 'predicate': 'was born in', 'object': 'Germany'},
        ]
        
        # Store triples
        handler.store_triples(test_triples, graph_name="test_graph")
        
        # Get stats
        stats = handler.get_graph_stats(graph_name="test_graph")
        print(f"Graph stats: {stats}")
        
        # Query entity
        relationships = handler.query_entity_relationships("Einstein", graph_name="test_graph")
        print(f"Einstein relationships: {relationships}")
        
    except Exception as e:
        print(f"Error testing Neo4j handler: {e}")
        print("Make sure Neo4j is running and credentials are correct")