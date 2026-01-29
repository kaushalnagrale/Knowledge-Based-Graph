"""
Knowledge Graph Extraction System
Main Streamlit Application
"""

import streamlit as st
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import json

from kg_extractor import KnowledgeExtractor
from kg_graph_builder import GraphBuilder
from neo4j_handler import Neo4jHandler

# Load environment variables
load_dotenv()

# Page configuration
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E75B6;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5B9BD5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E75B6;
        color: white;
        font-weight: bold;
    }
    .triple-card {
    color: #000000;
    font-size: 0.95rem;
}
        
    .triple-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: black;              /* ✅ ADD THIS */
    }
    .triple-card b {
        color: black;              /* ✅ ADD THIS */
    }
    </style>
""", unsafe_allow_html=True)



def initialize_session_state():
    """Initialize session state variables"""
    if 'triples' not in st.session_state:
        st.session_state.triples = []
    if 'graph' not in st.session_state:
        st.session_state.graph = None
    if 'neo4j_connected' not in st.session_state:
        st.session_state.neo4j_connected = False


def connect_neo4j():
    neo4j_uri = os.getenv("NEO4J_URI")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    try:
        handler = Neo4jHandler(
            neo4j_uri,
            neo4j_user,
            neo4j_password
        )

        st.session_state.neo4j_handler = handler
        st.session_state.neo4j_connected = True
        return True

    except Exception as e:
        st.error(f"Neo4j connection error: {e}")
        return False


def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">🧠 Knowledge Graph Extraction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract structured knowledge from text using NLP and LLMs</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Extraction mode selection
        extraction_mode = st.radio(
            "Select Extraction Mode:",
            ["Online (LLM)", "Offline (spaCy)"],
            help="Online mode uses Groq API for better accuracy. Offline mode works without internet."
        )
        
        st.divider()
        
        # Neo4j settings
        st.subheader("🗄️ Neo4j Database")
        
        neo4j_enabled = st.checkbox("Enable Neo4j Storage", value=False)
        
        if neo4j_enabled:
            if st.button("Connect to Neo4j"):
                with st.spinner("Connecting to Neo4j..."):
                    if connect_neo4j():
                        st.success("✅ Connected to Neo4j!")
                    else:
                        st.error("❌ Connection failed. Check your credentials.")
            
            if st.session_state.neo4j_connected:
                st.success("🟢 Neo4j Connected")
                
                # Graph name input
                graph_name = st.text_input(
                    "Graph Name:",
                    value="my_graph",
                    help="Give your knowledge graph a name"
                )
                st.session_state.graph_name = graph_name
        
        st.divider()
        
        # API Key for LLM mode
        if extraction_mode == "Online (LLM)":
            st.subheader("🔑 API Configuration")
            groq_api_key = st.text_input(
                "Groq API Key:",
                type="password",
                value=os.getenv('GROQ_API_KEY', ''),
                help="Enter your Groq API key"
            )
        else:
            groq_api_key = None
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.info(
            "This system extracts knowledge triples (subject-predicate-object) "
            "from text and visualizes them as a graph. Use LLM mode for better "
            "accuracy or spaCy mode for offline operation."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Text")
        
        # Sample text
        sample_text = """Albert Einstein was a German-born theoretical physicist who developed 
the theory of relativity. He was born in Ulm, Germany in 1879. Einstein won the Nobel Prize 
in Physics in 1921 for his explanation of the photoelectric effect. He moved to the United 
States in 1933 and became an American citizen in 1940. Einstein is widely regarded as one 
of the most influential scientists of all time."""
        
        # Text input
        input_text = st.text_area(
            "Enter text to extract knowledge from:",
            value=sample_text,
            height=250,
            help="Enter any text and we'll extract knowledge triples from it"
        )
        
        # Extract button
        if st.button("🔍 Extract Knowledge", type="primary"):
            if not input_text.strip():
                st.error("Please enter some text to extract knowledge from.")
            else:
                with st.spinner("Extracting knowledge..."):
                    try:
                        # Initialize extractor
                        extractor = KnowledgeExtractor(groq_api_key=groq_api_key)
                        
                        # Extract triples
                        mode = "llm" if extraction_mode == "Online (LLM)" else "spacy"
                        triples = extractor.extract(input_text, mode=mode)
                        
                        if not triples:
                            st.warning("No knowledge triples were extracted. Try with different text.")
                        else:
                            st.session_state.triples = triples
                            
                            # Build graph
                            builder = GraphBuilder()
                            graph = builder.build_graph(triples)
                            st.session_state.graph = graph
                            st.session_state.builder = builder
                            
                            st.success(f"✅ Extracted {len(triples)} knowledge triples!")
                            
                            # Store in Neo4j if enabled
                            if neo4j_enabled and st.session_state.neo4j_connected:
                                try:
                                    st.session_state.neo4j_handler.store_triples(
                                        triples,
                                        graph_name=st.session_state.graph_name
                                    )
                                    st.success(f"💾 Saved to Neo4j as '{st.session_state.graph_name}'")
                                except Exception as e:
                                    st.error(f"Failed to save to Neo4j: {e}")
                    
                    except Exception as e:
                        st.error(f"Error during extraction: {e}")
    
    with col2:
        st.header("📊 Extracted Triples")
        
        if st.session_state.triples:
            # Display triples
            for i, triple in enumerate(st.session_state.triples, 1):
                st.markdown(f"""
                <div class="triple-card">
                    <b>Triple {i}:</b><br>
                    <b>Subject:</b> {triple['subject']}<br>
                    <b>Predicate:</b> {triple['predicate']}<br>
                    <b>Object:</b> {triple['object']}
                </div>
                """, unsafe_allow_html=True)
            
            # Download button
            json_data = json.dumps(st.session_state.triples, indent=2)
            st.download_button(
                label="📥 Download Triples (JSON)",
                data=json_data,
                file_name="knowledge_triples.json",
                mime="application/json"
            )
        else:
            st.info("👆 Extract knowledge from text to see triples here")
    
    # Graph visualization section
    if st.session_state.graph is not None:
        st.header("🕸️ Knowledge Graph Visualization")
        
        tab1, tab2, tab3 = st.tabs(["📈 Graph", "📊 Statistics", "🔍 Query"])
        
        with tab1:
            # Visualize graph
            fig = st.session_state.builder.visualize()
            st.pyplot(fig)
            
            # Download graph button
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Graph (PNG)",
                data=img_buffer,
                file_name="knowledge_graph.png",
                mime="image/png"
            )
        
        with tab2:
            # Display graph statistics
            stats = st.session_state.builder.get_graph_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Nodes", stats.get('num_nodes', 0))
            with col2:
                st.metric("Edges", stats.get('num_edges', 0))
            with col3:
                st.metric("Density", f"{stats.get('density', 0):.3f}")
            
            # Central nodes
            st.subheader("Most Central Entities")
            central_nodes = st.session_state.builder.get_central_nodes(5)
            for node, score in central_nodes:
                st.write(f"**{node}**: {score:.3f}")
        
        with tab3:
            # Query interface for Neo4j
            if st.session_state.neo4j_connected:
                st.subheader("Query Neo4j Database")
                
                query_entity = st.text_input("Enter entity name to query:")
                
                if st.button("🔍 Query Entity"):
                    if query_entity:
                        with st.spinner("Querying..."):
                            results = st.session_state.neo4j_handler.query_entity_relationships(
                                query_entity,
                                graph_name=st.session_state.graph_name
                            )
                            
                            if results:
                                st.write(f"Found {len(results)} relationships:")
                                for result in results:
                                    st.write(f"**{result['subject']}** → *{result['predicate']}* → **{result['object']}**")
                            else:
                                st.info("No relationships found for this entity.")
                
                # Show all entities in database
                if st.checkbox("Show all entities in database"):
                    entities = st.session_state.neo4j_handler.get_all_entities(
                        graph_name=st.session_state.graph_name
                    )
                    st.write(f"**Entities ({len(entities)}):**", ", ".join(entities))
            else:
                st.info("Connect to Neo4j to enable querying features.")


if __name__ == "__main__":
    # Import required for streamlit
    import io
    main()