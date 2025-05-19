import streamlit as st
import chromadb
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import plotly.express as px
import pandas as pd

def load_embeddings():
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path="chroma_db")
    
    # Get the collection
    collection = client.get_collection("documents")
    
    # Get all embeddings and their metadata
    results = collection.get(include=["embeddings", "metadatas", "documents"])
    
    # Check if embeddings exist and are not empty
    if not results or 'embeddings' not in results or len(results['embeddings']) == 0:
        st.error("No embeddings found in the database")
        return None
    
    return results

def create_visualization(results, method='t-SNE', perplexity=30, n_iter=1000, dimensions=2):
    # Convert embeddings to numpy array
    embeddings = np.array(results['embeddings'])
    
    # Apply dimensionality reduction
    if method == 't-SNE':
        reducer = TSNE(n_components=dimensions, perplexity=perplexity, n_iter=n_iter, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=dimensions)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create DataFrame for visualization
    if dimensions == 2:
        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    else:  # 3D
        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y', 'z'])
    
    # Add metadata if available
    if results.get('metadatas') and len(results['metadatas']) > 0:
        metadata_df = pd.DataFrame(results['metadatas'])
        df = pd.concat([df, metadata_df], axis=1)
    
    # Add document text if available
    if results.get('documents') and len(results['documents']) > 0:
        df['document'] = results['documents']
    
    return df

def main():
    st.title("Document Embeddings Visualization")
    st.write("Explore the semantic relationships between your documents using dimensionality reduction techniques.")
    
    # Sidebar controls
    st.sidebar.header("Visualization Settings")
    
    # Add dimension selector
    dimensions = st.sidebar.radio(
        "Visualization Dimensions",
        [2, 3],
        format_func=lambda x: f"{x}D"
    )
    
    method = st.sidebar.selectbox(
        "Dimensionality Reduction Method",
        ['t-SNE', 'PCA']
    )
    
    if method == 't-SNE':
        perplexity = st.sidebar.slider(
            "Perplexity",
            min_value=5,
            max_value=50,
            value=30,
            help="Higher values create more global structure, lower values focus on local structure"
        )
        n_iter = st.sidebar.slider(
            "Number of iterations",
            min_value=250,
            max_value=2000,
            value=1000,
            step=250
        )
    else:
        perplexity = 30
        n_iter = 1000
    
    # Load data
    with st.spinner("Loading embeddings..."):
        results = load_embeddings()
    
    if results is None:
        return
    
    # Create visualization
    df = create_visualization(results, method, perplexity, n_iter, dimensions)
    
    # Visualization options
    st.sidebar.header("Plot Options")
    
    # Get available columns for coloring, excluding coordinate columns and document
    coordinate_columns = ['x', 'y', 'z'] if dimensions == 3 else ['x', 'y']
    available_columns = [col for col in df.columns if col not in coordinate_columns + ['document']]
    color_by = st.sidebar.selectbox(
        "Color by",
        ['None'] + available_columns
    )
    
    size = st.sidebar.slider(
        "Point size",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Create plot
    if dimensions == 2:
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color=color_by if color_by != 'None' else None,
            hover_data=[col for col in df.columns if col not in coordinate_columns],
            title=f"{method} Visualization of Document Embeddings (2D)",
            width=800,
            height=600
        )
        fig.update_layout(
            xaxis_title="Semantic Content (Primary Pattern)",
            yaxis_title="Semantic Context (Secondary Pattern)"
        )
    else:  # 3D
        fig = px.scatter_3d(
            df,
            x='x',
            y='y',
            z='z',
            color=color_by if color_by != 'None' else None,
            hover_data=[col for col in df.columns if col not in coordinate_columns],
            title=f"{method} Visualization of Document Embeddings (3D)",
            width=800,
            height=600
        )
        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method} Dimension 1",
                yaxis_title=f"{method} Dimension 2",
                zaxis_title=f"{method} Dimension 3"
            )
        )
    
    fig.update_traces(marker=dict(size=size))
    st.plotly_chart(fig, use_container_width=True)
    
    # Show data table
    st.subheader("Document Data")
    st.dataframe(df)

if __name__ == "__main__":
    main() 