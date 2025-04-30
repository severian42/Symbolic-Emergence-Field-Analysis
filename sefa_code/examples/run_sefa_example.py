"""Example script demonstrating the use of the SEFA library."""

import sys
import os
import json
import time
import logging
import math
import concurrent.futures
from datetime import datetime
from typing import Optional, Dict, Any

# --- Add SEFA_Python root to sys.path --- 
# This ensures the local sefa package can be found
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
from numba import njit # Numba is required for performance
import plotly.graph_objects as go
import plotly.io as pio
from pyvis.network import Network

# --- Local Application Imports ---
from sefa import SEFA, SEFAConfig, SavgolParams # Direct import

# --- Configure logging --- 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info("Required libraries (numpy, matplotlib, networkx, numba, plotly, pyvis) imported.")
logger.info("SEFA library imported from local path.")

# Rydberg constant in eV
RYDBERG_CONSTANT_EV = 13.6

def generate_hydrogen_levels(n_max: int) -> np.ndarray:
    """Generates Hydrogen atom energy levels based on the Rydberg formula."""
    if n_max < 1:
        raise ValueError("n_max must be at least 1.")
    n_values = np.arange(1, n_max + 1)
    # E_n = -R_H / n^2
    energy_levels = -RYDBERG_CONSTANT_EV / (n_values**2)
    # The levels are naturally sorted from most negative (n=1) to closer to 0
    return energy_levels 

def create_output_dir(base_dir="sefa_outputs"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_array(arr, path):
    np.save(path, arr)

def save_plot(fig, path):
    fig.savefig(path, bbox_inches='tight')

def save_config(config, path):

    with open(path, 'w') as f:
        json.dump(config.__dict__, f, indent=2, default=str)

def save_summary(metrics, path):
    with open(path, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

@njit
def _check_visibility_numba(i: int, n: int, y_values: np.ndarray, series_data: np.ndarray) -> list[tuple[int, int]]:
    """Numba-accelerated helper to find visible nodes from node i."""
    visible_edges = []
    y_i = y_values[i]
    series_i = series_data[i]
    for j in range(i + 1, n):
        y_j = y_values[j]
        series_j = series_data[j]
        
        # Check visibility
        is_visible = True
        # Avoid division by zero (should be unlikely with float y_values)
        if abs(y_j - y_i) < 1e-12: continue # Use tolerance for float comparison

        k = i + 1
        while k < j:
            y_k = y_values[k]
            series_k = series_data[k]
            # Calculate height of line-of-sight at y_k
            los_height = series_i + (series_j - series_i) * (y_k - y_i) / (y_j - y_i)
            # If any intermediate point is above or on the line-of-sight, not visible
            # Add small tolerance for floating point comparisons if needed: >= los_height - 1e-9
            if series_k >= los_height:
                is_visible = False
                break
            k += 1
        
        if is_visible:
            visible_edges.append((i, j))
            
    return visible_edges

# --- Top-level helper function for parallel execution --- 
def _calculate_visibility_for_node_task(args):
    """Worker function for ProcessPoolExecutor. Calls the Numba-jitted check.
    
    Args:
        args (tuple): A tuple containing (i, n, y_values, series_data)
        
    Returns:
        list[tuple[int, int]]: List of visible edges (i, j) from node i.
    """
    i, n, y_values, series_data = args
    # Ensure numpy arrays are passed if required by Numba function signature,
    # although Numba often handles conversion implicitly.
    if not isinstance(y_values, np.ndarray):
        y_values = np.asarray(y_values)
    if not isinstance(series_data, np.ndarray):
        series_data = np.asarray(series_data)
        
    return _check_visibility_numba(i, n, y_values, series_data)

def build_visibility_graph(y_values: np.ndarray, series_data: np.ndarray, feature_results: Dict[str, Any]) -> Optional[nx.Graph]:
    """Builds a Natural Visibility Graph from a 1D data series, accelerated with Numba and parallel processing.

    Args:
        y_values: The domain values (x-coordinates). Must be numpy array for Numba.
        series_data: The data series (e.g., SEFA score) used for visibility (y-coordinates). Must be numpy array for Numba.
        feature_results: Dictionary containing other calculated features to add as node attributes.

    Returns:
        A NetworkX graph, or None if networkx is not available.
    """

    n = len(y_values)
    if n != len(series_data):
        raise ValueError("y_values and series_data must have the same length.")
    if not isinstance(y_values, np.ndarray) or not isinstance(series_data, np.ndarray):
         logger.warning("y_values and series_data should be NumPy arrays for optimal performance with Numba. Converting...")
         y_values = np.asarray(y_values)
         series_data = np.asarray(series_data)


    logger.info(f"Building Visibility Graph for {n} points... (Using Numba and parallel processing)")
    start_time = time.time()
    
    G = nx.Graph()

    # --- Add nodes with attributes ---
    for i in range(n):
        # Ensure values are standard Python floats for JSON compatibility if saving graph attributes later
        node_attrs = {
            "y_value": float(y_values[i]) if not np.isnan(y_values[i]) else None,
            "sefa_score": float(series_data[i]) if not np.isnan(series_data[i]) else None
        }
        # Add other features if available
        for feat_key in ["amplitude_A", "curvature_C", "frequency_F", "entropy_S", "entropy_alignment_E"]:
             feat_arr = feature_results.get(feat_key)
             if feat_arr is not None and i < len(feat_arr) and not np.isnan(feat_arr[i]):
                 # Ensure attribute value is JSON serializable (float, int, str, bool, None)
                 val = feat_arr[i]
                 if isinstance(val, (np.generic, np.ndarray)): # Convert numpy types
                      val = val.item() 
                 if isinstance(val, (float, int, str, bool)) or val is None:
                      node_attrs[feat_key] = val
                 else:
                      # Handle unexpected types if necessary, e.g., convert to string or skip
                      logger.debug(f"Skipping non-serializable attribute '{feat_key}' for node {i}: type {type(val)}")

        # Filter out None values before adding node
        node_attrs = {k: v for k, v in node_attrs.items() if v is not None}
        G.add_node(i, **node_attrs)

    # --- Add edges based on visibility ---
    all_edges = []
    num_workers = os.cpu_count() 
    logger.info(f"Calculating visibility using {num_workers} worker processes...")

    # Use ProcessPoolExecutor for CPU-bound Numba tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Prepare arguments for each task
        tasks_args = [(i, n, y_values, series_data) for i in range(n)]
        
        # Map the top-level worker function to the arguments
        # Pass the necessary data (y_values, series_data) to each worker process
        future_to_node = {executor.submit(_calculate_visibility_for_node_task, args): args[0] for args in tasks_args}
        
        # Collect results as they complete
        completed_count = 0
        total_tasks = n
        for future in concurrent.futures.as_completed(future_to_node):
            node_i = future_to_node[future]
            try:
                visible_edges_from_i = future.result() # Get the list of edges like [(i, j), ...]
                all_edges.extend(visible_edges_from_i)
                completed_count += 1
                # Optional: Log progress periodically
                if completed_count % max(1, total_tasks // 10) == 0 or completed_count == total_tasks:
                    logger.info(f"Visibility calculation progress: {completed_count}/{total_tasks} nodes completed.")
            except Exception as exc:
                logger.error(f'Node {node_i} generated an exception during visibility check: {exc}', exc_info=True) # Log traceback

    # Add all collected edges to the graph at once
    if all_edges:
        G.add_edges_from(all_edges)
    else:
        logger.warning("No edges were calculated or added to the graph. Check for errors during visibility calculation.")
                
    end_time = time.time()
    logger.info(f"Visibility Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges. (Took {end_time - start_time:.2f} seconds)")
    return G

def plot_graph_3d(G: nx.Graph, out_path: str):
    """Attempts a basic 3D plot of the graph using Matplotlib.

    Nodes positioned by (y_value, sefa_score, 0), colored by degree.
    Requires Matplotlib 3D toolkit.
    """
        
    if G is None:
        print("Graph object is None, skipping plot.")
        return

    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("Matplotlib 3D toolkit not found. Cannot create 3D graph plot. Install with: pip install matplotlib")
        return
        
    logger.info("Generating 3D graph plot (basic)... ")

    # Node positions: use y_value and sefa_score for X, Y, set Z=0 initially
    pos = {}
    scores = []
    valid_nodes = []
    for node, data in G.nodes(data=True):
        if data.get('y_value') is not None and data.get('sefa_score') is not None:
             pos[node] = [data['y_value'], data['sefa_score'], 0]
             scores.append(data['sefa_score'])
             valid_nodes.append(node)
        # else: handle nodes missing data? Skip for now.
        
    if not pos:
        logger.warning("No valid node positions found for 3D plot.")
        return

    # Convert pos dict to array format needed by scatter
    node_xyz = np.array([pos[v] for v in valid_nodes])

    # Calculate node degrees for coloring/sizing
    degrees = [G.degree(v) for v in valid_nodes]
    min_degree, max_degree = min(degrees) if degrees else (0,1), max(degrees) if degrees else (1,1)
    degree_range = max(1, max_degree - min_degree) # Avoid division by zero
    # Normalize degrees for color mapping
    node_colors = [(d - min_degree) / degree_range for d in degrees]
    node_sizes = [(d - min_degree) / degree_range * 50 + 10 for d in degrees] # Scale size

    # Create 3D plot
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot edges
    for u, v in G.edges():
        if u in pos and v in pos:
            ax.plot(
                [pos[u][0], pos[v][0]], # x coords (y_value)
                [pos[u][1], pos[v][1]], # y coords (sefa_score)
                [pos[u][2], pos[v][2]], # z coords (0)
                color="gray",
                alpha=0.3,
                linewidth=0.5
            )

    # Plot nodes
    scatter = ax.scatter(
        node_xyz[:, 0],
        node_xyz[:, 1],
        node_xyz[:, 2],
        s=node_sizes,
        c=node_colors,
        cmap="viridis",
        edgecolors='k', 
        linewidths=0.5
    )

    ax.set_title("SEFA Visibility Graph (3D Basic Projection)")
    ax.set_xlabel("Domain (y)")
    ax.set_ylabel("SEFA Score")
    ax.set_zlabel("Z (Unused)")
    # Add colorbar
    cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    cbar.set_label('Normalized Node Degree')
    
    plt.savefig(out_path, bbox_inches='tight')
    logger.info(f"3D graph plot saved to: {out_path}")
    plt.close(fig)

def plot_graph_plotly(G: nx.Graph, out_path: str):
    """Generates an interactive 2D graph plot using Plotly.

    Nodes positioned by (y_value, sefa_score), colored and sized by degree.
    Saves the plot as an HTML file.
    """
    
    if G is None:
        logger.warning("Graph object is None, skipping interactive Plotly graph generation.")
        return

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.info(f"Generating interactive Plotly graph ({num_nodes} nodes, {num_edges} edges)...")

    # --- Prepare Data --- 
    pos = {node: (data.get('y_value', 0), data.get('sefa_score', 0)) 
           for node, data in G.nodes(data=True) 
           if data.get('y_value') is not None and data.get('sefa_score') is not None}
    
    if not pos:
        logger.warning("No valid node positions (y_value, sefa_score) found for Plotly plot.")
        return
        
    valid_nodes = list(pos.keys())
    degrees = dict(G.degree(valid_nodes))
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    degree_range = max(1, max_degree - min_degree)

    # --- Edges --- 
    edge_x = []
    edge_y = []
    for edge in G.edges():
        # Only include edges where both nodes have positions
        if edge[0] in pos and edge[1] in pos:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='rgba(136, 136, 136, 0.1)'), # Use RGBA for transparency
        hoverinfo='none',
        mode='lines')

    # --- Nodes --- 
    node_x = [pos[node][0] for node in valid_nodes]
    node_y = [pos[node][1] for node in valid_nodes]
    node_degrees = [degrees.get(node, 0) for node in valid_nodes]
    # Normalize degrees for color mapping and size
    norm_degrees = [(d - min_degree) / degree_range for d in node_degrees]
    node_sizes = [10 + nd * 30 for nd in norm_degrees] # Adjust size scaling as needed
    
    # Prepare hover text
    node_text = []
    for node in valid_nodes:
        attrs = G.nodes[node]
        text = f"Node: {node}<br>"
        text += f"y: {attrs.get('y_value', 'N/A'):.4f}<br>"
        text += f"SEFA: {attrs.get('sefa_score', 'N/A'):.4f}<br>"
        text += f"Degree: {degrees.get(node, 0)}<br>"
        # Add other attributes if desired
        for feat in ["amplitude_A", "curvature_C", "frequency_F", "entropy_S", "entropy_alignment_E"]:
            if feat in attrs:
                 text += f"{feat}: {attrs[feat]:.4f}<br>"
        node_text.append(text)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis', 
            reversescale=False,
            color=norm_degrees,
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title=dict(
                    text='Node Degree (Normalized)', 
                    side='right'
                ),
                xanchor='left'
            ),
            line_width=1,
            line_color='black')
        )

    # --- Create Figure --- 
    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=dict( 
                        text='Interactive SEFA Visibility Graph (Plotly)',
                        font=dict(size=16) 
                    ),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Node position (x, y) = (Domain y, SEFA Score)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(title='Domain (y)', showgrid=True, zeroline=False, showticklabels=True),
                    yaxis=dict(title='SEFA Score', showgrid=True, zeroline=False, showticklabels=True),
                    plot_bgcolor='white' 
                    )\
                 )
    
    # Save to HTML
    try:
        fig.write_html(out_path)
        logger.info(f"Interactive Plotly graph saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to save interactive Plotly graph: {e}")

def plot_graph_2d_static(G: nx.Graph, out_path: str):
    """Generates a static 2D graph plot using Matplotlib and NetworkX.

    Nodes positioned by (y_value, sefa_score), colored and sized by degree.
    Saves the plot as a PNG file.
    """
        
    if G is None:
        logger.warning("Graph object is None, skipping static 2D graph plot.")
        return

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.info(f"Generating static 2D graph plot ({num_nodes} nodes, {num_edges} edges)...")

    # --- Prepare Data --- 
    pos = {node: (data.get('y_value', 0), data.get('sefa_score', 0)) 
           for node, data in G.nodes(data=True) 
           if data.get('y_value') is not None and data.get('sefa_score') is not None}
    
    if not pos:
        logger.warning("No valid node positions (y_value, sefa_score) found for static 2D plot.")
        return
        
    valid_nodes = list(pos.keys()) # Nodes that have position data
    degrees = dict(G.degree(valid_nodes))
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    degree_range = max(1, max_degree - min_degree)

    # Normalize degrees for color mapping and size
    node_colors = [(degrees.get(node, 0) - min_degree) / degree_range for node in valid_nodes]
    node_sizes = [10 + ((degrees.get(node, 0) - min_degree) / degree_range * 50) for node in valid_nodes] # Same scaling as 3D

    # --- Create Plot --- 
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Use Viridis colormap
    cmap = plt.get_cmap('viridis')

    # Draw edges
    nx.draw_networkx_edges(
        G, pos, 
        nodelist=valid_nodes, 
        ax=ax, 
        edge_color="gray", 
        alpha=0.05, 
        width=0.5
    )

    # Draw nodes
    nodes = nx.draw_networkx_nodes(
        G, pos, 
        nodelist=valid_nodes, 
        ax=ax, 
        node_size=node_sizes, 
        node_color=node_colors,
        cmap=cmap,
        alpha=0.9, 
        linewidths=0.5, 
        edgecolors='black'
    )
    
    ax.set_title("SEFA Visibility Graph (2D Static Projection)")
    ax.set_xlabel("Domain (y)")
    ax.set_ylabel("SEFA Score")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_degree, vmax=max_degree))
    sm.set_array([]) # You need this for the colorbar to work with ScalarMappable
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, aspect=20)
    cbar.set_label('Node Degree')

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    logger.info(f"Static 2D graph plot saved to: {out_path}")
    plt.close(fig)

def save_interactive_graph_html(G: nx.Graph, out_path: str):
    """Saves the graph as an interactive HTML file using pyvis."""

    if G is None:
        logger.warning("Graph object is None, skipping interactive HTML generation.")
        return

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.info(f"Generating interactive graph HTML ({num_nodes} nodes, {num_edges} edges)...")

    # Add warning for very large graphs
    if num_nodes > 10000 or num_edges > 1000000:
        logger.warning(f"Graph size is large ({num_nodes} nodes, {num_edges} edges). Pyvis HTML generation may be VERY slow and the resulting file may be huge.")


    net = Network(notebook=False, height='800px', width='100%', heading='SEFA Visibility Graph')

    # Prepare node attributes for pyvis
    node_data = []
    degrees = dict(G.degree())
    max_degree = max(degrees.values()) if degrees else 1
    min_degree = min(degrees.values()) if degrees else 0
    degree_range = max(1, max_degree - min_degree)

    for node, attrs in G.nodes(data=True):
        node_id = int(node) # Ensure node ID is int for pyvis
        title = f"Node {node_id}\n" \
                f"y: {attrs.get('y_value', 'N/A'):.4f}\n" \
                f"SEFA: {attrs.get('sefa_score', 'N/A'):.4f}\n" \
                f"Degree: {degrees.get(node, 0)}"
        
        # Scale size by degree
        node_size = 10 + (degrees.get(node, 0) - min_degree) / degree_range * 40 
        
        # Color by SEFA score (example - could use degree or other features)
        score = attrs.get('sefa_score')
        node_color = '#d3d3d3' # Default grey if no score
        if score is not None:
            # Normalize score for coloring (adjust range as needed)
            min_score = G.nodes[min(G.nodes, key=lambda n: G.nodes[n].get('sefa_score', np.inf))].get('sefa_score', 0)
            max_score = G.nodes[max(G.nodes, key=lambda n: G.nodes[n].get('sefa_score', -np.inf))].get('sefa_score', 1)
            score_range = max(1e-9, max_score - min_score) # Avoid division by zero
            norm_score = (score - min_score) / score_range
            # Use a colormap (e.g., coolwarm: blue=low, red=high)
            # --- Remove matplotlib colormap ---
            # try: 
            #     cmap = plt.get_cmap('coolwarm')
            #     rgba = cmap(norm_score)
            #     node_color = matplotlib.colors.rgb2hex(rgba)
            # except NameError:
            #      # Fallback if matplotlib isn't imported or available here
            # Simple blue-red gradient based on score
            red = int(255 * norm_score)
            blue = 255 - red
            node_color = f'#{red:02x}00{blue:02x}'
            # --- End Remove ---


        node_data.append({
            'id': node_id,
            'label': str(node_id),
            'title': title,
            'size': node_size,
            'color': node_color,
            'physics': True 
        })
        
    # Add nodes using add_nodes which is faster for many nodes
    node_ids = [d['id'] for d in node_data]
    labels = [d['label'] for d in node_data]
    titles = [d['title'] for d in node_data]
    sizes = [d['size'] for d in node_data]
    colors = [d['color'] for d in node_data]
    net.add_nodes(node_ids, label=labels, title=titles, size=sizes, color=colors)

    # Add edges
    net.add_edges([(u, v) for u, v in G.edges()])

    # Configure physics and interaction
    net.show_buttons(filter_=['physics', 'nodes', 'edges'])
    # Enable physics simulation for layout (can be slow for large graphs)
    # net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08, damping=0.4, overlap=0)
    # BarnesHut is often faster for larger graphs
    # --- Remove barnes_hut config as physics is off by default ---
    # net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=250, spring_strength=0.04, damping=0.09, overlap=0.1)
    # --- End Remove ---
    # Disable physics by default for faster initial loading of large graphs
    net.toggle_physics(False)

    try:
        net.save_graph(out_path)
        logger.info(f"Interactive graph saved to: {out_path}")
    except Exception as e:
        logger.error(f"Failed to save interactive graph: {e}")

def run_example():
    """Runs a SEFA example using Hydrogen atom energy levels as drivers."""
    logger.info("Starting SEFA Hydrogen energy level driver example...")

    # 1. Generate Hydrogen Energy Levels as Drivers
    max_principal_quantum_number = 50 # Max n value - REDUCED FOR DEMO SPEED
    logger.info(f"Generating Hydrogen energy levels up to n={max_principal_quantum_number}...")
    drivers = generate_hydrogen_levels(max_principal_quantum_number)
    logger.info(f"Generated {len(drivers)} drivers (energy levels). Range: [{drivers.min():.4f} eV, {drivers.max():.4f} eV]")

    # Define domain based on energy level range, add some padding
    if len(drivers) > 1:
        padding = (drivers.max() - drivers.min()) * 0.1 # Add 10% padding
    else:
        padding = abs(drivers[0]) * 0.1 if len(drivers) == 1 else 1.0 # Handle single driver or default
        
    y_min = drivers.min() - padding
    y_max = drivers.max() + padding # Max energy is close to 0
    M = 2500 # Number of points - REDUCED FOR DEMO SPEED
    logger.info(f"Domain set to: [{y_min:.4f}, {y_max:.4f}] with {M} points")
    logger.info(f"Using {len(drivers)} Hydrogen energy levels as drivers (gamma_k).")

    # 2. Configure SEFA
    # Keep configuration similar, window size depends on M
    # Use a slightly larger relative window for entropy due to potentially denser features
    window_size = max(51, int(0.01 * M)) # Use 1% of M, min 51 (ensure odd)
    if window_size % 2 == 0: window_size += 1 # Ensure odd

    config = SEFAConfig(
        entropy_window_size=window_size,
        derivative_method='savgol',
        savgol_frequency_params=SavgolParams(window_length=window_size, polyorder=3, deriv_order=1),
        savgol_curvature_params=SavgolParams(window_length=window_size, polyorder=4, deriv_order=2),
        boundary_method='discard',
        boundary_discard_fraction=0.05
    )
    logger.info(f"SEFA Configuration: {config}")

    # 3. Initialize and run SEFA analysis
    sefa_analyzer = SEFA(config=config)
    try:
        logger.info("Running SEFA pipeline...")
        sefa_analyzer.run_pipeline(
            drivers_gamma=drivers,
            ymin=y_min,
            ymax=y_max,
            num_points=M
        )
        logger.info("SEFA pipeline finished.")
    except Exception as e:
        logger.error(f"Error during SEFA pipeline execution: {e}", exc_info=True)
        return

    # 4. Get results
    results = sefa_analyzer.get_results()
    sefa_score = results['sefa_score']
    processed_domain_y = results['processed_domain_y']
    # Fetch the indices corresponding to the processed data points
    processed_indices = results['processed_indices']
    # Get the original full domain and field for separate plotting
    field_V0_full = results['field_V0']
    domain_y_full = results['domain_y']

    if sefa_score is None or processed_domain_y is None:
        logger.error("SEFA score or processed domain could not be retrieved.")
        return

    # Add assertion to catch shape mismatch early
    assert len(processed_domain_y) == len(sefa_score), \
        f"Shape mismatch after SEFA: processed_domain_y ({len(processed_domain_y)}) vs sefa_score ({len(sefa_score)})"

    # --- Analysis & Metrics --- 
    print("\n--- SEFA Analysis Results ---")

    # Driver Weights (w_k = 1 / (1 + |gamma_k|^beta))
    weights = results['weights_w']
    print(f"Driver Weights (beta={config.beta:.2f}): {np.round(weights, 4)}")
    print("  - Higher weights for lower frequency drivers.")

    # Information Deficits (w_X = Max(0, Log(B) - I_X))
    deficits = results['information_deficits_w']
    print(f"Information Deficits (w_X):")
    for name, w in deficits.items():
        print(f"  - w_{name}: {w:.4f}")
    print("  - Higher deficit means the feature (globally) has lower entropy / more structure.")

    # Exponents (alpha_X = p * w_X / W_Total)
    exponents = results['exponents_alpha']
    W_Total = sum(deficits.values())
    print(f"Exponents (alpha_X) (p={config.p_features}, W_Total={W_Total:.4f}):")
    for name, alpha in exponents.items():
        print(f"  - alpha_{name}: {alpha:.4f}")
    print(f"  - Sum of alpha: {sum(exponents.values()):.4f} (should approx equal p={config.p_features})")
    print("  - Features with higher deficit contribute more strongly to the final score.")

    # SEFA Score Statistics
    print(f"SEFA Score Statistics:")
    print(f"  - Min: {np.min(sefa_score):.4g}")
    print(f"  - Max: {np.max(sefa_score):.4g}")
    print(f"  - Mean: {np.mean(sefa_score):.4g}")
    print(f"  - Std Dev: {np.std(sefa_score):.4g}")

    # 5. Thresholding and Symbol Detection
    # thresholding_method = 'otsu'
    thresholding_method = 'percentile' 
    threshold_percentile = 95
    threshold = np.nan
    mask = None
    symbol_locs = np.array([])
    symbol_indices_processed = np.array([], dtype=int)

    # --- Applying thresholding ---
    try:
        logger.info(f"Applying thresholding method: '{thresholding_method}' (percentile={threshold_percentile})...")
        # Pass the percentile argument
        mask = sefa_analyzer.threshold_score(method=thresholding_method, percentile=threshold_percentile)
        if mask is not None:
            # Estimate threshold value for plotting
            valid_scores = sefa_score[~np.isnan(sefa_score)]
            if valid_scores.size > 0:
                try:
                    # Recalculate percentile threshold just for plotting label
                    threshold = np.percentile(valid_scores, threshold_percentile)
                except Exception as e:
                    logger.warning(f"Error calculating percentile threshold for plot label: {e}")

            symbol_indices_processed = np.where(mask)[0]
            if symbol_indices_processed.size > 0:
                symbol_locs = processed_domain_y[symbol_indices_processed]
                logger.info(f"Thresholding applied. Threshold ({thresholding_method} {threshold_percentile}%): {threshold:.4g}")
                logger.info(f"Detected {len(symbol_locs)} potential symbols at y = {np.round(symbol_locs, 3)}")
            else:
                 logger.info(f"Thresholding applied. No symbols detected above {thresholding_method} {threshold_percentile}% threshold ({threshold:.4g}).")
        else:
            logger.warning(f"Thresholding method '{thresholding_method}' failed or returned None.")

    except Exception as e:
        logger.error(f"Error during thresholding/symbol detection: {e}", exc_info=True)
    # --- End of thresholding section ---

    # 6. Plotting Results
    logger.info("Generating plots...")
    plt.style.use('seaborn-v0_8-darkgrid')
    # Increase figure height for 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

    # --- Plot 1: Drivers and Weights ---
    ax0 = axes[0]
    line_drivers = ax0.stem(drivers, weights, linefmt='C0-', markerfmt='C0o', basefmt=' ', label='Driver Weights')
    ax0.set_ylabel('SEFA Weight ($w_k$)')
    ax0.set_title('Hydrogen Energy Level Drivers & SEFA Weights', fontsize=14)
    # Add text labels for the first few drivers (n=1, 2, 3)
    n_levels_to_label = 3
    for i in range(min(n_levels_to_label, len(drivers))):
        ax0.text(drivers[i], weights[i] + 0.02, f'n={i+1}\n{drivers[i]:.2f} eV', 
                 ha='center', va='bottom', fontsize=9)
    ax0.grid(True, linestyle='--', alpha=0.5)
    ax0.legend(loc='upper right')

    # --- Plot 2: V0 and SEFA Score --- 
    ax1 = axes[1] # Now axes[1]
    ax1_twin = ax1.twinx()

    # Plot V0 on original full domain/indices
    ax1.plot(domain_y_full, field_V0_full, label='$V_0(y)$ (Original Field)', color='blue', alpha=0.7, linewidth=1.5)
    ax1.set_ylabel('$V_0$ Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot SEFA score on processed domain 
    line_sefa, = ax1_twin.plot(processed_domain_y, sefa_score, label='SEFA(y) Score', color='red', linewidth=2)
    ax1_twin.set_ylabel('SEFA Score', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.spines['right'].set_color('red')

    # --- Add Vertical Lines for Drivers (NEW) ---
    driver_lines = []
    for i, driver_val in enumerate(drivers):
        line = ax1.axvline(driver_val, color='gray', linestyle='--', linewidth=0.8, alpha=0.6, zorder=1)
        if i == 0:
             driver_lines.append(line) # Only add one line to legend
    if driver_lines:
        driver_lines[0].set_label('Driver Locations') # Set label for the first line
    # --- End Driver Lines --- 

    # Plot threshold and detected symbols on SEFA axis
    line_thresh = None
    if not np.isnan(threshold):
        threshold_label = f'{thresholding_method.capitalize()} {threshold_percentile}% Threshold ({threshold:.3f})'
        if thresholding_method == 'otsu': threshold_label = f'Otsu Threshold ({threshold:.3f})'
        line_thresh = ax1_twin.axhline(threshold, color='black', linestyle=':', linewidth=1.5, label=threshold_label)
    
    scatter_sym = None
    if symbol_indices_processed.size > 0:
        symbol_y_coords = sefa_score[symbol_indices_processed]
        scatter_sym = ax1_twin.scatter(symbol_locs, symbol_y_coords, color='black',
                       marker='o', s=80, zorder=10, edgecolors='white',
                       label=f'Detected Symbols ({len(symbol_locs)})')

    ax1.set_title('SEFA Field Construction and Final Score', fontsize=14)
    handles = [ax1.get_legend_handles_labels()[0][0], line_sefa] # V0, SEFA
    labels = [ax1.get_legend_handles_labels()[1][0], line_sefa.get_label()]
    if driver_lines: handles.append(driver_lines[0]); labels.append(driver_lines[0].get_label()) # Add driver line legend item
    if line_thresh: handles.append(line_thresh); labels.append(line_thresh.get_label())
    if scatter_sym: handles.append(scatter_sym); labels.append(scatter_sym.get_label())
    ax1_twin.legend(handles, labels, loc='upper right')

    # --- Plot 3: Geometric Features (A, C, F) --- 
    ax2 = axes[2] # Now axes[2]
    ax2_twin = ax2.twinx()

    # Plot other features also using processed_domain_y
    amplitude_A = results.get('amplitude_A')
    curvature_C = results.get('curvature_C')
    frequency_F = results.get('frequency_F')

    if amplitude_A is not None:
        assert len(processed_domain_y) == len(amplitude_A), f"Shape mismatch: processed_domain_y ({len(processed_domain_y)}) vs amplitude_A ({len(amplitude_A)})"
        line_a, = ax2.plot(processed_domain_y, amplitude_A, label='Amplitude A(y)', color='green', alpha=0.8)
    else: line_a = None

    if curvature_C is not None:
        assert len(processed_domain_y) == len(curvature_C), f"Shape mismatch: processed_domain_y ({len(processed_domain_y)}) vs curvature_C ({len(curvature_C)})"
        line_c, = ax2.plot(processed_domain_y, curvature_C, label='Curvature C(y)', color='purple', alpha=0.8)
    else: line_c = None

    ax2.set_ylabel('Amplitude / Curvature', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5)

    if frequency_F is not None:
        assert len(processed_domain_y) == len(frequency_F), f"Shape mismatch: processed_domain_y ({len(processed_domain_y)}) vs frequency_F ({len(frequency_F)})"
        line_f, = ax2_twin.plot(processed_domain_y, frequency_F, label='Frequency F(y)', color='orange', linestyle='--')
    else: line_f = None

    ax2_twin.set_ylabel('Frequency', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.spines['right'].set_color('orange')

    ax2.set_title('Geometric Features', fontsize=14)
    geom_handles = [h for h in [line_a, line_c, line_f] if h is not None]
    geom_labels = [h.get_label() for h in geom_handles]
    if geom_handles: ax2_twin.legend(geom_handles, geom_labels, loc='upper right')

    # --- Plot 4: Entropy Features (S, E) --- 
    ax3 = axes[3] # Now axes[3]
    ax3_twin = ax3.twinx()

    entropy_S = results.get('entropy_S')
    entropy_alignment_E = results.get('entropy_alignment_E')

    if entropy_S is not None:
        assert len(processed_domain_y) == len(entropy_S), f"Shape mismatch: processed_domain_y ({len(processed_domain_y)}) vs entropy_S ({len(entropy_S)})"
        line_s, = ax3.plot(processed_domain_y, entropy_S, label='Entropy S(y)', color='cyan')
    else: line_s = None

    ax3.set_ylabel('Local Entropy S', color='cyan')
    ax3.tick_params(axis='y', labelcolor='cyan')
    ax3.spines['left'].set_color('cyan')
    ax3.grid(True, linestyle='--', alpha=0.5)

    if entropy_alignment_E is not None:
        assert len(processed_domain_y) == len(entropy_alignment_E), f"Shape mismatch: processed_domain_y ({len(processed_domain_y)}) vs entropy_alignment_E ({len(entropy_alignment_E)})"
        line_e, = ax3_twin.plot(processed_domain_y, entropy_alignment_E, label='Entropy Alignment E(y)', color='magenta', linestyle='--')
    else: line_e = None

    ax3_twin.set_ylabel('Entropy Alignment E', color='magenta')
    ax3_twin.tick_params(axis='y', labelcolor='magenta')
    ax3_twin.spines['right'].set_color('magenta')

    ax3.set_title('Entropy Features', fontsize=14)
    ax3.set_xlabel('Domain (y) - Energy (eV)', fontsize=12) # Updated x-label for clarity
    ent_handles = [h for h in [line_s, line_e] if h is not None]
    ent_labels = [h.get_label() for h in ent_handles]
    if ent_handles: ax3_twin.legend(ent_handles, ent_labels, loc='upper right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout slightly for suptitle and 4 panels
    fig.suptitle(f'Detailed SEFA Analysis with Hydrogen Atom Drivers (n=1..{max_principal_quantum_number})', fontsize=18, y=0.995) # Adjust y for suptitle
    logger.info("Showing plots...")
    plt.show()

    out_dir = create_output_dir()
    logger.info(f"Saving all results to: {out_dir}")

    # Save config
    save_config(config, os.path.join(out_dir, "sefa_config.json"))

    # Save all key arrays
    save_array(sefa_score, os.path.join(out_dir, "sefa_score.npy"))
    save_array(processed_domain_y, os.path.join(out_dir, "processed_domain_y.npy"))
    save_array(field_V0_full, os.path.join(out_dir, "field_V0_full.npy"))
    save_array(domain_y_full, os.path.join(out_dir, "domain_y_full.npy"))
    for key in ["amplitude_A", "curvature_C", "frequency_F", "entropy_S", "entropy_alignment_E"]:
        arr = results.get(key)
        if arr is not None:
            save_array(arr, os.path.join(out_dir, f"{key}.npy"))
    if mask is not None:
        save_array(mask, os.path.join(out_dir, "threshold_mask.npy"))
    if symbol_locs is not None and len(symbol_locs) > 0:
        save_array(symbol_locs, os.path.join(out_dir, "detected_symbol_locs.npy"))

    # Save main plots
    save_plot(fig, os.path.join(out_dir, "sefa_analysis.png"))

    # Save histograms for all features and SEFA score
    for key, arr in [("sefa_score", sefa_score), ("amplitude_A", results.get("amplitude_A")), ("curvature_C", results.get("curvature_C")), ("frequency_F", results.get("frequency_F")), ("entropy_S", results.get("entropy_S")), ("entropy_alignment_E", results.get("entropy_alignment_E"))]:
        if arr is not None:
            plt.figure()
            plt.hist(arr[~np.isnan(arr)], bins=100, alpha=0.7)
            plt.title(f"Histogram: {key}")
            plt.xlabel(key)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"hist_{key}.png"))
            plt.close()

    # --- NetworkX Graph Generation and Visualization --- 
    # Build graph using SEFA score
    visibility_graph = build_visibility_graph(processed_domain_y, sefa_score, results)
        
    if visibility_graph is not None:
        # Save graph
        graph_path = os.path.join(out_dir, "sefa_visibility_graph.graphml")
        try:
            nx.write_graphml(visibility_graph, graph_path)
            logger.info(f"Visibility graph saved to: {graph_path}")
        except Exception as e:
            logger.error(f"Failed to save graph: {e}")

        # Plot graph (basic 3D)
        # plot_path = os.path.join(out_dir, "sefa_visibility_graph_3d.png")
        # plot_graph_3d(visibility_graph, plot_path)
        
        # --- New: Plot Static 2D Graph ---
        static_2d_plot_path = os.path.join(out_dir, "sefa_visibility_graph_2d_static.png")
        plot_graph_2d_static(visibility_graph, static_2d_plot_path)
        
        # --- Plot Interactive Plotly Graph ---
        plotly_plot_path = os.path.join(out_dir, "sefa_visibility_graph_plotly.html")
        plot_graph_plotly(visibility_graph, plotly_plot_path)

        # Generate interactive HTML graph if pyvis is available
        # --- Pyvis plot generation ---
        html_path = os.path.join(out_dir, "sefa_visibility_graph_interactive.html")
        save_interactive_graph_html(visibility_graph, html_path)
        # --- End ---
    else:
         logger.warning("Visibility graph could not be built.")

    # Compute and save detailed metrics
    def feature_stats(arr):
        arr = arr[~np.isnan(arr)]
        return {
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "25%": float(np.percentile(arr, 25)),
            "50%": float(np.percentile(arr, 50)),
            "75%": float(np.percentile(arr, 75)),
        }
    metrics = {}
    metrics["config"] = str(config)
    metrics["driver_count"] = len(drivers)
    metrics["driver_type"] = "Hydrogen Energy Levels" # Add driver type
    metrics["max_quantum_number"] = max_principal_quantum_number
    metrics["rydberg_constant_eV"] = RYDBERG_CONSTANT_EV
    metrics["domain"] = f"[{y_min}, {y_max}]"
    metrics["num_points"] = M
    metrics["window_size"] = window_size
    metrics["weights"] = np.round(weights, 4).tolist()
    metrics["information_deficits"] = {k: float(v) for k, v in deficits.items()}
    metrics["exponents"] = {k: float(v) for k, v in exponents.items()}
    metrics["sefa_score_stats"] = feature_stats(sefa_score)
    for key in ["amplitude_A", "curvature_C", "frequency_F", "entropy_S", "entropy_alignment_E"]:
        arr = results.get(key)
        if arr is not None:
            metrics[f"{key}_stats"] = feature_stats(arr)
    if mask is not None:
        metrics["num_points_above_threshold"] = int(np.sum(mask))
    if symbol_locs is not None and len(symbol_locs) > 0:
        metrics["detected_symbol_locs"] = symbol_locs.tolist()
    save_summary(metrics, os.path.join(out_dir, "summary.txt"))
    logger.info(f"All results and metrics saved to: {out_dir}")

if __name__ == "__main__":
    run_example()
