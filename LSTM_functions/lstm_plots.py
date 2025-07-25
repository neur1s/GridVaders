import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from scipy.stats import pearsonr, stats
from scipy.spatial import distance
import pandas as pd
# Try both import methods
try:
    # For when running from notebook in parent directory
    from LSTM_functions.lstm_explorer import SEED
except ImportError:
    # For when running standalone
    from lstm_explorer import SEED

def plot_tsne(hidden_states, current_positions, nback_positions, n_back, step_hidden=40):
    """Plot two t-SNE plots side by side comparing current vs n-back positions"""
    assert step_hidden != 0, "Step zero has trivial representation."
    
    def to_grid_ids(positions):
        if isinstance(positions[0], (tuple, list, np.ndarray)) and len(positions[0]) == 2:
            return np.array([x * 5 + y for x, y in positions])
        return np.array(positions)
    
    current_ids = to_grid_ids(current_positions)
    nback_ids = to_grid_ids(nback_positions)
    
    np.random.seed(SEED)
    transformed = TSNE(n_components=2, perplexity=30, random_state=SEED).fit_transform(hidden_states)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=current_ids.astype(str), ax=ax1)
    ax1.set_title(f"Hidden States at step {step_hidden}\nColored by CURRENT position (step {step_hidden})")
    ax1.set_xlabel("t-SNE Dim 1")
    ax1.set_ylabel("t-SNE Dim 2")
    
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=nback_ids.astype(str), ax=ax2)
    ax2.set_title(f"Hidden States at step {step_hidden}\nColored by N-BACK position (step {step_hidden - n_back})")
    ax2.set_xlabel("t-SNE Dim 1")
    ax2.set_ylabel("")
    
    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: int(x[0])))
        ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Grid Position")
    
    plt.tight_layout()
    plt.show()

def plot_spatial_correlation(hidden_states, grid_positions, hidden_step, N=0):
    """Plot spatial correlation with improved styling and formatting"""
    # Convert positions to coordinates
    positions = np.array([(p//5, p%5) for p in grid_positions])
    
    # Calculate pairwise distances more efficiently
    grid_dists = distance.pdist(positions.astype(float))
    hidden_dists = distance.pdist(hidden_states)
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(grid_dists, hidden_dists)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Grid Distance': grid_dists,
        'Hidden Distance': hidden_dists,
        'Distance Bin': np.floor(grid_dists).astype(int)
    })
    
    # Set up plot style
    plt.figure(figsize=(6, 5))
    sns.set_style("white")
    
    # Create regression plot with custom styling
    ax = sns.regplot(
        data=df,
        x='Grid Distance',
        y='Hidden Distance',
        ci=None,
        scatter_kws={
            'alpha': 0.1,
            'color': '#1f77b4',  # Matplotlib default blue
            's': 15             # Smaller point size
        },
        line_kws={
            'color': '#d62728',  # Matplotlib default red
            'lw': 2,
            'alpha': 0.8
        }
    )
    
    # Custom grid and spines
    ax.grid(True, which='both', linestyle='--', alpha=0.2)
    sns.despine()
    
    # Determine position step for title
    pos_step = hidden_step if N == 0 else hidden_step - N
    
    # Format title with exact requested format
    title = (f"{N}-back task | Hidden step {hidden_step} vs Position step {pos_step}\n"
             f"Spatial Correlation (r = {corr:.3f}, p = {p_value:.3g})")
    plt.title(title, pad=12)
    
    # Axis labels
    plt.xlabel("Actual Grid Distance (units)", labelpad=10)
    plt.ylabel("Hidden State Distance", labelpad=10)
    
    # Adjust layout and show
    plt.tight_layout()
    plt.show()

def calculate_spatial_correlation(hidden_states, grid_positions):
    if isinstance(grid_positions[0], int):
        positions = np.array([(p//5, p%5) for p in grid_positions])
    else:
        positions = np.array(grid_positions)
    
    grid_dists = np.zeros((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(len(positions)):
            grid_dists[i,j] = np.linalg.norm(positions[i] - positions[j])
    
    hidden_dists = np.zeros((len(hidden_states), len(hidden_states)))
    for i in range(len(hidden_states)):
        for j in range(len(hidden_states)):
            hidden_dists[i,j] = np.linalg.norm(hidden_states[i] - hidden_states[j])
    
    grid_flat = grid_dists[np.triu_indices_from(grid_dists, k=1)]
    hidden_flat = hidden_dists[np.triu_indices_from(hidden_dists, k=1)]
    corr, p_value = pearsonr(grid_flat, hidden_flat)
    
    return corr, p_value, grid_dists, hidden_dists

def plot_accuracies(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accuracies)), accuracies, marker='o', linewidth=2, markersize=8)
    plt.xlabel("N-back", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.xticks(range(len(accuracies)))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.show()
