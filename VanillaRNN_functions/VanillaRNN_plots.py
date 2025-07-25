import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy import stats, spatial
import pandas as pd

def plot_tsne(hidden_states, current_positions, nback_positions, n_back, step_hidden=40):
    """Plot two t-SNE plots side by side comparing current vs n-back positions"""
    if len(hidden_states) == 0 or len(current_positions) == 0:
        print("Warning: No valid data points to plot")
        return
    
    def to_grid_ids(positions):
        positions = np.array(positions)
        if positions.ndim == 2 and positions.shape[1] == 2:  # If positions are (x,y) pairs
            return np.array([x * 5 + y for x, y in positions])
        return positions  # Assume already grid IDs
    
    current_ids = to_grid_ids(current_positions)
    nback_ids = to_grid_ids(nback_positions)
    
    np.random.seed(42)
    transformed = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(hidden_states)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot current positions
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=current_ids.astype(str), ax=ax1)
    ax1.set_title(f"Hidden States at step {step_hidden}\nColored by CURRENT position (step {step_hidden})")
    
    # Plot n-back positions
    sns.scatterplot(x=transformed[:, 0], y=transformed[:, 1], hue=nback_ids.astype(str), ax=ax2)
    ax2.set_title(f"Hidden States at step {step_hidden}\nColored by N-BACK position (n = {n_back}, step {step_hidden - n_back})")
    
    # Adjust legends
    for ax in [ax1, ax2]:
        handles, labels = ax.get_legend_handles_labels()
        if labels:  # Only proceed if we have labels
            labels, handles = zip(*sorted(zip(labels, handles), key=lambda x: int(x[0])))
            ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Grid Position")
    
    plt.tight_layout()
    plt.show()

def calculate_spatial_correlation(hidden_states, grid_positions):
    # Convert positions to (x,y) coordinates if they aren't already
    if isinstance(grid_positions[0], int):
        # Assuming 5x5 grid with flat IDs
        positions = np.array([(p//5, p%5) for p in grid_positions])
    else:
        positions = np.array(grid_positions)
    
    # Calculate pairwise grid distances
    grid_dists = np.zeros((len(positions), len(positions)))
    for i in range(len(positions)):
        for j in range(len(positions)):
            grid_dists[i,j] = np.linalg.norm(positions[i] - positions[j])
    
    # Calculate pairwise hidden state distances
    hidden_dists = np.zeros((len(hidden_states), len(hidden_states)))
    for i in range(len(hidden_states)):
        for j in range(len(hidden_states)):
            hidden_dists[i,j] = np.linalg.norm(hidden_states[i] - hidden_states[j])
    
    # Flatten the matrices and compute correlation
    grid_flat = grid_dists[np.triu_indices_from(grid_dists, k=1)]
    hidden_flat = hidden_dists[np.triu_indices_from(hidden_dists, k=1)]
    corr, p_value = stats.pearsonr(grid_flat, hidden_flat)
    
    return corr, p_value, grid_dists, hidden_dists

def plot_spatial_correlation(hidden_states, grid_positions, hidden_step, N=0):
    # Convert positions to coordinates
    positions = np.array([(p//5, p%5) for p in grid_positions])
    
    # Calculate pairwise distances
    grid_dists = spatial.distance.pdist(positions.astype(float))
    hidden_dists = spatial.distance.pdist(hidden_states)
    
    # Calculate correlation
    corr, p_value = stats.pearsonr(grid_dists, hidden_dists)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Grid Distance': grid_dists,
        'Hidden Distance': hidden_dists,
        'Distance Bin': np.floor(grid_dists).astype(int)  # Bin by integer part
    })
    
    # Plot 
    plt.figure(figsize=(6,5))
    sns.set_style("white")
    ax = sns.regplot(
        data=df,
        x='Grid Distance',
        y='Hidden Distance',
        ci=None,
        marker='.',
        scatter_kws={'alpha':0.05},  
        line_kws={'color':'r'}        
    )
    ax.grid(False)
    
    # Determine position step for title
    pos_step = hidden_step if N == 0 else hidden_step - N 
    
    # Formatting (exact title format you requested)
    plt.title(f"{N}-back task | Hidden step {hidden_step} vs Position step {pos_step}\n"
              f"Spatial Correlation (r = {corr:.3f}, p = {p_value:.3f})")
    plt.xlabel("Actual Grid Distance (units)")
    plt.ylabel("Hidden State Distance")
    sns.despine()
    plt.tight_layout()
    plt.show()
    
    return corr

def accuracy_results(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(accuracies)), accuracies, marker='o')
    plt.xlabel("N-back")
    plt.ylabel("Accuracy")
    plt.xticks(range(len(accuracies)))
    plt.grid(True)
    plt.show()