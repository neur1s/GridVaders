from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from scipy import stats, spatial
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def plot_hidden(model, dataset, step_pos=-1, step_hidden=-1, method='tsne', label_over=False, n_back=None):

    assert step_hidden != 0, "Step zero has trivial representation."
    assert method in ['tsne', 'umap', 'pca']
    
    model.eval()
    
    with torch.no_grad():
        _, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,step_hidden]
    np.random.seed(0)
    if method == 'tsne':
        transformed = TSNE(2).fit_transform(hidden_states)
        method = 't-SNE'
    elif method == 'umap':
        transformed = UMAP().fit_transform(hidden_states)
        method = 'UMAP'
    elif method == 'pca':
        transformed = PCA(2).fit_transform(hidden_states)
        method = 'PCA'

    labels = np.array([str(idx2loc(idx)) for idx in dataset.y.numpy()[:,step_pos]])
    
    sns.set()
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=labels)
    
    plt.title(f'{method} of Hidden-states at step {step_hidden}\n colored according to position at step {step_pos} (N={n_back})')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    plt.legend(loc='center right', prop={'size': 7}, fancybox=True, shadow=True)

    if label_over:
        plt.legend('', frameon=False)
        for label in np.unique(labels):
            pos = transformed[labels == label].mean(axis=0)
            plt.text(pos[0], pos[1], label, horizontalalignment='center', verticalalignment='top')
            

def plot_hidden_3d(model, dataset, step_pos=-1, step_hidden=-1, method='tsne', label_over=False, n_back=None):

    assert step_hidden != 0, "Step zero has trivial representation."
    assert method in ['tsne', 'pca']
    
    model.eval()
    
    with torch.no_grad():
        _, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,step_hidden]
    np.random.seed(0)
    if method == 'tsne':
        transformed = TSNE(3).fit_transform(hidden_states)
        method = 't-SNE'
    elif method == 'pca':
        transformed = PCA(3).fit_transform(hidden_states)
        method = 'PCA'

    labels = np.array([str(idx2loc(idx)) for idx in dataset.y.numpy()[:,step_pos]])
    
    sns.set()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    colors = sns.color_palette(n_colors=len(np.unique(labels)))
    
    for color, label in zip(colors, np.unique(labels)):
        ax.scatter(transformed[labels == label,0], transformed[labels == label,1], transformed[labels == label,2], color=color, label=label)
    
    plt.title(f'{method} of Hidden-states at step {step_hidden}\n colored according to position at step {step_pos} (N={n_back})')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    ax.set(zlabel='Dim 3')
    plt.legend(loc='center right', prop={'size': 7}, fancybox=True, shadow=True)

    if label_over:
        plt.legend('', frameon=False)
        for label in np.unique(labels):
            pos = transformed[labels == label].mean(axis=0)
            plt.text(pos[0], pos[1], label, horizontalalignment='center', verticalalignment='top')

    
def scree_plot(model, dataset, plot_components=50):
    model.eval()
    
    with torch.no_grad():
        logits, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,-1]
    pca = PCA(hidden_states.shape[-1])
    transformed = pca.fit_transform(hidden_states)

    PC_values = np.arange(pca.n_components_) + 1
    plt.plot(PC_values[:plot_components], pca.explained_variance_ratio_[:plot_components], 'o-', linewidth=2, color='blue')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()
    
def past_coding_plot(model, dataset, n_back_max, plot_components=50, model_name=None):
    
    model.eval()
    
    with torch.no_grad():
        logits, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,-1]
    transformed = FastICA(plot_components).fit_transform(hidden_states)

    num_classes = logits.size()[1]
    num_components = transformed.shape[1]
    
    y = F.one_hot(dataset.y, num_classes=num_classes).numpy()

    r2_matrix = np.zeros([n_back_max, num_components])
    
    for n in range(n_back_max):
        pos = y[:, -(n+1)]
        betas = np.linalg.inv(pos.T @ pos) @ pos.T @ transformed
        resids = pos @ betas - transformed
        r2 = 1 - np.var(resids, axis=0)/np.var(transformed, axis=0)
        r2_matrix[n] = r2

    r2_matrix_ = r2_matrix[:,:plot_components]
    idx = np.lexsort(r2_matrix_[::-1])
    r2_matrix_ = r2_matrix_[:, idx]
    
    fig, ax = plt.subplots(figsize=(9,8))
    im = ax.imshow(r2_matrix_, aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im)
    sns.reset_orig()
    plt.title(f'Explained variance of last state IC by previous positions ({model_name})')
    plt.xlabel('Independent Component')
    plt.ylabel('Delay')
    plt.show()
    
    
def plot_spatial_correlation(model, dataset, step_pos=-1, step_hidden=-1, n_back=None):

    assert step_hidden != 0, "Step zero has trivial representation."
    
    model.eval()
    
    with torch.no_grad():
        _, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,step_hidden]
    grid_positions = dataset.y.numpy()[:,step_pos]
    
    # Calculate ACTUAL grid distances
    positions = np.array([(p//5, p%5) for p in grid_positions])
    grid_dists = spatial.distance.pdist(positions.astype(float))
    
    # Calculate hidden state distances
    hidden_dists = spatial.distance.pdist(hidden_states)
    
    corr, p_value = stats.pearsonr(grid_dists, hidden_dists)
    
    # Bin by integer distances (0,1,2,3,4,5,6)
    df = pd.DataFrame({
        'Grid Distance': grid_dists,
        'Hidden Distance': hidden_dists,
        'Distance Bin': np.floor(grid_dists).astype(int)  # Bin by integer part
    })
    
    # Plot
    sns.set()
    plt.figure(figsize=(5,5))
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
    
    # Label with actual distance ranges
    # bin_labels = [f"{int(bin)}-{int(bin)+1}" for bin in sorted(df['Distance Bin'].unique())]
    # plt.xticks(ticks=range(len(bin_labels)), labels=bin_labels)
    
    plt.title(f"{n_back}-back task, hidden state {step_hidden} and position at step {step_pos}\nSpatial Correlation (r = {corr:.3f}, p = {p_value:.3f})")
    plt.xlabel("Actual Grid Distance Range (units)")
    plt.ylabel("Hidden State Distance")
    plt.tight_layout()
    plt.show()
