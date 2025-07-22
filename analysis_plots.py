from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hidden_tsne(model, dataset, step_pos=-1, step_hidden=-1):

    assert step_hidden != 0, "Step zero has trivial representation."
    
    model.eval()
    
    with torch.no_grad():
        _, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,step_hidden]
    np.random.seed(0)
    transformed = TSNE(2).fit_transform(hidden_states)
    
    sns.set()
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=analysis_dataset.y.numpy()[:,step_pos].astype('str'))
    plt.legend('', frameon=False)
    plt.title(f't-SNE of Hidden-states at step {step_hidden} colored according to position at step {step_pos}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')

def plot_hidden_pca(model, dataset, step_pos=-1, step_hidden=-1):

    assert step_hidden != 0, "Step zero has trivial representation."
    
    
    model.eval()
    
    with torch.no_grad():
        _, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,step_hidden]
    np.random.seed(0)
    transformed = PCA(2).fit_transform(hidden_states)
    
    sns.set()
    sns.scatterplot(x=transformed[:,0], y=transformed[:,1], hue=analysis_dataset.y.numpy()[:,step_pos].astype('str'))
    plt.legend('', frameon=False)

    plt.title(f'PCA of Hidden-states at step {step_hidden} colored according to position at step {step_pos}')
    plt.xlabel('Dim 1')
    plt.ylabel('Dim 2')
    
    
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
    
def past_coding_plot(model, dataset, n_back_max, plot_components=50):
    
    model.eval()
    
    with torch.no_grad():
        logits, hidden_states = model.forward(dataset.x.to(device), return_hidden=True)

    hidden_states = hidden_states.cpu().numpy()[:,-1]
    transformed = FastICA(plot_components).fit_transform(hidden_states)

    num_classes = logits.size()[1]
    num_components = transformed.shape[1]
    
    y = F.one_hot(analysis_dataset.y, num_classes=num_classes).numpy()

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
    plt.title('Explained variance of last state IC by previous positions')
    plt.xlabel('Independent Component')
    plt.ylabel('Delay')
    plt.show()
