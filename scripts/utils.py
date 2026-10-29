from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
from numpy.linalg import svd, eig
from scipy.sparse.linalg import svds
from tqdm import tqdm

def polar_rs(A):
    """
    SVD-based polar decomposition: A = R S  with R orthogonal (possibly det=-1) and S symmetric p.s.d.
    Returns (R, S, s) where s are singular values of A (principal stretches).
    """
    # Use torch.linalg.svd for faster computation (on CPU or GPU)
    if isinstance(A, np.ndarray):
        A_torch = torch.from_numpy(A)
    else:
        A_torch = A
    U, s, Vh = torch.linalg.svd(A_torch, full_matrices=False)
    U = U.cpu().numpy()
    s = s.cpu().numpy()
    Vt = Vh.cpu().numpy()
    R = U @ Vt
    S = Vt.T @ np.diag(s) @ Vt
    return R, S, s

def principal_rotation_angles(R, tol=1e-7):
    """
    Principal rotation angles of an orthogonal matrix R (radians, in (0, π]).
    Works even if det(R) = -1 (reflection contributes an angle ≈ π).
    """
    w, _ = torch.linalg.eig(torch.from_numpy(R))
    w = w.cpu().numpy()
    ang = np.abs(np.angle(w))              # in [0, π]
    ang = ang[ang > tol]                   # drop ~0 angles
    # Numerical eigenpairs come duplicated for conjugates; uniq to be tidy:
    ang = np.unique(np.round(ang, 12))
    return np.sort(ang)

def total_strain_from_sigmas(sigmas, eps=1e-14):
    """
    Intrinsic (Riemannian) total strain: ||log S||_F = sqrt(sum_i (log σ_i)^2).
    """
    s = np.clip(sigmas, eps, None)
    return np.linalg.norm(np.log(s))

def total_spin_from_angles(angles):
    """
    Total spin (geodesic distance on SO(d)): sqrt(sum_k θ_k^2).
    For det(R)=-1, an angle near π will appear; this counts as a large spin.
    """
    if angles.size == 0:
        return 0.0
    return np.sqrt(np.sum(angles**2))

def linearized_layerwise_dynamics(trajectories, device="cpu"):
    """
    Estimate linear dynamics and compute normalized residuals.

    Parameters:
        trajectories: np.ndarray of shape (n_data, n_layers, d)
        device: "cpu" or "cuda"

    Returns:
        A_list: list of (d, d) transformation matrices (NumPy arrays)
        residuals: list of normalized Frobenius residuals (floats)
    """
    n_data, n_layers, d = trajectories.shape
    # device = torch.device('cpu')
    trajectories_gpu = torch.from_numpy(trajectories).float().to(device)
    A_list = []
    residuals = []

    for t in tqdm(range(n_layers - 1), desc="Estimating linear dynamics"):
        X_t = trajectories_gpu[:, t, :]      # (n_data, d)
        X_tp1 = trajectories_gpu[:, t + 1, :]  # (n_data, d)
        
        # Center the tensors along the data dimension
        X_t = X_t - X_t.mean(dim=0, keepdim=True)
        X_tp1 = X_tp1 - X_tp1.mean(dim=0, keepdim=True)

        # Least-squares solution for A_t.T (shape d × d)
        # A_t_T = torch.linalg.lstsq(X_t, X_tp1, driver='gels').solution  # (d, d)
        A_t_T = torch.linalg.lstsq(X_t, X_tp1).solution  # (d, d)

        # Predicted next layer
        X_tp1_pred = X_t @ A_t_T  # (n_data, d)

        # Compute normalized residual using torch
        numerator = torch.norm(X_tp1 - X_tp1_pred, p='fro')
        denominator = torch.norm(X_tp1, p='fro')
        residual = (numerator / denominator).item()

        # Save CPU version of A_t
        A_list.append(A_t_T.T.cpu().numpy())  # (d, d)
        residuals.append(residual)

    return A_list, residuals

def calculate_rotation_angles(basis_vectors, dim = 0):
    """
    Calculate the angle between first basis vectors of consecutive layers.
    
    
    
    Args:
        basis_vectors: list of basis vector matrices from layered_PCA
    
    Returns:
        angles_degrees: array of rotation angles in degrees between consecutive layers
        cosine_similarities: array of cosine similarities between consecutive layers
    """
    n_layers = len(basis_vectors)
    angles_radians = []
    cosine_similarities = []
    
    for i in range(n_layers - 1):
        # Get first basis vectors from consecutive layers
        v1 = basis_vectors[i][:, dim]      # First basis vector of layer i
        v2 = basis_vectors[i + 1][:, dim]  # First basis vector of layer i+1
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cosine_sim = np.dot(v1, v2)
        cosine_similarities.append(cosine_sim)

        angle_rad = np.arccos(np.abs(cosine_sim))  # Use abs to get acute angle
        angles_radians.append(angle_rad)
    
    angles_degrees = np.array(angles_radians) * 180 / np.pi
    cosine_similarities = np.array(cosine_similarities)
    
    return angles_degrees, cosine_similarities

def calculate_pairwise_rotation_angles(basis_vectors, dim=0):
    """
    Calculate pairwise rotation angles between basis vectors of a specific dimension across layers.
    """
    n_layers = len(basis_vectors)
    
    # Initialize matrix to store angles
    angle_matrix = np.zeros((n_layers, n_layers))
    
    # Calculate angles between all pairs of layers for the specified basis vector dimension
    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            # Get the specified basis vector from each layer
            v1 = basis_vectors[i][:, dim]
            v2 = basis_vectors[j][:, dim]
            
            # Calculate cosine similarity
            cosine_sim = np.dot(v1, v2)
            
            # Calculate angle in degrees
            angle_rad = np.arccos(np.abs(cosine_sim))  # Use abs to get acute angle
            angle_deg = angle_rad * 180 / np.pi
            
            # Store in both (i, j) and (j, i) due to symmetry
            angle_matrix[i, j] = angle_deg
            angle_matrix[j, i] = angle_deg
    
    return angle_matrix

def get_rainbow_colors(embeddings, rainbowl_layer, rainbowl_axis):
    """
    Generate rainbow gradient colors based on values along a specific axis of a specific layer.
    
    Args:
        embeddings: list of arrays, each of shape (n_traj, nDim) - embeddings for each layer
        rainbowl_layer: int - layer index to use for coloring
        rainbowl_axis: int - axis index within that layer to use for coloring
    
    Returns:
        colors: array of shape (n_traj,) - normalized values for coloring (0 to 1)
        colormap: matplotlib colormap object
    """
    if rainbowl_layer is None:
        return None, None
    
    # Extract values from the specified layer and axis
    color_values = embeddings[rainbowl_layer][:, rainbowl_axis]
    
    # Normalize to [0, 1] range for colormap
    min_val = np.min(color_values)
    max_val = np.max(color_values)
    
    if max_val == min_val:
        # All values are the same
        normalized_colors = np.zeros_like(color_values)
    else:
        normalized_colors = (color_values - min_val) / (max_val - min_val)
    
    # Use rainbow colormap
    colormap = plt.cm.rainbow
        # colormap = plt.cm.viridis
    
    return normalized_colors, colormap

def layered_PCA(pilot_trajs, center = False, bulk_cutoff = None, fixed_basis_at_layer = None, GPU = True):
    """
    Perform PCA on each layer of the trajectory data.
    
    Args:
        pilot_trajs: numpy array of shape (n_traj, n_layer, nDim)
        center: bool, whether to center the data
        bulk_cutoff: int, cutoff for number of trajectories to use for covariance computation
        fixed_basis_at_layer: int, if provided, compute PCA basis only at this layer and reuse for all layers
    
    Returns:
        basis_vectors: list of arrays, each of shape (nDim, nDim) - PCA basis vectors for each layer
        variances: list of arrays, each of shape (nDim,) - explained variance for each component per layer
        embeddings: list of arrays, each of shape (n_traj, nDim) - projected embeddings for each layer
    """
    # Check that pilot_trajs have 3 dimensions: n_traj, n_layer, nDim
    assert len(pilot_trajs.shape) == 3, f"Expected 3D array, got {len(pilot_trajs.shape)}D"
    
    n_traj, n_layer, nDim = pilot_trajs.shape
    
    # # Validate fixed_basis_at_layer parameter
    # if fixed_basis_at_layer is not None:
    #     assert 0 <= fixed_basis_at_layer < n_layer, f"fixed_basis_at_layer must be between 0 and {n_layer-1}"
    
    basis_vectors = []
    variances = []
    embeddings = []
    
    # If fixed_basis_at_layer is specified, compute the fixed basis first
    fixed_eigenvectors = None
    if fixed_basis_at_layer is not None:
        # Extract data for the fixed basis layer: (n_traj, nDim)
        fixed_layer_data = pilot_trajs[:, fixed_basis_at_layer, :]
        
        # Center the data if requested
        if center:
            fixed_layer_data = fixed_layer_data - np.mean(fixed_layer_data, axis=0)
        
        # Compute covariance matrix
        if bulk_cutoff is not None:
            cov_matrix = np.cov(fixed_layer_data[:bulk_cutoff].T)
        else:
            cov_matrix = np.cov(fixed_layer_data.T)
           
        print(f"cov_matrix dtype: {cov_matrix.dtype}")
        
        # Perform eigendecomposition
        # Use GPU for eigendecomposition if available, else fallback to numpy         
        if torch.cuda.is_available() and GPU:
            print("GPU available")
            cov_matrix_torch = torch.from_numpy(cov_matrix).cuda()
            eigenvalues_torch, eigenvectors_torch = torch.linalg.eigh(cov_matrix_torch)
            eigenvalues = eigenvalues_torch.cpu().numpy()
            eigenvectors = eigenvectors_torch.cpu().numpy()
        else:
            print("GPU not available")
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        
        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Normalize eigenvectors (they should already be normalized from eigh, but ensure it)
        fixed_eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
    
    # For each layer, compute variances and embeddings
    for layer_idx in tqdm(range(n_layer), desc="Computing PCA per layer"):
        # Extract data for this layer: (n_traj, nDim)
        layer_data_uncentered = pilot_trajs[:, layer_idx, :]
        
        # Center the data if requested
        if center:
            layer_data = layer_data_uncentered - np.mean(layer_data_uncentered, axis=0)
        else:
            layer_data = layer_data_uncentered
        # Use fixed basis or compute new basis for this layer
        if fixed_basis_at_layer is not None:
            # Use the precomputed fixed basis
            eigenvectors = fixed_eigenvectors
            
            # Compute variances for this layer using the fixed basis
            # Project data onto the fixed basis and compute variance of each component
            layer_embeddings = layer_data @ eigenvectors
            eigenvalues = np.var(layer_embeddings, axis=0)
        else:
            # Compute covariance matrix for this layer
            if bulk_cutoff is not None:
                cov_matrix = np.cov(layer_data[:bulk_cutoff].T)
            else:
                cov_matrix = np.cov(layer_data.T)
            # print(f"cov_matrix dtype: {cov_matrix.dtype}")
            # Perform eigendecomposition          
            if torch.cuda.is_available() and GPU:
                print("GPU available")

                cov_matrix_torch = torch.from_numpy(cov_matrix).cuda()  # preserves float64 if that's what cov_matrix is
                # print(f"cov_matrix_torch dtype: {cov_matrix_torch.dtype}")
                eigenvalues_torch, eigenvectors_torch = torch.linalg.eigh(cov_matrix_torch)
                eigenvalues = eigenvalues_torch.cpu().numpy()
                eigenvectors = eigenvectors_torch.cpu().numpy()
            else:
                print("GPU not available")
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    
            # Sort eigenvalues and eigenvectors in descending order
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[sorted_indices]
            eigenvectors = eigenvectors[:, sorted_indices]
            
            # Normalize eigenvectors (they should already be normalized from eigh, but ensure it)
            eigenvectors = eigenvectors / np.linalg.norm(eigenvectors, axis=0)
            
            # Project data onto the principal components
            layer_embeddings = layer_data_uncentered @ eigenvectors
        
        # Store results
        basis_vectors.append(eigenvectors)  # Shape: (nDim, nDim)
        variances.append(eigenvalues)       # Shape: (nDim,)
        embeddings.append(layer_embeddings) # Shape: (n_traj, nDim)
    
    return basis_vectors, variances, embeddings


# ── Out-embedding artifacts (lm_head + final LayerNorm per checkpoint) ─────────

LARGE_MODEL_MIN_PARAMS_M = 2800   # models >= 2.8B: save outembed then purge HF weights


def _pythia_params_m(model_short: str) -> float:
    """Parameter count in millions from a model short name (e.g. pythia-1.4B → 1400)."""
    import re
    m = re.search(r'([\d.]+)\s*([mMbB])', model_short)
    if not m:
        raise ValueError(f'Cannot parse model size from {model_short!r}')
    val, unit = float(m.group(1)), m.group(2).lower()
    return val if unit == 'm' else val * 1000.0


def is_large_pythia_model(model_name: str) -> bool:
    return _pythia_params_m(model_name.split('/')[-1]) >= LARGE_MODEL_MIN_PARAMS_M


def outembed_path(traj_dir: str, step: int) -> str:
    import os
    return os.path.join(traj_dir, f'step{step}_outembed.npz')


def _lm_head_and_final_ln(model):
    if hasattr(model, 'lm_head'):
        lm_head = model.lm_head
    elif hasattr(model, 'embed_out'):
        lm_head = model.embed_out
    else:
        raise ValueError(f'Unsupported model architecture: {type(model)}')
    if hasattr(model, 'gpt_neox'):
        final_ln = model.gpt_neox.final_layer_norm
    elif hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_ln = model.model.norm
    else:
        raise ValueError(f'Cannot find final layer norm for {type(model)}')
    return lm_head, final_ln


def save_outembed_artifacts(model, traj_dir: str, step: int) -> str:
    """Save lm_head weights and final LayerNorm params for a checkpoint."""
    import os
    lm_head, final_ln = _lm_head_and_final_ln(model)
    path = outembed_path(traj_dir, step)
    os.makedirs(traj_dir, exist_ok=True)
    np.savez_compressed(
        path,
        lm_head=lm_head.weight.detach().cpu().numpy(),
        final_ln_weight=final_ln.weight.detach().cpu().numpy(),
        final_ln_bias=final_ln.bias.detach().cpu().numpy(),
        step=np.int64(step),
    )
    return path


def load_outembed_artifacts(traj_dir: str, step: int):
    """
    Load saved out-embedding artifacts.

    Returns
    -------
    (lm_head, final_ln_weight, final_ln_bias) or None if the file is missing.
    """
    import os
    path = outembed_path(traj_dir, step)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return data['lm_head'], data['final_ln_weight'], data['final_ln_bias']


def load_final_ln_for_step(traj_dir: str, step: int, model_short=None):
    """Return (weight, bias) of final LayerNorm saved for ``step``."""
    saved = load_outembed_artifacts(traj_dir, step)
    if saved is None:
        raise FileNotFoundError(
            f'Missing outembed file for step {step} in {traj_dir}'
            + (f' ({model_short})' if model_short else '')
        )
    _, ln_w, ln_b = saved
    return ln_w, ln_b


def apply_final_ln_numpy_array(arr, ln_w, ln_b, eps: float = 1e-5):
    """Apply GPTNeoX-style final LayerNorm along the last axis of ``arr``."""
    x = np.asarray(arr, dtype=np.float64)
    ln_w = np.asarray(ln_w, dtype=np.float64)
    ln_b = np.asarray(ln_b, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return (x_norm * ln_w + ln_b).astype(np.float32)


def purge_hf_revision(repo_id: str, revision: str, hf_hub_cache: str) -> None:
    """Delete one HF Hub revision snapshot from the local cache."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir(hf_hub_cache)
    ref_hits = {revision, f'refs/heads/{revision}'}
    hashes = [
        rev.commit_hash
        for repo in info.repos if repo.repo_id == repo_id
        for rev in repo.revisions
        if ref_hits & set(rev.refs)
    ]
    if hashes:
        info.delete_revisions(*hashes).execute()


def purge_hf_repo(repo_id: str, hf_hub_cache: str) -> None:
    """Delete all cached revisions for a Hugging Face repo."""
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir(hf_hub_cache)
    hashes = [
        rev.commit_hash
        for repo in info.repos if repo.repo_id == repo_id
        for rev in repo.revisions
    ]
    if hashes:
        info.delete_revisions(*hashes).execute()


def load_model_and_tokenizer(data_name=None, checkpoint=None, revision=None):
    """
    Load model and tokenizer based on either filename pattern or explicit checkpoint/revision.
    
    Args:
        data_name: Filename pattern like "pythia-70M-step0_perturbed_layer3_sigma0.001_length20_n4000_Color.npy"
        checkpoint: Explicit checkpoint name like "EleutherAI/pythia-70b" or "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
        revision: Explicit revision like "step20000" or "step-5125000"
        
    Returns:
        tuple: (model, tokenizer, model_name_short)
    """
    
    if data_name:
        # Extract model name and step from filename
        # Pattern: extract model name from filename like "pythia-70M-step0_perturbed_..."
        filename_base = data_name.split('/')[-1]  # Get just the filename
        
        # Parse step number from filename (e.g., "pythia-70M-step0" -> step=0)
        import re
        step_match = re.search(r'-step(\d+)', filename_base)
        step_number = step_match.group(1) if step_match else None
        
        checkpoints_match = re.search(r'-intermediate-checkpoints-(\d+)', filename_base)
        checkpoints_number = checkpoints_match.group(1) if checkpoints_match else None
        # Extract base model name (remove step part)
        if step_match:
            model_name_short = filename_base[:step_match.start()]  # "pythia-70M"
        elif checkpoints_match:
            model_name_short = filename_base[:checkpoints_match.start()] 
        else:
            model_name_short = filename_base.split('_')[0]  # Fallback to original method
        
        print(model_name_short)
        
        # Map short name to full model name
        model_name_mapping = {
            "Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
            "Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
            "LLama-2-7b": "meta-llama/LLama-2-7b",
            'Llama-2-7b-hf': "meta-llama/Llama-2-7b-hf",
            'LLama-2-13b': "meta-llama/LLama-2-13b",
            'gemma-2b': "google/gemma-2b",
            
            'pythia-1b': "EleutherAI/pythia-1b",
            'pythia-70M': "EleutherAI/pythia-70M",
            'pythia-160M': "EleutherAI/pythia-160M",
            'pythia-410M': "EleutherAI/pythia-410M",
            'pythia-410m': "EleutherAI/pythia-410M",
            'pythia-1.4B': "EleutherAI/pythia-1.4B",
            'pythia-1.4b': "EleutherAI/pythia-1.4B",
            'pythia-2.8B': "EleutherAI/pythia-2.8B",
            'pythia-6.9B': "EleutherAI/pythia-6.9B",
            'pythia-12B': "EleutherAI/pythia-12B",
            
            # SmolLM2 models
            'SmolLM2-1.7B': "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints",
            'SmolLM2-135M': "HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints",
        }

        model_name = model_name_mapping.get(model_name_short, model_name_short)
        
        # Load model with or without revision based on step number
        if step_number or checkpoints_number:
            # Handle different revision formats
            if model_name.startswith("HuggingFaceTB/SmolLM2"):
                revision = f"step-{checkpoints_number}"  # SmolLM2 uses "step-{number}" format
            else:
                revision = f"step{step_number}"   # Pythia uses "step{number}" format
                
            tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, output_hidden_states=True)
            model_name_short = f"{model_name_short}-step{step_number}"
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True)
        
    elif checkpoint and revision:
        # Load with explicit checkpoint and revision
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(checkpoint, revision=revision, output_hidden_states=True)
        
        # Generate model name short based on checkpoint and revision
        checkpoint_name = checkpoint.split('/')[-1]
        # Handle different revision formats for model name generation
        if revision.startswith('step-'):
            step_part = revision.split('-')[-1]  # Extract just the number from "step-5125000"
        else:
            step_part = revision.replace('step', '')  # Extract just the number from "step5125000"
        
        model_name_short = f"{checkpoint_name}-{step_part}"
        
    else:
        raise ValueError("Must provide either data_name or both checkpoint and revision")
    
    return model, tokenizer, model_name_short

def get_step_based_model(checkpoint, step):
    """
    Convenience function for loading step-based models.
    
    Args:
        checkpoint: Base checkpoint like "EleutherAI/pythia-70b" or "HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints"
        step: Step number like 20000 or 5125000
        
    Returns:
        tuple: (model, tokenizer, model_name_short)
    """
    # Determine revision format based on checkpoint type
    if checkpoint.startswith("HuggingFaceTB/SmolLM2"):
        revision = f"step-{step}"  # SmolLM2 format
    else:
        revision = f"step{step}"   # Pythia format
    
    return load_model_and_tokenizer(checkpoint=checkpoint, revision=revision) 

@torch.no_grad()
def metric_2d_from_proj(W, h, proj_2D):
    """
    Compute the 2D Fisher information metric (pullback of KL)
    restricted to a given orthonormal 2D basis in hidden space.

    Args:
        W        : (V, d) tensor
                   The model's final linear projection (vocab_size × hidden_dim).
                   Each row corresponds to the "logit weight vector" for a token.
        h        : (d,) tensor
                   The hidden state vector at which we want to evaluate the metric.
        proj_2D  : (2, d) tensor
                   An orthonormal basis for a 2D subspace of the hidden space.
                   Each row is a unit vector in hidden_dim, and rows are orthogonal.

    Returns:
        G2 : (2, 2) tensor
             The Fisher pullback metric in the given 2D plane.
        p  : (V,) tensor
             The softmax distribution over the vocabulary at this h.
    """

    z = W @ h                          # (V,)
    p = torch.softmax(z - z.max(), 0)  # (V,)
    B = proj_2D.T                      # (d, 2)
    A = W @ B                          # (V, 2)
    AD = p.unsqueeze(1) * A            # (V, 2), rows scaled by probabilities
    S2 = A.T @ AD                      # (2, 2)
    m = A.T @ p                        # (2,)
    G2 = S2 - torch.outer(m, m)        # (2, 2)

    return G2, p

# def plot_metric_contour_with_eig_axes(G, lim=2.0, res=400, levels=20, c_level=1.0, ax=None, legend = True, center = np.array([0,0])):
#     """
#     Plot contour lines of q(x,y) = [x y] G [x y]^T and overlay both eigen-directions:
#         - Red solid line: most sensitive axis (top eigenvector)
#         - Blue dashed line: less sensitive axis (second eigenvector)
#     Line lengths are scaled to match the semi-axes of the FIRST contour level (value = c_level).

#     Args:
#         G       : (2,2) positive semidefinite metric tensor
#         lim     : half-width of the plotting window in each axis
#         res     : grid resolution
#         levels  : total number of contour levels to draw
#         c_level : value of the FIRST contour (defines the reference ellipse)
#         ax      : matplotlib Axes to plot on. If None, creates a new figure and axes.
#     """
    
#     G = np.asarray(G, dtype=np.float64)

#     # Grid and quadratic form
#     x = np.linspace(-lim, lim, res)
#     y = np.linspace(-lim, lim, res)
#     X, Y = np.meshgrid(x, y)
#     pts = np.stack([X, Y], axis=-1)
#     q = np.einsum('...i,ij,...j->...', pts, G, pts)

#     # Eigen-decomposition (ascending), reorder so eigvals[0] is largest
#     eigvals, eigvecs = np.linalg.eigh(G)
#     order = eigvals.argsort()[::-1]
#     eigvals = eigvals[order]
#     eigvecs = eigvecs[:, order]

#     # Ensure positive values for a valid ellipse (handle tiny/zero eigenvalues)
#     eps = 1e-12
#     eigvals = np.clip(eigvals, eps, None)

#     # Semi-axes lengths of the ellipse v^T G v = c_level
#     a = np.sqrt(c_level / eigvals[0])  # along top eigenvector
#     b = np.sqrt(c_level / eigvals[1])  # along second eigenvector

#     # Contour levels starting at c_level
#     # Space them geometrically to show more structure near the origin
#     if levels < 2:
#         level_values = [c_level]
#     else:
#         # span up to a value that fits in the view based on lim and G
#         # approximate max q at boundary for scaling
#         corner = np.array([lim, lim])
#         approx_max = float(corner @ G @ corner)
#         approx_max = max(approx_max, c_level * 10)
#         level_values = np.geomspace(c_level, approx_max, levels)

#     # Use provided ax or create new one
#     created_fig = False
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(6, 6))
#         created_fig = True

#     cs = ax.contour(X, Y, q, levels=level_values)
#     ax.clabel(cs, inline=True, fontsize=8)

#     # Draw eigen axes, scaled to ellipse semi-axes for the first contour
#     v1 = eigvecs[:, 0]  # top eigenvector
#     v2 = eigvecs[:, 1]  # second eigenvector

#     # Red solid line along ±a*v1
#     t = np.linspace(-1.0, 1.0, 200)
#     xs1 = t * a * v1[0]
#     ys1 = t * a * v1[1]
#     ax.plot(xs1, ys1, color='red', linewidth=2, label='more sensitive (λ₁)')

#     # Blue dashed line along ±b*v2
#     xs2 = t * b * v2[0]
#     ys2 = t * b * v2[1]
#     ax.plot(xs2, ys2, color='blue', linestyle='--', linewidth=2, label='less sensitive (λ₂)')

#     # ax.set_xlabel("Direction 1")
#     # ax.set_ylabel("Direction 2")
#     # ax.set_title(f"Metric contour with eigen-directions (first contour = {c_level:g})")
#     ax.set_aspect('equal', adjustable='box')
#     if legend:
#         ax.legend(loc='upper right')
#     if created_fig:
#         plt.show()

# ---------------------------------------------------------------------------
# Embedding / trajectory cloud alignment (LOT Pythia angle analysis, etc.)
# ---------------------------------------------------------------------------

def unit_vector(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def first_principal_axis(X, center=True):
    """First PC direction (feature space) via truncated SVD; unit vector."""
    X = np.asarray(X, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if X.shape[0] < 2:
        return unit_vector(X.sum(axis=0))
    _, _, vt = svds(X, k=1)
    return unit_vector(vt[0])


def align_sign(v, prev):
    """Flip v to maximise agreement with previous snapshot (PCA / dipole sign fix)."""
    v = unit_vector(v)
    if prev is not None and np.dot(v, prev) < 0:
        v = -v
    return v


def direction_overlap(u, v):
    """|cos θ| between undirected unit directions; 1 = aligned, 0 = orthogonal."""
    u, v = unit_vector(u), unit_vector(v)
    return float(np.clip(abs(np.dot(u, v)), 0.0, 1.0))


def squared_matrix_cosine_similarity(A, B):
    """|A^T B|_F^2 / (|A|_F^2 |B|_F^2) for row-stored point clouds (via A @ B.T)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    cross = A @ B.T
    num = np.sum(cross ** 2)
    den = np.sum(A ** 2) * np.sum(B ** 2)
    return float(num / den) if den > 0 else np.nan


def _load_embedding_and_traj_last(model_short, step, traj_dir):
    """Return (W_emb, traj_last): lm_head weights (V, D) and last-layer traj cloud."""
    import plotter_utils

    files = plotter_utils.find_trajectory_files(
        model_short, step, traj_dir, required='global',
    )
    if files is None or files[1] is None:
        raise FileNotFoundError(f'No global trajectory for step {step}')

    traj = np.load(files[1], mmap_mode='r')[:, :, -1, :]
    traj = np.expand_dims(traj, axis=0)

    loop_model, _, _ = load_model_and_tokenizer(
        checkpoint=f'EleutherAI/{model_short}', revision=f'step{step}',
    )
    W = plotter_utils.get_lm_head(loop_model).weight.data.cpu().numpy().astype(np.float64)
    traj_normed = plotter_utils.apply_final_layer_norm(loop_model, traj)
    traj_last = traj_normed[0, :, -1, :]

    del loop_model
    torch.cuda.empty_cache()
    gc.collect()
    return W, traj_last


def load_embedding_and_traj_last(model_short, step, traj_dir):
    """Return (W_emb, traj_last) for one checkpoint; W (V, D), traj (n_chunks, D)."""
    return _load_embedding_and_traj_last(model_short, step, traj_dir)


# ---------------------------------------------------------------------------
# PCA alignment & interactive embedding plotters (shared by toy_model and
# real_LLMs_embedding_checkpoints notebooks)
# ---------------------------------------------------------------------------

def frequency_dipole(
    W,
    p,
    center_frequencies=True,
    method="moment",
    ridge_lambda=None,
    probability_floor=1e-10,
    exclude_zeros=True,
):
    """
    Compute the frequency dipole of an embedding matrix.

    The default ``method="moment"`` preserves the original behavior: assign
    token charges from probabilities and compute a charge-weighted average in
    embedding space.

    ``method="optimal_prior"`` computes the information-geometric optimal prior
    direction ``d_star``: the unit direction whose unembedding projection
    ``softmax(beta_star * W @ d_star)`` best recovers the unigram distribution
    in least-squares log-probability space. This differs from ``d_tilde`` by
    fitting a regression coefficient against centered log-probabilities rather
    than taking a weighted average in probability space. The returned ``r2``
    is the coefficient of determination for that one-direction log-frequency
    fit; values close to 1 mean a single unembedding direction captures the
    empirical log-frequency structure well. The returned ``nkl`` compares the
    fitted softmax prior to the empirical unigram distribution:
    ``KL(p_unigram || p_prior) / KL(p_unigram || uniform)``.

    Assign each token charge ``q_k = p_k - mean(p)`` (when ``center_frequencies``)
    and define:

        d       = Σ_k  q_k · W_k              (charge-weighted centroid)
        Z       = Σ_k  ‖W_k‖                  (partition function / total norm)
        d_tilde = d / Z                        (normalised dipole)

    Args:
        W : (n, m) array-like – embedding matrix; each row is one token's vector.
        p : (n,)  array-like – token probabilities or empirical frequencies.
        center_frequencies : if True, use charges ``p - mean(p)``; if False, use
                           ``p`` directly as weights.
        method : "moment" keeps the original frequency dipole; "optimal_prior"
                 returns a dict with d_star, beta_star, and r2.
        ridge_lambda : optional ridge penalty for "optimal_prior". If None, use
                       1e-4 * trace(W_tilde.T @ W_tilde) / d_model.
        probability_floor : floor used before logs when zeros are smoothed.
        exclude_zeros : if True for "optimal_prior", tokens with p <= 0 are
                        excluded before taking logs; otherwise they are floored.

    Returns:
        For method="moment":
            d_tilde : (m,) ndarray – the normalised frequency dipole.
            d       : (m,) ndarray – the unnormalised dipole Σ q_k W_k.
            Z       : float        – the partition function Σ ‖W_k‖.
        For method="optimal_prior":
            dict with d_star, beta_star, r2, nkl, kl_to_prior,
            kl_to_uniform, ridge_lambda, d_tilde, cos_d_star_d_tilde, and
            n_tokens.
    """
    W = np.asarray(W, dtype=float)
    p = np.asarray(p, dtype=float)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array with shape (n_tokens, d_model).")
    if p.ndim != 1 or p.shape[0] != W.shape[0]:
        raise ValueError("p must be a 1D array with length matching W.shape[0].")

    if method == "optimal_prior":
        if exclude_zeros:
            mask = p > 0
            W_fit = W[mask]
            p_fit = p[mask]
        else:
            mask = np.ones_like(p, dtype=bool)
            W_fit = W
            p_fit = np.maximum(p, probability_floor)

        if W_fit.shape[0] == 0:
            raise ValueError("No tokens have usable probability information.")

        p_fit = np.asarray(p_fit, dtype=float)
        p_fit = p_fit / p_fit.sum()
        p_fit = np.maximum(p_fit, probability_floor)
        p_fit = p_fit / p_fit.sum()

        W_tilde = W_fit - W_fit.mean(axis=0, keepdims=True)
        ell = np.log(p_fit)
        ell = ell - ell.mean()

        gram = W_tilde.T @ W_tilde
        rhs = W_tilde.T @ ell
        d_model = W_tilde.shape[1]
        if ridge_lambda is None:
            ridge_lambda = 1e-4 * float(np.trace(gram)) / max(d_model, 1)

        coef = np.linalg.solve(gram + ridge_lambda * np.eye(d_model), rhs)
        coef_norm = np.linalg.norm(coef)
        if coef_norm <= 0:
            raise ValueError("Optimal prior direction has zero norm.")

        d_star = coef / coef_norm

        d_tilde, _, _ = frequency_dipole(
            W_fit,
            p_fit,
            center_frequencies=center_frequencies,
            method="moment",
        )
        d_tilde_norm = np.linalg.norm(d_tilde)
        cos_d_star_d_tilde = np.nan
        if d_tilde_norm > 0:
            d_tilde_unit = d_tilde / d_tilde_norm
            cos_d_star_d_tilde = float(np.dot(d_star, d_tilde_unit))
            if cos_d_star_d_tilde < 0:
                d_star = -d_star
                cos_d_star_d_tilde = -cos_d_star_d_tilde

        wd = W_tilde @ d_star
        denom = float(wd @ wd)
        beta_star = float((wd @ ell) / denom) if denom > 0 else np.nan
        residual = beta_star * wd - ell
        ell_norm_sq = float(ell @ ell)
        r2 = 1.0 - float(residual @ residual) / ell_norm_sq if ell_norm_sq > 0 else np.nan

        prior_logits = beta_star * (W_fit @ d_star)
        prior_logits = prior_logits - np.max(prior_logits)
        log_prior_fit = prior_logits - np.log(np.exp(prior_logits).sum())
        empirical_log = np.log(p_fit)
        kl_to_prior = float(np.sum(p_fit * (empirical_log - log_prior_fit)))
        kl_to_uniform = float(np.sum(p_fit * (empirical_log + np.log(p_fit.size))))
        nkl = kl_to_prior / kl_to_uniform if kl_to_uniform > 0 else np.nan

        return {
            "d_star": d_star,
            "beta_star": beta_star,
            "r2": r2,
            "nkl": nkl,
            "kl_to_prior": kl_to_prior,
            "kl_to_uniform": kl_to_uniform,
            "ridge_lambda": float(ridge_lambda),
            "d_tilde": d_tilde,
            "cos_d_star_d_tilde": cos_d_star_d_tilde,
            "n_tokens": int(W_fit.shape[0]),
            "mask": mask,
        }

    if method != "moment":
        raise ValueError("method must be 'moment' or 'optimal_prior'.")

    q = p - p.mean() if center_frequencies else p
    norms = np.linalg.norm(W, axis=1)        # (n,)
    Z     = norms.sum()                       # scalar
    d     = (q[:, None] * W).sum(axis=0)     # (m,)
    d_tilde = d / Z if Z > 0 else d
    return d_tilde, d, Z


def extract_checkpoint_vectors(model_short, step, traj_dir, p_zipf):
    """Load one checkpoint; return unit vectors (embed_pc0, dipole, traj_pc0)."""
    W, traj_last = _load_embedding_and_traj_last(model_short, step, traj_dir)
    embed_pc0 = first_principal_axis(W, center=True)
    traj_pc0 = first_principal_axis(traj_last, center=True)
    d_tilde, _, _ = frequency_dipole(W, p_zipf)
    return embed_pc0, unit_vector(d_tilde), traj_pc0


def frequency_radial_moment(W, p, center_frequencies=True):
    """
    Compute the frequency radial moment of an embedding matrix.

    Assign each token charge ``q_k = p_k - mean(p)`` (when ``center_frequencies``)
    and define:

        M       = Σ_k  q_k ‖W_k‖²           (raw radial moment, scalar)
        ‖W‖_F²  = Σ_k ‖W_k‖²              (total squared embedding norm)
        m_tilde = M / ‖W‖_F²               (normalized radial moment; scale invariant)

    Args:
        W : (n, m) array-like – embedding matrix; each row is one token's vector.
        p : (n,)  array-like – token probabilities or empirical frequencies.
        center_frequencies : if True, use charges ``p - mean(p)``; if False, use
                           ``p`` directly as charges.

    Returns:
        m_tilde : float – normalized frequency radial moment.
        M       : float – raw radial moment.
        W_fro_sq : float – ‖W‖_F² = Σ_k ‖W_k‖².
    """
    W = np.asarray(W, dtype=float)
    p = np.asarray(p, dtype=float)
    q = p - p.mean() if center_frequencies else p
    sq_norms = np.einsum('ij,ij->i', W, W)   # ‖W_k‖² per token
    W_fro_sq = sq_norms.sum()
    M = float(np.dot(q, sq_norms))
    m_tilde = M / W_fro_sq if W_fro_sq > 0 else 0.0
    return m_tilde, M, W_fro_sq


def flip_align(embeddings_PCA, axes_to_flip, companion=None):
    """
    Greedily fix PC sign-flips between consecutive snapshots.

    For each pair of adjacent snapshots and each requested PC axis, compare
    the L2 distance between the two projections with and without a sign flip
    on that axis; apply the flip that minimises the distance.

    Args:
        embeddings_PCA : list of (n_points, n_components) arrays, one per snapshot.
        axes_to_flip   : sequence of int – PC axes to consider (e.g. (0, 1)).
        companion      : optional list of arrays with the same length as
                         ``embeddings_PCA``; the same sign flips are applied to
                         ``companion[i+1]`` (e.g. dark-matter projections).

    Returns:
        If ``companion`` is None: list of sign-corrected visible embeddings.
        Otherwise: ``(aligned_visible, aligned_companion)`` tuple.
    """
    aligned = [arr.copy() for arr in embeddings_PCA]
    aligned_companion = (
        [arr.copy() for arr in companion] if companion is not None else None
    )
    for i in range(len(aligned) - 1):
        a, b = aligned[i], aligned[i + 1]
        for d in axes_to_flip:
            if np.linalg.norm(a[:, d] + b[:, d]) < np.linalg.norm(a[:, d] - b[:, d]):
                b[:, d] *= -1
                if aligned_companion is not None:
                    aligned_companion[i + 1][:, d] *= -1
    if aligned_companion is not None:
        return aligned, aligned_companion
    return aligned


def _pca_index_scatter(ax, x, y, n_points, *, s=100, alpha=0.8, marker='o', zorder=2):
    """Scatter coloured by point index: red = low index, blue = high index."""
    return ax.scatter(
        x, y,
        c=np.arange(n_points), cmap='coolwarm_r',
        s=s, alpha=alpha, marker=marker, zorder=zorder,
    )


def _pca_dark_matter_scatter(ax, x, y, n_points, *, s=100, alpha=0.8, marker='o', zorder=3):
    """Scatter dark-matter points coloured by index (viridis)."""
    return ax.scatter(
        x, y,
        c=np.arange(n_points), cmap='viridis',
        s=s, alpha=alpha, marker=marker, zorder=zorder,
    )


def _pca_plot_axes(figsize=(7, 5), dpi=120, log_polar=False):
    if log_polar:
        return plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})
    return plt.subplots(figsize=figsize, dpi=dpi)


def _pca_figure_layout(log_polar=False, n_colorbars=1, figsize=(7, 5), dpi=120):
    """
    Return ``(fig, ax, caxes)`` for PCA slider plots.

    ``caxes`` is None when ``n_colorbars == 1`` (use ``plt.colorbar(..., ax=ax)``).
    Otherwise ``caxes`` is a list of dedicated colorbar axes (works with polar).
    """
    if n_colorbars <= 1:
        fig, ax = _pca_plot_axes(figsize=figsize, dpi=dpi, log_polar=log_polar)
        return fig, ax, None

    fig_w = figsize[0] + 1.5 * n_colorbars
    fig = plt.figure(figsize=(fig_w, figsize[1]), dpi=dpi, layout='constrained')
    width_ratios = [1.0] + [0.07] * n_colorbars
    gs = fig.add_gridspec(1, 1 + n_colorbars, width_ratios=width_ratios)
    if log_polar:
        ax = fig.add_subplot(gs[0, 0], projection='polar')
    else:
        ax = fig.add_subplot(gs[0, 0])
    caxes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_colorbars)]
    return fig, ax, caxes


def _pca_xy_to_plot_coords(x, y, log_polar, r_floor=None):
    """Return (plot_x, plot_y) for Cartesian or log-polar (polar projection) axes."""
    if not log_polar:
        return x, y
    r = np.hypot(x, y)
    theta = np.arctan2(y, x)
    if r_floor is None:
        r_pos = r[r > 0]
        r_floor = (0.5 * r_pos.min()) if r_pos.size else 1e-12
    return theta, np.maximum(r, r_floor)


def _pca_global_polar_rlim(embeddings_PCA_aligned, axes_to_plot, bias_projections=None,
                          embeddings_dark_PCA_aligned=None, *, pad_frac=0.05):
    """Radial limits (linear r) spanning every snapshot; for log-polar fixed_scale."""
    ax1, ax2 = axes_to_plot
    radii = []
    for pcs in embeddings_PCA_aligned:
        radii.append(np.hypot(pcs[:, ax1], pcs[:, ax2]).ravel())
    if embeddings_dark_PCA_aligned is not None:
        for pcs in embeddings_dark_PCA_aligned:
            radii.append(np.hypot(pcs[:, ax1], pcs[:, ax2]).ravel())
    if bias_projections is not None:
        for b_proj in bias_projections:
            if b_proj is not None:
                radii.append(np.array([np.hypot(b_proj[0], b_proj[1])]))
    r_flat = np.concatenate(radii) if radii else np.array([0.0])
    r_pos = r_flat[r_flat > 0]
    r_min = (0.5 * r_pos.min()) if r_pos.size else 1e-12
    r_max = float(r_pos.max()) if r_pos.size else 1.0
    if r_max <= r_min:
        r_max = r_min * 10.0
    pad = pad_frac * (r_max - r_min)
    return r_min, r_max + pad


def _apply_pca_polar_rlim(ax, r_min, r_max):
    ax.set_rscale('log')
    ax.set_rlim(r_min, r_max)


def plot_pca_cloud_slider(embeddings_PCA_aligned, axes_to_plot=(0, 1), labels=None,
                          bias_projections=None, log_polar=False, fixed_scale=False,
                          dark_matter=False, embeddings_dark_PCA_aligned=None):
    """
    Interactive scatter plot of PCA projections, coloured by point index.

    A slider lets the user step through every snapshot in
    ``embeddings_PCA_aligned``.

    Args:
        embeddings_PCA_aligned : list of (n_points, n_components) arrays.
        axes_to_plot           : pair of ints selecting which PC axes to plot.
        labels                 : optional list whose i-th entry is used in the
                                 plot title (e.g. training-step numbers).
                                 Defaults to the snapshot index.
        bias_projections       : optional list of (2,) arrays, one per snapshot,
                                 giving the bias vector projected onto
                                 (axes_to_plot[0], axes_to_plot[1]).  When
                                 provided, an arrow from the origin is drawn.
        log_polar              : if True, plot in polar coordinates with a
                                 logarithmic radial axis.
        fixed_scale            : if True with log_polar, use the same radial
                                 limits on every snapshot (global min/max r).
        dark_matter            : if True, overlay invisible / dark-matter rows
                                 projected onto the same PC axes.
        embeddings_dark_PCA_aligned : list of (n_dark, n_components) arrays,
                                 one per snapshot (same length as visible list).
                                 Required when ``dark_matter=True``.
    """
    import ipywidgets as widgets

    ax1, ax2 = axes_to_plot
    n = len(embeddings_PCA_aligned)

    if dark_matter:
        if embeddings_dark_PCA_aligned is None:
            raise ValueError(
                "embeddings_dark_PCA_aligned is required when dark_matter=True"
            )
        if len(embeddings_dark_PCA_aligned) != n:
            raise ValueError(
                "embeddings_dark_PCA_aligned must have the same length as "
                "embeddings_PCA_aligned"
            )

    if fixed_scale and not log_polar:
        raise ValueError("fixed_scale=True requires log_polar=True")

    polar_rlim = None
    r_floor = None
    if fixed_scale:
        polar_rlim = _pca_global_polar_rlim(
            embeddings_PCA_aligned, axes_to_plot, bias_projections,
            embeddings_dark_PCA_aligned=(
                embeddings_dark_PCA_aligned if dark_matter else None
            ),
        )
        r_floor = polar_rlim[0]

    def _plot(snapshot_idx):
        pcs   = embeddings_PCA_aligned[snapshot_idx]
        label = labels[snapshot_idx] if labels is not None else snapshot_idx

        x_raw, y_raw = pcs[:, ax1], pcs[:, ax2]
        plot_x, plot_y = _pca_xy_to_plot_coords(
            x_raw, y_raw, log_polar, r_floor=r_floor,
        )

        n_colorbars = 2 if dark_matter else 1
        fig, ax, caxes = _pca_figure_layout(log_polar=log_polar, n_colorbars=n_colorbars)
        sc = _pca_index_scatter(ax, plot_x, plot_y, pcs.shape[0], zorder=3)

        sc_dm = None
        if dark_matter:
            dm_pcs = embeddings_dark_PCA_aligned[snapshot_idx]
            dm_x_raw, dm_y_raw = dm_pcs[:, ax1], dm_pcs[:, ax2]
            dm_plot_x, dm_plot_y = _pca_xy_to_plot_coords(
                dm_x_raw, dm_y_raw, log_polar, r_floor=r_floor,
            )
            sc_dm = _pca_dark_matter_scatter(
                ax, dm_plot_x, dm_plot_y, dm_pcs.shape[0], zorder=2,
            )

        if log_polar:
            if polar_rlim is not None:
                _apply_pca_polar_rlim(ax, *polar_rlim)
            else:
                ax.set_rscale('log')

        if bias_projections is not None:
            b_proj = bias_projections[snapshot_idx]   # (2,)
            if log_polar:
                b_theta, b_r = _pca_xy_to_plot_coords(
                    np.array([b_proj[0]]), np.array([b_proj[1]]), True,
                    r_floor=r_floor,
                )
                b_xy = (float(b_theta[0]), float(b_r[0]))
            else:
                b_xy = (b_proj[0], b_proj[1])
            r_origin = (polar_rlim[0] if polar_rlim is not None else ax.get_rmin())
            ax.annotate(
                "", xy=b_xy,
                xytext=(0.0, 0.0) if not log_polar else (0.0, r_origin),
                arrowprops=dict(arrowstyle="-|>", color='black',
                                lw=2.0, mutation_scale=16),
                zorder=5,
            )
            origin = (0.0, 0.0) if not log_polar else (0.0, r_origin)
            ax.scatter(*origin, color='black', s=30, zorder=6, marker='+')
            ax.annotate(r'$\mathbf{b}$', b_xy,
                        textcoords='offset points', xytext=(5, 5),
                        fontsize=10, color='black', fontweight='bold')

        if not log_polar:
            ax.set_xlabel(f'PC{ax1}')
            ax.set_ylabel(f'PC{ax2}')
        scale_tag = ' (log-polar)' if log_polar else ''
        dm_tag = ' + dark matter' if dark_matter else ''
        ax.set_title(
            f'Embedding Cloud: PC{ax1} vs PC{ax2}{scale_tag}{dm_tag}\nSnapshot {label}'
        )
        ax.grid(True, alpha=0.3)
        if not log_polar:
            ax.set_aspect('equal', adjustable='box')

        if caxes is None:
            fig.colorbar(
                sc, ax=ax,
                label='Visible index' if dark_matter else 'Data index',
                shrink=0.85 if log_polar else 1.0,
                pad=0.12 if log_polar else 0.04,
            )
            fig.tight_layout()
        else:
            fig.colorbar(sc, cax=caxes[0], label='Visible index')
            fig.colorbar(sc_dm, cax=caxes[1], label='Dark matter index')

        plt.show()

    widgets.interact(
        _plot,
        snapshot_idx=widgets.IntSlider(
            value=n - 1, min=0, max=n - 1, step=1,
            description='Snapshot',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px'),
        ),
    )


def plot_delta_w_slider(embeddings_PCA_aligned, w_checkpoints,
                        axes_to_plot=(0, 1), labels=None, log_polar=False,
                        fixed_scale=False):
    """
    Interactive scatter plot of PCA projections, coloured by ‖ΔW‖ per point.

    Delta norms are precomputed as ``‖W[i] − W[i−1]‖₂`` per row; snapshot 0
    gets all-zero colours (no prior step).  A single shared colour scale spans
    all snapshots.

    Args:
        embeddings_PCA_aligned : list of (n_points, n_components) arrays.
        w_checkpoints          : list of (n_points, hidden_dim) tensors or arrays,
                                 one per snapshot (same order as embeddings).
        axes_to_plot           : pair of ints selecting which PC axes to plot.
        labels                 : optional list of snapshot labels for plot titles.
        log_polar              : if True, plot in polar coordinates with a
                                 logarithmic radial axis.
        fixed_scale            : if True with log_polar, use the same radial
                                 limits on every snapshot (global min/max r).
    """
    import ipywidgets as widgets

    ax1, ax2 = axes_to_plot
    n = len(embeddings_PCA_aligned)

    if fixed_scale and not log_polar:
        raise ValueError("fixed_scale=True requires log_polar=True")

    polar_rlim = None
    r_floor = None
    if fixed_scale:
        polar_rlim = _pca_global_polar_rlim(embeddings_PCA_aligned, axes_to_plot)
        r_floor = polar_rlim[0]

    _w = [cp.numpy() if hasattr(cp, 'numpy') else np.asarray(cp)
          for cp in w_checkpoints]
    delta_norms_all = [np.zeros(_w[0].shape[0])]
    for i in range(1, len(_w)):
        delta_norms_all.append(np.linalg.norm(_w[i] - _w[i - 1], axis=1))
    global_vmax = (max(a.max() for a in delta_norms_all[1:])
                   if len(delta_norms_all) > 1 else 1.0)

    def _plot(snapshot_idx):
        pcs         = embeddings_PCA_aligned[snapshot_idx]
        delta_norms = delta_norms_all[snapshot_idx]
        label       = labels[snapshot_idx] if labels is not None else snapshot_idx

        x_raw, y_raw = pcs[:, ax1], pcs[:, ax2]
        plot_x, plot_y = _pca_xy_to_plot_coords(
            x_raw, y_raw, log_polar, r_floor=r_floor,
        )

        fig, ax = _pca_plot_axes(log_polar=log_polar)
        sc = ax.scatter(
            plot_x, plot_y,
            c=delta_norms, cmap='viridis',
            # vmin=0, vmax=global_vmax,
            s=100, alpha=0.8, marker='o',
        )
        plt.colorbar(sc, ax=ax, label='‖ΔW‖  (from previous snapshot)')

        if log_polar:
            if polar_rlim is not None:
                _apply_pca_polar_rlim(ax, *polar_rlim)
            else:
                ax.set_rscale('log')

        if not log_polar:
            ax.set_xlabel(f'PC{ax1}')
            ax.set_ylabel(f'PC{ax2}')
        scale_tag = ' (log-polar)' if log_polar else ''
        ax.set_title(f'ΔW Magnitude in PCA Space{scale_tag}\nSnapshot {label}')
        ax.grid(True, alpha=0.3)
        if not log_polar:
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    widgets.interact(
        _plot,
        snapshot_idx=widgets.IntSlider(
            value=n - 1, min=0, max=n - 1, step=1,
            description='Snapshot',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px'),
        ),
    )


def plot_pca_cloud_with_highlights_slider(
    embeddings_PCA_aligned, tag_tokens, tag_token_ids,
    axes_to_plot=(0, 1), labels=None, log_polar=False, fixed_scale=False,
):
    """
    Interactive PCA scatter coloured by token index with labelled highlights.

    Like ``plot_pca_cloud_slider`` but additionally marks a selected set of
    tokens with pink dots and bold text labels.

    Args:
        embeddings_PCA_aligned : list of (vocab_size, n_components) arrays.
        tag_tokens             : list of token strings to highlight.
        tag_token_ids          : list of int vocab indices matching tag_tokens.
        axes_to_plot           : pair of ints for PC axes.
        labels                 : optional list of snapshot labels (e.g. step numbers).
        log_polar              : if True, plot in polar coordinates with a
                                 logarithmic radial axis.
        fixed_scale            : if True with log_polar, use the same radial
                                 limits on every snapshot (global min/max r).
    """
    import ipywidgets as widgets

    ax1, ax2 = axes_to_plot
    n = len(embeddings_PCA_aligned)

    if fixed_scale and not log_polar:
        raise ValueError("fixed_scale=True requires log_polar=True")

    polar_rlim = None
    r_floor = None
    if fixed_scale:
        polar_rlim = _pca_global_polar_rlim(embeddings_PCA_aligned, axes_to_plot)
        r_floor = polar_rlim[0]

    # Filter out None entries and any IDs out of range for the projection array
    vocab_size = embeddings_PCA_aligned[0].shape[0]
    valid = [
        (tok, int(tid))
        for tok, tid in zip(tag_tokens, tag_token_ids)
        if tok is not None and tid is not None and 0 <= int(tid) < vocab_size
    ]
    _tag_tokens    = [t for t, _ in valid]
    _tag_token_ids = [i for _, i in valid]

    def _plot(snapshot_idx):
        proj  = embeddings_PCA_aligned[snapshot_idx]
        label = labels[snapshot_idx] if labels is not None else snapshot_idx

        x_raw, y_raw = proj[:, ax1], proj[:, ax2]
        plot_x, plot_y = _pca_xy_to_plot_coords(
            x_raw, y_raw, log_polar, r_floor=r_floor,
        )

        fig, ax = _pca_plot_axes(figsize=(7, 7), dpi=150, log_polar=log_polar)
        sc = _pca_index_scatter(ax, plot_x, plot_y, proj.shape[0], s=8, alpha=0.5)
        plt.colorbar(sc, ax=ax, label='Token index')

        if log_polar:
            if polar_rlim is not None:
                _apply_pca_polar_rlim(ax, *polar_rlim)
            else:
                ax.set_rscale('log')

        highlight_proj = proj[_tag_token_ids] if _tag_token_ids else []
        for tok, coord in zip(_tag_tokens, highlight_proj):
            hx, hy = coord[ax1], coord[ax2]
            if log_polar:
                hx, hy = _pca_xy_to_plot_coords(
                    np.array([hx]), np.array([hy]), True, r_floor=r_floor,
                )
                hx, hy = float(hx[0]), float(hy[0])
            ax.scatter(hx, hy, s=100, color='pink', edgecolor='k', zorder=5)
            ax.text(hx, hy, rf"$\mathbf{{{tok}}}$",
                    fontsize=9, color='black', zorder=6, ha='right', va='bottom')

        if not log_polar:
            ax.set_xlabel(f'PC{ax1}')
            ax.set_ylabel(f'PC{ax2}')
        scale_tag = ' (log-polar)' if log_polar else ''
        ax.set_title(f'Output Embedding PCA{scale_tag}  —  snapshot {label}')
        ax.grid(True)
        if not log_polar:
            ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        plt.show()

    widgets.interact(
        _plot,
        snapshot_idx=widgets.IntSlider(
            value=n - 1, min=0, max=n - 1, step=1,
            description='Snapshot',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px'),
        ),
    )


def plot_metric_contour_with_eig_axes(
    G, lim=2.0, res=400, levels=20, c_level=1.0, ax=None, legend=True, center=np.array([0, 0]), eigen_axis = False
):
    """
    Plot contour lines of q(x,y) = [x y] G [x y]^T and overlay both eigen-directions,
    centered at 'center'.

    Args:
        G       : (2,2) positive semidefinite metric tensor
        lim     : half-width of the plotting window in each axis
        res     : grid resolution
        levels  : total number of contour levels to draw
        c_level : value of the FIRST contour (defines the reference ellipse)
        ax      : matplotlib Axes to plot on. If None, creates a new figure and axes.
        legend  : whether to show legend
        center  : 2D numpy array specifying the new origin of the plot
    """
    import numpy as np
    import matplotlib.pyplot as plt

    G = np.asarray(G, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    # Grid centered at 'center'
    x = np.linspace(-lim + center[0], lim + center[0], res)
    y = np.linspace(-lim + center[1], lim + center[1], res)
    X, Y = np.meshgrid(x, y)

    # Quadratic form: q(x,y) = (x-c)^T G (x-c)
    dX = X - center[0]
    dY = Y - center[1]
    pts = np.stack([dX, dY], axis=-1)
    q = np.einsum('...i,ij,...j->...', pts, G, pts)

    # Eigen-decomposition (ascending), reorder so eigvals[0] is largest
    eigvals, eigvecs = np.linalg.eigh(G)
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Ensure positive values for a valid ellipse (handle tiny/zero eigenvalues)
    eps = 1e-12
    eigvals = np.clip(eigvals, eps, None)

    # Semi-axes lengths of the ellipse v^T G v = c_level
    a = np.sqrt(c_level / eigvals[0])  # along top eigenvector
    b = np.sqrt(c_level / eigvals[1])  # along second eigenvector

    # Contour levels starting at c_level
    if levels < 2:
        level_values = [c_level]
    else:
        corner = np.array([lim, lim])
        approx_max = float(corner @ G @ corner)
        approx_max = max(approx_max, c_level * 10)
        level_values = np.geomspace(c_level, approx_max, levels)

    # Use provided ax or create new one
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True

    # Draw contours
    cs = ax.contour(X, Y, q, levels=level_values)
    if legend:
        ax.clabel(cs, inline=True, fontsize=8)

    if eigen_axis:
        # Draw eigen axes (translated by 'center')
        v1 = eigvecs[:, 0]  # top eigenvector
        v2 = eigvecs[:, 1]  # second eigenvector
        t = np.linspace(-1.0, 1.0, 200)

        # ±a*v1 and ±b*v2, shifted by 'center'
        xs1 = center[0] + t * a * v1[0]
        ys1 = center[1] + t * a * v1[1]
        xs2 = center[0] + t * b * v2[0]
        ys2 = center[1] + t * b * v2[1]
        ax.plot(xs1, ys1, color='red', linewidth=2, label='more sensitive (λ₁)')
        ax.plot(xs2, ys2, color='blue', linestyle='--', linewidth=2, label='less sensitive (λ₂)')

    ax.set_aspect('equal', adjustable='box')
    
    if legend:
        ax.legend(loc='upper right')

    if created_fig:
        plt.show()
