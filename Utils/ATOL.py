import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from gudhi.representations.vector_methods import Atol
from gudhi import CubicalComplex
from Utils.PCAMdataset import get_feature_dataset
from torchvision import transforms as T


def get_feature_extractor(path_dir:Path):
    """ Extract ATOL features from a batch of images.
    
    Args:
        images (np.ndarray): Array of shape (N, C, H, W) representing the images.
        n_clusters (int): Number of clusters for KMeans.
        atol_params (dict): Parameters for the Atol transformer.
    """
    mean = [0.7008, 0.5384, 0.6916]
    std = [0.2350, 0.2774, 0.2129]
    # Define transforms
    train_tf = T.Compose([
        T.Normalize(mean, std), # standardize
        T.Grayscale(num_output_channels=1), # convert to grayscale
    ])

    data, _, _ = get_feature_dataset(
        x_path=path_dir / 'pcam' / 'camelyonpatch_level_2_split_valid_x.h5',
        y_path=path_dir / 'pcam' /'camelyonpatch_level_2_split_valid_y.h5',
        transform=train_tf
        )
    
    diagrams = []
    for i in tqdm(range(data.shape[0]), desc="Computing persistence diagrams"):
        diag = image_to_persistence(data[i], homology_coeff=2, min_persistence=0.0)
        diagrams.append(diag)
    
    atol_vectoriser = Atol(quantiser=KMeans(n_clusters=40, random_state=202006, n_init=10))
    atol_vectoriser.fit(X=diagrams).centers

    ####### Feature Extraction Function #######
    def feature_transform(img:np.ndarray, atol_vectoriser=atol_vectoriser) -> np.ndarray:
        """ Example feature transformation: (C, H, W) """
        feature_vector = atol_vectoriser.transform(image_to_persistence(img, homology_coeff=2, min_persistence=0.0))
        return feature_vector

    return feature_transform

def image_to_persistence(img: np.ndarray,
                         homology_coeff: int = 2,
                         min_persistence: float = 0.0) -> np.ndarray:
    """
    Convert 2D grayscale image `img` (shape [H, W]) to a persistence diagram.
    Returns an (N, 2) array of (birth, death) pairs (for all homology dimensions).
    """
    # Optionally invert the intensities so that bright features become low
    
    img2 = img.copy()
    H, W = img2.shape
    # Flatten top-dimensional cells
    flat = img2.flatten()
    # Build cubical complex
    cc = CubicalComplex(dimensions=[H, W], top_dimensional_cells=flat)
    # Compute persistence
    cc.compute_persistence(homology_coeff_field=homology_coeff,
                           min_persistence=min_persistence)
    # Get diagram
    diag = cc.persistence()
    return np.array(diag)