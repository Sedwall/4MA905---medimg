import torch as pt
import numpy as np
import h5py
from pathlib import Path

"""
This script processes and saves the PCAM dataset in PyTorch format.
It loads data from HDF5 files, processes it (e.g., normalization), and saves
it as .pt files for efficient loading in PyTorch.

Adjust the TRAIN, VALID, and TEST flags to choose which part of the dataset to process.
Ensure the dataset is downloaded and placed in the specified directory structure.

Directory structure:
DATASET_FOLDER: Path to the folder containing the original HDF5 files.
PROCESSED_DATA_FOLDER: Path to the folder where processed .pt files will be saved.

If a .pt files already exist, the script will skip processing for that data.
"""
# Choose which data to load
TRAIN = True
VALID = True
TEST = True

CHUNK = 10000 #Number of samples to process in chunk

# Set up directories
# DIR = Path(__file__).parent.joinpath("dataset_test")
# DATASET_FOLDER = Path(DIR.joinpath("./pcam/"))
# PROCESSED_DATA_FOLDER = Path(DIR.joinpath("./pcam_pt/"))

DIR = Path(__file__).parent
DATASET_FOLDER = DIR.parent.parent / "dataset" / "pcam"
PROCESSED_DATA_FOLDER = DIR.parent.parent / "dataset" / "pcam_pt"


class DataProcessing:
    """Data processing class for PCAM dataset
    this class handles loading, processing, and saving the data.

    The only method you might want to modify is `process_data`.
    This method is where you can add your own data processing logic.
    """
    def __init__(self):
        self.train_mean = None
        self.train_std = None

    def process_data(self, data: pt.Tensor) -> pt.Tensor:
        """Data processing function
        data shape: (N, 96, 96, 3)
        """
        if data.max() > 1.5:
            data = data.float() / 255.0  # Normalize to [0, 1]
        # Normalize using training data statistics
        data = (data - self.train_mean) / self.train_std
        return data
    
    def fit(self, train_data: pt.Tensor):
        """Compute mean and std from training data"""
        if train_data.max() > 1.5:
            train_data = train_data.float() / 255.0  # Normalize to [0, 1]
        # Compute mean and std
        self.train_mean = train_data.mean(dim=(0,1,2))
        self.train_std  = train_data.std(dim=(0,1,2), unbiased=False).clamp(min=1e-8)

    def load_data(self, file_path):
        print(f"Loading data from {str(file_path).split("\\")[-1]}...")
        with h5py.File(file_path, 'r') as f:
            key = str(f.keys())[-4]
            data = pt.tensor(np.array(f[key]), dtype=pt.float32)
        return data


    def save_data(self, data, file_path):
        if not PROCESSED_DATA_FOLDER.exists():
            PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        print(f"Save data to {str(file_path).split("\\")[-1]}...")
        pt.save(data, file_path)


    def handle_data(self, path):
        # Check if processed file already exists
        if (PROCESSED_DATA_FOLDER / self.find_file(path)).exists():
            print(f"{self.find_file(path)} already exists. Skipping processing.\n")
            return None
        
        # Load data
        data = self.load_data(DATASET_FOLDER / path)
        # Process data
        if "train_x" in path:
            self.fit(data)
            data = self.process_data(data)
        elif "valid_x" in path or "test_x" in path:
            if self.train_mean is None or self.train_std is None:
                raise ValueError("Train data must be processed before validation or test data.")
            data = self.process_data(data)
        self.save_data(data, PROCESSED_DATA_FOLDER / self.find_file(path))
        del data
        print("Data processed and saved.\n")


    def find_file(self, path):
        for name in ["train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"]:
            if name in path:
                return name + ".pt"


# Start processing
print("Loading data from...")
print("Dataset folder:", DATASET_FOLDER)

processor = DataProcessing()

# Load training data
if TRAIN:
    processor.handle_data('camelyonpatch_level_2_split_train_x.h5')
    processor.handle_data('camelyonpatch_level_2_split_train_y.h5')
# Load validation data
if VALID:
    processor.handle_data('camelyonpatch_level_2_split_valid_x.h5')
    processor.handle_data('camelyonpatch_level_2_split_valid_y.h5')
# Load test data
if TEST:
    processor.handle_data('camelyonpatch_level_2_split_test_x.h5')
    processor.handle_data('camelyonpatch_level_2_split_test_y.h5')

if all(not flag for flag in [TRAIN, VALID, TEST]):
    print("No data type selected. Please set TRAIN, VALID, or TEST to True.")


