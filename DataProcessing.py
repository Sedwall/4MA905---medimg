import torch as pt
import numpy as np
import h5py
from pathlib import Path
from skimage.feature import hog

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
TRAIN = False
VALID = False
TEST = True

# Set up directories
DIR = Path(__file__).parent.parent.parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam/"))
PROCESSED_DATA_FOLDER = Path(DIR.joinpath("./pcam_HOG/"))



class DataProcessing:
    """Data processing class for PCAM dataset
    this class handles loading, processing, and saving the data.

    The only method you might want to modify is `process_data`.
    This method is where you can add your own data processing logic.
    """


    def process_data(self, data: pt.Tensor) -> pt.Tensor:
        """Data processing function
        data shape: (N, 3, 96, 96)
        
        Example processing 1: Normalize data
        data = (data - pt.mean(data)) / pt.std(data)

        Example processing 2: Data augmentation (e.g., random horizontal flip)
        if pt.rand(1) > 0.5:
            data = pt.flip(data, dims=[3])  # Horizontal flip
        return data

        Example processing 3: Image cropping
        data = data[:, :, 32:64, 32:64]  # Crop
        """
        # data= data[:, 32:64, 32:64, :]  # Crop
        result = []
        for i in range(data.shape[0]):
            fd = hog(
                data[i].numpy().astype(int),
                orientations=8,
                pixels_per_cell=(16, 16),
                cells_per_block=(5, 5),
                visualize=False,
                channel_axis=-1,
            )
            result.append(fd)
        return pt.tensor(result)
    

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
        if (PROCESSED_DATA_FOLDER / self.find_file(path)).exists():
            print(f"{self.find_file(path)} already exists. Skipping processing.\n")
            return None
        data = self.load_data(DATASET_FOLDER / path)
        if "x" in path:
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


