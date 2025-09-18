import torch as pt
import numpy as np
import h5py
from pathlib import Path



from TDA import calculate_betti_numbers, calculate_persistence
from tqdm import tqdm
import concurrent.futures

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
VALID = True
TEST = True

# Set up directories
DIR = Path(__file__).parent.parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam/"))
PROCESSED_DATA_FOLDER = Path(DIR.joinpath("./pcam_pt_TDA/"))


def get_Betti_numbers(img):
    """Calculate Betti numbers for a image."""
    img_np = img.numpy()
    persistence = calculate_persistence(img_np)
    betti_numbers = calculate_betti_numbers(persistence)
    return pt.tensor(betti_numbers, dtype=pt.float32)

class DataProcessing:
    """Data processing class for PCAM dataset
    this class handles loading, processing, and saving the data.

    The only method you might want to modify is `process_data`.
    This method is where you can add your own data processing logic.
    """


    def process_data(self, data: pt.Tensor) -> pt.Tensor:
        """Data processing function
        data shape: (N, 3, 96, 96)

        """
        
        r, g, b = data[:, :, :, 0], data[:, :, :, 1], data[:, :, :, 2]
        data = 0.2989 * r + 0.5870 * g + 0.1140 * b
       
        data = data.to(pt.uint8)  # Convert to integer for LBP processing
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
            results = list(tqdm(executor.map(get_Betti_numbers, data), total=data.shape[0], desc="Processing data"))
        del data
        processed_data = pt.stack(results)
        return processed_data
    

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

if __name__ == "__main__":
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


