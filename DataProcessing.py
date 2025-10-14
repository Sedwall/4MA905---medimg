import torch as pt
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
from skimage.feature import hog
from PCAMdataset import PCAMdataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

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
VALID = False
TEST = True

CHUNK_SIZE = 1000

# Set up directories
DIR = Path(__file__).parent.parent.parent.joinpath("dataset")
DATASET_FOLDER = Path(DIR.joinpath("./pcam/"))
PROCESSED_DATA_FOLDER = Path(DIR.joinpath("./pcam_HOG_h5/"))



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

        data = data.permute(0,2,3,1) # (batch, chanels, xdim, ydim) -> (batch, xdim, ydim, chanels)

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
        return pt.tensor(np.array(result))
    

    def load_data(self, file_path):
        print(f"Loading data from {str(file_path).split("\\")[-1]}...")
        with h5py.File(file_path, 'r') as f:
            key = str(f.keys())[-4]
            data = pt.tensor(np.array(f[key]), dtype=pt.float32)
        return data

    def create_dataset(self, file_path: Path, shape):
        # Create a new file and an extendable dataset
        with h5py.File(file_path, "w") as f:
            dset = f.create_dataset(
                "data",
                shape=(0, *shape),        # start with zero rows, 1000 columns
                maxshape=(None, *shape),  # allow unlimited rows
                chunks=(CHUNK_SIZE, *shape),    # enable efficient appends
                compression="gzip"
            )
        assert file_path.exists()


    def save_data(self, data: pt.Tensor, file_path):
        shape = data.shape[1:] # remove batch dim
        if not file_path.exists():
            self.create_dataset(file_path, shape)

        ## Append
        with h5py.File(file_path, "a") as f:  # reopen in append mode
            dset = f["data"]
            old_size = dset.shape[0]
            new_size = old_size + data.shape[0] # resizeing to add new chunk

            # Resize the dataset to accommodate new data
            dset.resize((new_size, *shape))

            # Write new data at the end
            dset[old_size:new_size, :] = data



    def handle_data(self, path):
        if (PROCESSED_DATA_FOLDER / self.find_file(path)).exists():
            print(f"{self.find_file(path)} already exists. Skipping processing.\n")
            return None
        

        y_path = path[:-4] + "y.h5"
    
        # Create datasets
        train_data = PCAMdataset(
            x_path=DATASET_FOLDER / path,
            y_path=DATASET_FOLDER / y_path,
            # transform=train_tf
        )

        train_dl = DataLoader(train_data, batch_size=CHUNK_SIZE, shuffle=False,
                            num_workers=1, pin_memory=True, persistent_workers=True)

        
        for data_batch, _, _ in tqdm(train_dl):
            data_batch = self.process_data(data_batch)
            self.save_data(data_batch, PROCESSED_DATA_FOLDER / self.find_file(path))
            del data_batch
        print("Data processed and saved.\n")


    def find_file(self, path):
        for name in ["train_x", "train_y", "valid_x", "valid_y", "test_x", "test_y"]:
            if name in path:
                return name + ".h5"


# Start processing
print("Loading data from...")
print("Dataset folder:", DATASET_FOLDER)

if not PROCESSED_DATA_FOLDER.exists():
    PROCESSED_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

processor = DataProcessing()

# Load training data
if TRAIN:
    processor.handle_data('camelyonpatch_level_2_split_train_x.h5')
# Load validation data
if VALID:
    processor.handle_data('camelyonpatch_level_2_split_valid_x.h5')
# Load test data
if TEST:
    processor.handle_data('camelyonpatch_level_2_split_test_x.h5')

if all(not flag for flag in [TRAIN, VALID, TEST]):
    print("No data type selected. Please set TRAIN, VALID, or TEST to True.")


