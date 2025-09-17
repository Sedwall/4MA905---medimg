from skimage.feature import local_binary_pattern
import numpy as np



class LBP:
    def __init__(self):
        self.n_bins = None

    def LBP_feature_extraction(self, image):
        radius = 4
        n_points = 8 * radius
        METHOD = 'uniform'

        # Ensure image is a numpy array of numbers
        image = np.asarray(image)

        lbp = local_binary_pattern(image, n_points, radius, METHOD)
        if self.n_bins is None:
            self.n_bins = int(lbp.max() + 1)

        # Convert LBP to a fixed-sized feature vector using histogram
        hist, _ = np.histogram(lbp.ravel(), bins=self.n_bins, range=(0, self.n_bins), density=True)
        feature_vector = hist  # This is your fixed-sized feature vector
        return feature_vector



if __name__ == "__main__":
    
    from skimage import data
    image = data.astronaut().astype('uint8').mean(axis=2)

    feature_vector = LBP_feature_extraction(image)
    print("Image size:", image.shape)
    print("LBP Feature Vector size:", feature_vector.shape)