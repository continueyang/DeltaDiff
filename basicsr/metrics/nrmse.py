import numpy as np
from skimage.io import imread
from sklearn.metrics import mean_squared_error
from glob import glob


def nrmse(generated_folder, original_folder):
    # Get all image paths from folders
    generated_images = glob(generated_folder + '/*')
    original_images = glob(original_folder + '/*')

    # Check folders contain same number of images
    if len(generated_images) != len(original_images):
        raise ValueError("Image folders must contain the same number of images!")

    total_nrmse = 0

    # Iterate through corresponding images
    for gen_image, orig_image in zip(generated_images, original_images):
        # Read images
        gen_img = imread(gen_image)
        orig_img = imread(orig_image)

        # Calculate NRMSE for each pair of images
        nrmse = calculate_nrmse(gen_img, orig_img)

        # Add to total
        total_nrmse += nrmse

    # Calculate average NRMSE
    avg_nrmse = total_nrmse / len(generated_images)

    return avg_nrmse


def calculate_nrmse(generated_image, original_image):
    # Flatten the images
    gen_flat = generated_image.flatten()
    orig_flat = original_image.flatten()

    # Calculate MSE
    mse = mean_squared_error(gen_flat, orig_flat)

    # Calculate RMSE
    rmse = np.sqrt(mse)

    # Calculate mean value of original image
    orig_mean = np.mean(orig_flat)

    # Calculate NRMSE
    nrmse = rmse / orig_mean

    return nrmse