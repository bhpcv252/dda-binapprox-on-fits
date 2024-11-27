from astropy.io import fits
import numpy as np
from helper import running_stats

def median_bins_fits(imgs, B):
    mean, std = running_stats(imgs)
    minval = mean - std
    maxval = mean + std

    # Initialize count and bins as 2D arrays of the same shape as mean
    count = np.zeros(mean.shape, dtype=int)
    bins = np.zeros(mean.shape + (B,), dtype=int)  # Extend shape for bins
    bin_width = (2 * std) / B

    for img in imgs:  # Iterate through each image
        hdulist = fits.open(img)
        data = hdulist[0].data
        for i in range(mean.shape[0]):  # Iterate over rows
            for j in range(mean.shape[1]):  # Iterate over columns
                pixel_value = data[i, j]
                if pixel_value < minval[i, j]:
                    count[i, j] += 1
                elif pixel_value < maxval[i, j]:
                    index = int((pixel_value - minval[i, j]) / bin_width[i, j])
                    bins[i, j, index] += 1

    return mean, std, count, bins


def median_approx_fits(imgs, B):
    mean, std, count, bins = median_bins_fits(imgs, B)
    N = len(imgs)  # Total number of images
    mid_point = (N + 1) / 2
    minval = mean - std
    bin_width = (2 * std) / B

    # Initialize a 2D array for the median
    median = np.zeros(mean.shape)

    # Iterate over each pixel location
    for i in range(mean.shape[0]):  # Rows
        for j in range(mean.shape[1]):  # Columns
            cumulative_count = count[i, j]
            for k, bin_count in enumerate(bins[i, j]):  # Iterate through bins
                cumulative_count += bin_count
                if cumulative_count >= mid_point:
                    median[i, j] = minval[i, j] + bin_width[i, j] * (k + 0.5)
                    break

    return median


if __name__ == '__main__':
    mean, std, left_bin, bins = median_bins_fits(['image{}.fits'.format(str(i)) for i in range(11)], 4)
    print(mean[100, 100], std[100, 100], left_bin[100, 100], bins[100, 100, :])
    median = median_approx_fits(['image{}.fits'.format(str(i)) for i in range(11)], 4)
    print(median[100, 100])
