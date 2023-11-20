import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage import img_as_ubyte
from skimage.color import rgb2gray
from skimage.exposure import histogram, cumulative_distribution
from scipy.stats import cauchy, logistic

cathedral = imread('cathedral.jpg')
plt.imshow(cathedral)
plt.title('Manila Cathedral')

fig, ax = plt.subplots(1,2, figsize=(15,5))
cathedral_gray = rgb2gray(cathedral)
ax[0].imshow(cathedral_gray, cmap='gray')
ax[0].set_title('Grayscale Image')
ax1 = ax[1]
ax2 = ax1.twinx()
freq_h, bins_h = histogram(cathedral_gray)
freq_c, bins_c = cumulative_distribution(cathedral_gray)
ax1.step(bins_h, freq_h*1.0/freq_h.sum(), c='b', label='PDF')
ax2.step(bins_c, freq_c, c='r',  label='CDF')
ax1.set_ylabel('PDF', color='b')
ax2.set_ylabel('CDF', color='r')
ax[1].set_xlabel('Intensity value')
ax[1].set_title('Histogram of Pixel Intensity');

image_intensity = img_as_ubyte(cathedral_gray)
freq, bins = cumulative_distribution(image_intensity)
target_bins = np.arange(255)
target_freq = np.linspace(0, 1, len(target_bins))
new_vals = np.interp(freq, target_freq, target_bins)
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].step(bins, freq, c='b', label='Actual CDF')
ax[0].plot(target_bins, target_freq, c='r', label='Target CDF')
ax[0].legend()
ax[0].set_title('Grayscale: Actual vs. '
                'Target Cumulative Distribution')
ax[1].imshow(new_vals[image_intensity].astype(np.uint8), 
             cmap='gray')
ax[1].set_title('Corrected Image in Grayscale');


def show_linear_cdf(image, channel, name, ax):
    image_intensity = img_as_ubyte(image[:,:,channel])
    freq, bins = cumulative_distribution(image_intensity)
    target_bins = np.arange(255)
    target_freq = np.linspace(0, 1, len(target_bins))
    ax.step(bins, freq, c='b', label='Actual CDF')
    ax.plot(target_bins, target_freq, c='r', label='Target CDF')
    ax.legend()
    ax.set_title('{} Channel: Actual vs. '
                 'Target Cumulative Distribution'.format(name))
def linear_distribution(image, channel):
    image_intensity = img_as_ubyte(image[:,:,channel])
    freq, bins = cumulative_distribution(image_intensity)
    target_bins = np.arange(255)
    target_freq = np.linspace(0, 1, len(target_bins))
    new_vals = np.interp(freq, target_freq, target_bins)
    return new_vals[image_intensity].astype(np.uint8)

fig, ax = plt.subplots(3,2, figsize=(12,14))
red_channel = linear_distribution(cathedral, 0)
green_channel = linear_distribution(cathedral, 1)
blue_channel = linear_distribution(cathedral, 2)
show_linear_cdf(cathedral, 0, 'Red', ax[0,0])
ax[0,1].imshow(red_channel, cmap='Reds')
ax[0,1].set_title('Corrected Image in Red Channel')
show_linear_cdf(cathedral, 1, 'Green', ax[1,0])
ax[1,1].imshow(green_channel, cmap='Greens')
ax[1,1].set_title('Corrected Image in Green Channel')
show_linear_cdf(cathedral, 2, 'Blue', ax[2,0])
ax[2,1].imshow(blue_channel, cmap='Blues')
ax[2,1].set_title('Corrected Image in Blue Channel')
                  
fig, ax = plt.subplots(1,2, figsize=(15,5))
ax[0].imshow(cathedral);
ax[0].set_title('Original Image')
ax[1].imshow(np.dstack([red_channel, green_channel, blue_channel]));
ax[1].set_title('Transformed Image');

def individual_channel(image, dist, channel):
    im_channel = img_as_ubyte(image[:,:,channel])
    freq, bins = cumulative_distribution(im_channel)
    new_vals = np.interp(freq, dist.cdf(np.arange(0,256)), 
                               np.arange(0,256))
    return new_vals[im_channel].astype(np.uint8)
def distribution(image, function, mean, std):
    dist = function(mean, std)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    
    image_intensity = img_as_ubyte(rgb2gray(image))
    freq, bins = cumulative_distribution(image_intensity)
    ax[0].step(bins, freq, c='b', label='Actual CDF')
    ax[0].plot(dist.cdf(np.arange(0,256)), 
               c='r', label='Target CDF')
    ax[0].legend()
    ax[0].set_title('Actual vs. Target Cumulative Distribution')
    
    red = individual_channel(image, dist, 0)
    green = individual_channel(image, dist, 1)
    blue = individual_channel(image, dist, 2)
    ax[1].imshow(np.dstack((red, green, blue)))
    ax[1].set_title('Transformed Image')
    return ax

    distribution(cathedral, cauchy, 90, 30);
    distribution(cathedral, logistic, 90, 30);
