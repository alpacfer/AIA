# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import slgbuilder

# Fix for deprecated numpy aliases used by slgbuilder
np.bool = bool  # Ensure compatibility with boolean type
np.int = int    # Ensure compatibility with integer type

# Load the input image and convert its data type for processing
I = skimage.io.imread(r'C:\Users\Alejandro\Documents\GitHub\AIA\week7\data\peaks_image.png').astype(np.int32)

# Initialize a figure for displaying the original and processed images
fig, ax = plt.subplots(1,2)
# Display the original image
ax[0].imshow(I, cmap='gray')

# Initialize two graph objects for segmentation with no initial on-surface cost
layers = [slgbuilder.GraphObject(0*I), slgbuilder.GraphObject(0*I)]

# Set up the maxflow builder for handling the segmentation
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)  # Add the graph objects to the builder

# Add regional costs for segmentation:
# The first layer focuses on the transition from a dark to a bright region.
# The second layer focuses on the transition from a bright to a dark region.
helper.add_layered_region_cost(layers[0], I, 255-I)  # Cost for the region below the first layer
helper.add_layered_region_cost(layers[1], 255-I, I)  # Cost for the region above the second layer

# Add geometric constraints to guide the segmentation:
# Boundary cost to penalize deviations from the expected boundary location.
# Smoothness constraint to ensure that the detected regions are not too jagged.
# Containment constraint to maintain a minimum margin between the two layers.
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=1, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=1)

# Perform the cut to segment the image based on the defined costs and constraints
helper.solve()
# Retrieve the segmentation results for each layer
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
# Compute the lines that represent the boundaries of the segmented regions
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

# Visualization of the segmentation results:
# Display the original image
ax[1].imshow(I, cmap='gray')
# Overlay the segmentation lines on the image
for line in segmentation_lines:
    ax[1].plot(line, 'r')

# Show the figure with the original and segmented images
plt.show()