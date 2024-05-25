# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

# Fix for deprecated numpy aliases used by slgbuilder for compatibility
np.bool = bool
np.int = int

# --- Step 1: Load and Display the Input Image ---
I = skimage.io.imread(r'C:\Users\Alejandro\Documents\GitHub\AIA\week7\data\rammed-earth-layers-limestone.jpg').astype(np.int32)

# Initialize figure for displaying results
fig, ax = plt.subplots(1, 5, figsize=(20, 4))  # Adjusted for better visibility
ax[0].imshow(I, cmap='gray')
ax[0].set_title('Input Image')

# --- Step 2: Detect the Darkest Line ---
delta = 3  # Smoothness parameter

# Create graph object and configure segmentation settings
layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

# Perform segmentation to find the darkest line
helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1, :], axis=0) - 1

# Display the result with the darkest line highlighted
ax[1].imshow(I, cmap='gray')
ax[1].plot(segmentation_line, 'r')
ax[1].set_title('Darkest Line')

# --- Step 3: Detect Two Lines ---
layers = [slgbuilder.GraphObject(I), slgbuilder.GraphObject(I)]  # Two layers for two lines
delta = 3  # Smoothness parameter

# Configure segmentation for detecting two lines with a minimum margin
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)
helper.add_layered_containment(layers[0], layers[1], min_margin=15)

# Perform segmentation to find the two lines
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1, :], axis=0) - 1 for s in segmentations]

# Display the result with two lines highlighted
ax[2].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[2].plot(line, 'r')
ax[2].set_title('Two Lines')

# --- Corrected Step 4: Detect Transitions from Dark to Light Regions ---
# Initialize graph objects for segmentation with no initial on-surface cost
layers = [slgbuilder.GraphObject(0*I), slgbuilder.GraphObject(0*I)]

# Configure the maxflow builder for handling the segmentation
helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)  # Add the graph objects to the builder

# Setting up regional costs to detect the transition. If detecting only one transition,
# ensure that the configuration reflects both intended transitions accurately.
helper.add_layered_region_cost(layers[0], I, 255-I)  # Transition from dark to bright
helper.add_layered_region_cost(layers[1], 255-I, I)  # Transition from bright to dark

# Add geometric constraints to guide the segmentation:
helper.add_layered_boundary_cost()  # Boundary cost
helper.add_layered_smoothness(delta=1, wrap=False)  # Smoothness constraint
helper.add_layered_containment(layers[0], layers[1], min_margin=1)  # Minimum margin between layers

# Perform the segmentation to detect transitions
helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1, :], axis=0) - 1 for s in segmentations]

# Displaying the results for transitions on the correct subplot
ax[3].imshow(I, cmap='gray')  # Ensure we're using the correct subplot index
for line in segmentation_lines:
    ax[3].plot(line, 'r')
ax[3].set_title('Transitions Dark to Light')


plt.tight_layout()
plt.show()

