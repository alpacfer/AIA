# Import necessary libraries
import numpy as np 
import matplotlib.pyplot as plt
import skimage.io 
import slgbuilder

# Fix for deprecated aliases in numpy still used in slgbuilder
np.bool = bool
np.int = int

# --- Step 1: Read and Display the Input Image ---
# Read the input image and convert to a 32-bit integer format
I = skimage.io.imread(r'C:\Users\Alejandro\Documents\GitHub\AIA\week7\data\layers_A.png').astype(np.int32)

# Initialize a figure for plotting
fig, ax = plt.subplots(1,4)
# Display the input image
ax[0].imshow(I, cmap='gray')
ax[0].set_title('input image')

# --- Step 2: Detect One Line with Specific Smoothness (Delta = 3) ---
# Define the smoothness constraint
delta = 3

# Initialize the graph object and the maxflow builder for segmentation
layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()  # Adds boundary cost to the graph
helper.add_layered_smoothness(delta=delta, wrap=False)  # Adds smoothness constraint

# Solve the segmentation problem
helper.solve()
# Determine which segments belong to the layer
segmentation = helper.what_segments(layer)
# Calculate the segmentation line
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

# Display the segmentation line on the image
ax[1].imshow(I, cmap='gray')
ax[1].plot(segmentation_line, 'r')
ax[1].set_title(f'delta = {delta}')

# --- Step 3: Detect One Line with Increased Smoothness (Delta = 1) ---
# Repeat the process with a smaller delta for a smoother line
delta = 1

layer = slgbuilder.GraphObject(I)
helper = slgbuilder.MaxflowBuilder()
helper.add_object(layer)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)

helper.solve()
segmentation = helper.what_segments(layer)
segmentation_line = segmentation.shape[0] - np.argmax(segmentation[::-1,:], axis=0) - 1

# Display the smoother segmentation line
ax[2].imshow(I, cmap='gray')
ax[2].plot(segmentation_line, 'r')
ax[2].set_title(f'delta = {delta}')

# --- Step 4: Detect Two Lines with a Minimum Margin ---
# Define two layers to detect and the smoothness constraint
layers = [slgbuilder.GraphObject(I), slgbuilder.GraphObject(I)]
delta = 3

helper = slgbuilder.MaxflowBuilder()
helper.add_objects(layers)
helper.add_layered_boundary_cost()
helper.add_layered_smoothness(delta=delta, wrap=False)  
helper.add_layered_containment(layers[0], layers[1], min_margin=15)  # Ensure a minimum margin between layers

helper.solve()
segmentations = [helper.what_segments(l).astype(np.int32) for l in layers]
segmentation_lines = [s.shape[0] - np.argmax(s[::-1,:], axis=0) - 1 for s in segmentations]

# Display both segmentation lines with the specified margin
ax[3].imshow(I, cmap='gray')
for line in segmentation_lines:
    ax[3].plot(line, 'r')
ax[3].set_title('two dark lines')

# Show the plot
plt.show()
