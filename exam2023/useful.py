import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
import numpy as np

def display_images(images, titles):
    """
    Displays 1, 2, 3, or 4 images with their respective titles and colorbars.
    
    Parameters:
    - images: List of images to display.
    - titles: List of titles corresponding to the images.
    """
    num_images = len(images)
    
    if num_images not in [1, 2, 3, 4]:
        raise ValueError("Number of images must be between 1 and 4.")
    
    if num_images == 1:
        fig, axs = plt.subplots(1, 1, figsize=(6, 5))
        axs = [axs]
    elif num_images == 2:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    elif num_images == 3:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    elif num_images == 4:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    axs = axs.flatten() if num_images > 1 else axs

    for i in range(num_images):
        im = axs[i].imshow(images[i], cmap='gray')
        axs[i].set_title(titles[i])
        axs[i].axis('off')
        fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

def display_points(data_list, colors=None, titles=None):
    """
    Displays points from a list of data arrays.

    Parameters:
    - data_list: List of numpy arrays containing the points.
    - colors: List of colors for each set of points. If None, default colors will be used.
    - titles: List of titles for each subplot. If None, no titles will be displayed.
    """
    num_data = len(data_list)
    
    if colors is None:
        colors = ['r.', 'g.', 'b.', 'c.', 'm.', 'y.', 'k.'] * (num_data // 7 + 1)
    
    if titles is None:
        titles = [''] * num_data

    fig, ax = plt.subplots(figsize=[10, 5])
    
    for i, data in enumerate(data_list):
        ax.plot(data[:, 0], data[:, 1], colors[i % len(colors)], label=titles[i])
    
    ax.set_aspect('equal')
    if any(titles):
        ax.legend()
    
    plt.show()

def compute_transformation(points_p, points_q):
    """
    Computes the optimal scale, translation, and rotation that aligns points_p to points_q.

    Parameters:
    - points_p: A numpy array of shape (n, 2) representing the source points.
    - points_q: A numpy array of shape (n, 2) representing the destination points.

    Returns:
    - s: Scale factor.
    - t: Translation vector.
    - theta: Rotation angle in degrees.
    """
    # Compute centroids
    centroid_p = np.mean(points_p, axis=0)
    centroid_q = np.mean(points_q, axis=0)

    # Center the points
    centered_p = points_p - centroid_p
    centered_q = points_q - centroid_q

    # Compute covariance matrix
    H = np.dot(centered_p.T, centered_q)

    # Compute the Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (no reflection)
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the scale
    scale = np.sum(S) / np.sum(centered_p ** 2)

    # Compute the translation
    translation = centroid_q - scale * np.dot(centroid_p, R.T)

    # Compute the rotation angle in degrees
    theta = np.degrees(np.arctan2(R[1, 0], R[0, 0]))

    return scale, translation, theta

def sift_matching_translation(sift_1_coord, sift_2_coord, sift_1_desc, sift_2_desc):
    """
    Computes the translation vector and its length between two sets of SIFT features.
    
    Returns:
    - mean_translation: The mean translation vector as a numpy array of shape (2,).
    - translation_length: The length of the translation vector.
    - matched_distances: A list of distances between the SIFT feature from Image 1 and its closest match in Image 2.
    """

    coord_1 = sift_1_coord
    coord_2 = sift_2_coord
    desc_1 = sift_1_desc
    desc_2 = sift_2_desc

    # Create a list of translations and matched distances
    translations = []
    matched_distances = []

    # Compute distance between one descriptor in set a and all in set b
    for i, desc in enumerate(desc_1):
        d = ((desc - desc_2) ** 2).sum(axis=1)
        sorted_indices = np.argsort(d)
        
        # Apply ratio test as per Lowe's paper
        if d[sorted_indices[0]] / d[sorted_indices[1]] < 0.8:
            best_match_idx = sorted_indices[0]
            translations.append(coord_1[i] - coord_2[best_match_idx])
            dist = np.sqrt(((coord_1[i] - coord_2[best_match_idx]) ** 2).sum())
            matched_distances.append(dist)

    # Make array of translations and compute the mean
    if translations:
        translations = np.vstack(translations)
        mean_translation = translations.mean(axis=0)
        translation_length = np.linalg.norm(mean_translation)
    else:
        mean_translation = np.array([0, 0])
        translation_length = 0

    return mean_translation, translation_length, matched_distances


def compute_rotation_from_covariance(C):
    """
    Computes the rotation matrix and angle from a given covariance matrix.

    Parameters:
    C (numpy.ndarray): 2x2 covariance matrix

    Returns:
    R (numpy.ndarray): 2x2 rotation matrix
    angle (float): rotation angle in degrees
    """
    # Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(C)
    
    # Compute the rotation matrix
    R = np.dot(U, Vt)
    
    # Ensure a proper rotation matrix (not a reflection)
    if np.linalg.det(R) < 0:
        Vt[1,:] *= -1
        R = np.dot(U, Vt)
    
    # Extract the rotation angle in radians
    angle_radians = np.arctan2(R[1, 0], R[0, 0])
    
    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return R, angle_degrees