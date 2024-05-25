import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.draw
import simple_snake_new as es  # Assuming this is a module you've defined

# Adjusted process_frame function to return additional info for labeling
def process_frame(I, initial_snake=None, num_iterations=10):
    N = 100
    radius = min(I.shape)/3
    center = np.array(I.shape)/1.7
    step_size = 0.0001
    alpha = 0.001
    beta = 0.001
    
    if initial_snake is None:
        snake = es.make_circular_snake(N, center, radius)
    else:
        snake = initial_snake

    B = es.regularization_matrix(N, alpha, beta)
    m_in, m_out = 0, 0  # Initialize mean intensities

    for i in range(num_iterations):
        mask = skimage.draw.polygon2mask(I.shape, snake)
        m_in = np.mean(I[mask])
        m_out = np.mean(I[~mask])
        normals = es.get_normals(snake)
        val = I[snake[:, 0].astype(int), snake[:, 1].astype(int)]
        f_ext = (m_in - m_out) * (2 * val - m_in - m_out)
        displacement = step_size * f_ext[:, None] * normals
        snake += displacement
        snake = B @ snake
        snake = es.distribute_points(snake)
        snake = es.remove_intersections(snake)
        snake = es.keep_snake_inside(snake, I.shape)
        
    return snake, mask, m_in, m_out

video_path = 'C:/Users/Alejandro/Documents/GitHub/AIA/week6/data/crawling_amoeba.mov'
cap = cv2.VideoCapture(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
num_iterations = 50  # Define the number of iterations for processing each frame

for i in range(frame_count):
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if i == 0:
        snake, mask, m_in, m_out = process_frame(frame, num_iterations=num_iterations)
    else:
        snake, mask, m_in, m_out = process_frame(frame, initial_snake=snake, num_iterations=num_iterations)
    
    # Optionally display or save the result for selected frames
    if i % 50 == 0:  # For example, every 50 frames
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(frame, cmap='gray')
        plt.plot(snake[:, 1], snake[:, 0], 'b.-')
        plt.text(10, 30, f'Frame: {i+1}\nIterations: {num_iterations}\nm_in: {int(m_in)}\nm_out: {int(m_out)}', color='yellow')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray')
        plt.show()

cap.release()
