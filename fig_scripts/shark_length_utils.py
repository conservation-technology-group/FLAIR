import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist, euclidean
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from skimage import measure

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([1, 0, 0, 0.3])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def get_skeleton_length_and_path(skeleton):
    points = np.array(np.where(skeleton)).T
    if len(points) == 0:
        return 0, []
    distances = cdist(points, points)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    start_point = points[i]
    end_point = points[j]
    max_length = distances[i, j]
    n_points = len(points)
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                diff = np.abs(points[i] - points[j])
                if np.all(diff <= 2):
                    dist = np.sqrt(np.sum((points[i] - points[j])**2))
                    dist_matrix[i, j] = dist * (1.1 if np.any(diff == 2) else 1.0)
    graph = csr_matrix(dist_matrix)
    start_idx = np.where((points == start_point).all(axis=1))[0][0]
    end_idx = np.where((points == end_point).all(axis=1))[0][0]
    try:
        distances, predecessors = dijkstra(graph, directed=False, indices=[start_idx], return_predecessors=True)
        if predecessors[0, end_idx] == -9999:
            return max_length, [start_point, end_point]
        path = []
        current = end_idx
        while current != start_idx:
            path.append(points[current])
            current = predecessors[0, current]
        path.append(points[start_idx])
        path.reverse()
        path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i + 1])) for i in range(len(path) - 1))
        return path_length, path
    except:
        return max_length, [start_point, end_point]

def get_total_mask_length(mask):
    if len(mask.shape) == 3:
        mask = mask[0]
    skeleton = skeletonize(mask)
    skeleton_length, skeleton_path = get_skeleton_length_and_path(skeleton)
    if not skeleton_path:
        return 0, []
    skeleton_path = np.array(skeleton_path)
    mask_coords = np.array(np.where(mask)).T
    distances = cdist(mask_coords, mask_coords)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    true_endpoint1, true_endpoint2 = mask_coords[i], mask_coords[j]
    skeleton_endpoints = np.array([skeleton_path[0], skeleton_path[-1]])
    dist_to_start = cdist([true_endpoint1], skeleton_endpoints)[0]
    dist_to_end = cdist([true_endpoint2], skeleton_endpoints)[0]

    if dist_to_start[0] + dist_to_end[1] < dist_to_start[1] + dist_to_end[0]:
        extension1 = np.linalg.norm(true_endpoint1 - skeleton_path[0])
        extension2 = np.linalg.norm(true_endpoint2 - skeleton_path[-1])
        full_path = np.vstack([true_endpoint1, skeleton_path, true_endpoint2])
    else:
        extension1 = np.linalg.norm(true_endpoint1 - skeleton_path[-1])
        extension2 = np.linalg.norm(true_endpoint2 - skeleton_path[0])
        skeleton_path = np.flip(skeleton_path, axis=0)
        full_path = np.vstack([true_endpoint1, skeleton_path, true_endpoint2])
    total_length = skeleton_length + extension1 + extension2
    return total_length, full_path

def show_enhanced_skeleton_with_path(mask, ax):
    if len(mask.shape) == 3:
        mask = mask[0]
    total_length, full_path = get_total_mask_length(mask)
    contours = measure.find_contours(mask, 0.5)
    #for contour in contours:
    #    ax.plot(contour[:, 1], contour[:, 0], 'k-', linewidth=1, alpha=0.5)
    skeleton = skeletonize(mask)
    y_coords, x_coords = np.where(skeleton)
    #ax.scatter(x_coords, y_coords, c='red', s=1, alpha=0.5, label='Skeleton')
    ax.plot(full_path[:, 1], full_path[:, 0], c='blue', linewidth=1, alpha=1.0, label='Full path')
    #ax.scatter([full_path[0, 1], full_path[-1, 1]], [full_path[0, 0], full_path[-1, 0]], c='green', s=50, alpha=1.0, label='True endpoints')
    #ax.legend()
    return total_length

def calculate_centerline_length(file_path, cv2, np, skeletonize, euclidean):
    mask = cv2.imread(file_path)
    lower_purple = np.array([150, 30, 100])
    upper_purple = np.array([250, 70, 190])
    purple_mask = cv2.inRange(mask, lower_purple, upper_purple)
    skeleton = skeletonize(purple_mask // 255).astype(np.uint8)
    points = np.column_stack(np.where(skeleton > 0))
    sorted_points = sorted(points, key=lambda x: (x[0], x[1]))
    length = sum(euclidean(sorted_points[i], sorted_points[i + 1]) for i in range(len(sorted_points) - 1)) if len(sorted_points) > 1 else 0
    return length
