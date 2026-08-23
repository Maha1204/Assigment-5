import os
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN


# Displays 3D point cloud interactively.

def show_cloud(points_plt, title="3D Point Cloud"):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_plt[:, 0], points_plt[:, 1], points_plt[:, 2], s=0.05, c=points_plt[:, 2], cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.show()

# Displays 2D scatter plot.

def show_scatter(x, y, title="2D Scatter"):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=1)
    plt.title(title)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()


# Task 1: Ground Level Detection

def get_ground_level(pcd, num_bins=100, dataset_name="dataset"):
    """
    Finds the ground level using a height histogram (np.histogram).
    The ground represents the peak/mode in Z distribution.
    """
    z_coords = pcd[:, 2]
    counts, bin_edges = np.histogram(z_coords, bins=num_bins)
    
    max_bin_idx = np.argmax(counts)
    ground_level = float(bin_edges[max_bin_idx + 1])
    
    # Plotting histogram

    plt.figure(figsize=(8, 5))
    plt.hist(z_coords, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(ground_level, color='red', linestyle='--', linewidth=2, label=f'Ground: {ground_level:.2f} m')
    plt.title(f'Height (Z) Histogram - {dataset_name}', fontsize=14)
    plt.xlabel('Z elevation (m)', fontsize=12)
    plt.ylabel('Point Count', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'images/{dataset_name}_task1_histogram.png')
    plt.show()
    
    return ground_level


# Task 2: Elbow Method & DBSCAN Clustering

def find_optimal_eps(points, min_samples=5, dataset_name="dataset"):
    
    tree = KDTree(points)
    distances, _ = tree.query(points, k=min_samples)
    k_distances = np.sort(distances[:, min_samples - 1])
    
    n_points = len(k_distances)
    coords = np.vstack((np.arange(n_points), k_distances)).T
    first_pt = coords[0]
    chord_vec = coords[-1] - coords[0]
    chord_unit_vec = chord_vec / np.linalg.norm(chord_vec)
    
    pt_vecs = coords - first_pt
    projections = np.sum(pt_vecs * chord_unit_vec, axis=1)
    dist_to_chord = np.linalg.norm(pt_vecs - np.outer(projections, chord_unit_vec), axis=1)
    
    elbow_idx = np.argmax(dist_to_chord)
    optimal_eps = float(k_distances[elbow_idx])
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances, label=f'{min_samples}-NN Distance curve', color='navy')
    plt.scatter(elbow_idx, optimal_eps, color='crimson', s=50, zorder=5, 
                label=f'Optimal $\epsilon$ = {optimal_eps:.2f}')
    plt.title(f'k-NN Distance Plot (Elbow Method) - {dataset_name}', fontsize=14)
    plt.xlabel('Points Sorted by Distance', fontsize=12)
    plt.ylabel(f'{min_samples}-NN Distance ($\epsilon$)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'images/{dataset_name}_task2_elbow.png')
    plt.show()
    
    return optimal_eps

# Executes DBSCAN and plots the 2D cluster results.

def run_dbscan(points, eps, min_samples=5, dataset_name="dataset"):
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='tab20', s=2)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f'DBSCAN: {n_clusters} Clusters ($\epsilon$={eps:.2f}) - {dataset_name}', fontsize=14)
    plt.xlabel('X axis (m)', fontsize=12)
    plt.ylabel('Y axis (m)', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'images/{dataset_name}_task2_clusters.png')
    plt.show()
    
    return clustering


# Task 3: Largest Catenary Cluster Detection

def find_catenary_cluster(points, clustering, dataset_name="dataset"):
    
    labels = clustering.labels_
    unique_labels = set(labels) - {-1} 
    
    max_area = 0.0
    catenary_id = None
    bounds = {}
    
    for c_id in unique_labels:
        cluster_pts = points[labels == c_id]
        min_x, max_x = np.min(cluster_pts[:, 0]), np.max(cluster_pts[:, 0])
        min_y, max_y = np.min(cluster_pts[:, 1]), np.max(cluster_pts[:, 1])
        
        area = (max_x - min_x) * (max_y - min_y)
        if area > max_area:
            max_area = area
            catenary_id = c_id
            bounds = {
                'min_x': float(min_x), 'max_x': float(max_x),
                'min_y': float(min_y), 'max_y': float(max_y),
                'area': float(area)
            }
            
    catenary_pts = points[labels == catenary_id]
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(catenary_pts[:, 0], catenary_pts[:, 1], c=catenary_pts[:, 2], cmap='plasma', s=3)
    plt.colorbar(sc, label='Z Height (m)')
    plt.title(f'Catenary Cluster #{catenary_id} (Area: {bounds["area"]:.2f} m²) - {dataset_name}', fontsize=14)
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Y (m)', fontsize=12)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    plt.savefig(f'images/{dataset_name}_task3_catenary.png')
    plt.show()
    
    return catenary_id, bounds


# Main Execution Pipeline

def process_dataset(filename):
    dataset_name = os.path.splitext(os.path.basename(filename))[0]
    print(f"\n==================== Processing {dataset_name} ====================")
    
    pcd = np.load(filename)
    print(f"Total points loaded: {pcd.shape[0]}")
    
    # Task 1
    ground_level = get_ground_level(pcd, dataset_name=dataset_name)
    pcd_above_ground = pcd[pcd[:, 2] > ground_level]
    print(f"[Task 1] Ground Level: {ground_level:.2f} m")
    print(f"Points above ground: {pcd_above_ground.shape[0]}")
    
    # Task 2
    optimal_eps = find_optimal_eps(pcd_above_ground, min_samples=5, dataset_name=dataset_name)
    clustering = run_dbscan(pcd_above_ground, eps=optimal_eps, min_samples=5, dataset_name=dataset_name)
    print(f"[Task 2] Optimal Epsilon: {optimal_eps:.2f}")
    
    # Task 3
    catenary_id, bounds = find_catenary_cluster(pcd_above_ground, clustering, dataset_name=dataset_name)
    print(f"[Task 3] Catenary Cluster ID: {catenary_id}")
    print(f"         min(x) = {bounds['min_x']:.2f}, max(x) = {bounds['max_x']:.2f}")
    print(f"         min(y) = {bounds['min_y']:.2f}, max(y) = {bounds['max_y']:.2f}")
    print(f"         Area of the catenary cluster = {bounds['area']:.2f} m^2")
    
    return {
        'ground_level': ground_level,
        'optimal_eps': optimal_eps,
        'area': bounds['area']
    }

if __name__ == "__main__":
    os.makedirs('images', exist_ok=True)
    
    # Process both datasets
    res1 = process_dataset("dataset1.npy")
    res2 = process_dataset("dataset2.npy")
    
    # Summary 
    print(f"I have attempted Task 3 and the results for dataset2.npy are: Ground level = {res2['ground_level']:.2f} m, Optimal epsilon = {res2['optimal_eps']:.2f}, Area of the catenary cluster = {res2['area']:.2f} m^2")