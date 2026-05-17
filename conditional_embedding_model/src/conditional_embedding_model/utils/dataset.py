import os
import numpy as np
import pybullet as p
import itertools
import matplotlib.pyplot as plt

class Dataset:
    def __init__(self, dataset_id, scene, object, table, obstacle_map, obj_footprint, grasping_approach, num_agents) -> None:
        self.dataset_id = dataset_id # Dataset id
        self.scene = scene # Save the scene configuration
        self.object = object # Save the object configuration
        self.table = table # Save the table configuration
        self.map = np.fliplr(np.flipud(obstacle_map)) # 2D array
        self.object_footprint = obj_footprint
        self.grasping_approach = grasping_approach
        self.combinations = self.generate_combinations(num_agents) # Generate all possible combinations [0,0], [0,1], [1,1], [1,0] ..
    
    def get_approach(self, combination:list):
        # Get the grasping approach given a combination
        actions = []
        for index in combination:
            approach = self.grasping_approach[index]
            if approach.sum() == 0:
                continue
            actions.append(self.grasping_approach[index])
        if len(actions) != len(combination) or combination[0] == combination[1]:
            return None
        
        return actions

    def set_value(self, combination, value):
        """
        combination: e.g (1, 34, -1)
        value: 0 or 1 or -1
        """
        # Set the value of the combination e.g combination [1,0] value 1
        indices = np.where((self.combinations[:, 0] == combination[0]) & (self.combinations[:, 1] == combination[1]))[0]
        if len(indices) > 0:
            # Update the value at the found index
            self.combinations[indices[0], -1] = value
    
    def generate_combinations(self, num_combinations):
        # Generate all possible combinations of the dataset
        values = range(self.grasping_approach.shape[0])
        combinations = np.array(list(itertools.combinations(values, num_combinations)))
        
        value_column = np.ones((combinations.shape[0], 1), dtype=np.int8) * -1 # Mean unsigned
        combinations = np.hstack((combinations, value_column))
        return combinations
    
    def generate_dict(self):
        # Function to generate dictionary with the dataset
        dataset_dict = {
            'scene': self.scene,
            'obstacle_map': self.map,
            'object_footprint': self.object_footprint,
            'grasping_approach': self.grasping_approach,
            'combinations': self.combinations
        }
        return dataset_dict

    def save_visualization(self, save_path='./dataset/images', image_name='dataset_visualization.png', max_range=1.5):
            # Create a figure and axis for the plot
            fig, ax = plt.subplots(figsize=(8, 8))
            # map_array = np.fliplr(np.flipud(self.map))
            ax.imshow(self.map, cmap='gray', interpolation='none')  # Display the obstacle map

            # Height of the map for Y-axis inversion
            map_height = self.map.shape[0]
            map_width = self.map.shape[1]
            
            # Scaling factors (assuming self.map dimensions correspond to max_range in meters)
            scale_x = self.map.shape[1] / max_range
            scale_y = self.map.shape[0] / max_range

             # Function to convert metric coordinates to image coordinates
            def convert_coordinates(coord):
                x, y = coord
                return (map_width/2 - (x * scale_x), map_height/2 - (y * scale_y))

            # Plot the object footprint
            for corner in self.object_footprint:
                translated_corner = convert_coordinates(corner)
                ax.plot(translated_corner[1], translated_corner[0], 'bo') 

            # Plot the grasping approaches
            for approach in self.grasping_approach:
                px, py, gx, gy = approach
                translated_px_py = convert_coordinates((px, py))
                translated_gx_gy = convert_coordinates((gx, gy))
                ax.plot(translated_px_py[1], translated_px_py[0], 'ro')  # Red circles for position
                ax.plot(translated_gx_gy[1], translated_gx_gy[0], 'go')  # Green circles for grasping point

            # Set labels and title
            ax.set_xlabel('X-coordinate')
            ax.set_ylabel('Y-coordinate')
            ax.set_title('Dataset Visualization')

            # Save the plot as an image
            plt.savefig(os.path.join(save_path, image_name))

            # Close the plot to free up memory
            plt.close(fig)
    
    ########################################################################
    ############ Update 1.0
    def get_mask_grasping_approach(self):
        # Return a square matrix with the mask of the grasping approach
        # the mask will be 1 if the grasping approach is not empty
        mask = np.ones(self.grasping_approach.shape[0])
        for i, approach in enumerate(self.grasping_approach):
            if approach.sum() == 0.0:
                mask[i] = 0
        # now create a matrix with the mask
        mask_matrix = np.zeros((self.grasping_approach.shape[0], self.grasping_approach.shape[0]))
        # Fill the possible grasping approaches and links in the mask adjacency matrix
        for i in range(self.grasping_approach.shape[0]):
            for j in range(self.grasping_approach.shape[0]):
                if i != j:
                    mask_matrix[i, j] = mask[i] * mask[j]
        return mask_matrix
    
    def save_visualization_combinations(self, save_path='./dataset/', image_name='dataset_visualization.png', max_range=1.5):
        # Create a figure and axis for the plot
        fig, ax = plt.subplots(figsize=(8, 8))
        # map_array = np.fliplr(np.flipud(self.map))
        ax.imshow(self.map, cmap='gray', interpolation='none')  # Display the obstacle map

        # Height of the map for Y-axis inversion
        map_height = self.map.shape[0]
        map_width = self.map.shape[1]

        # Scaling factors (assuming self.map dimensions correspond to max_range in meters)
        scale_x = self.map.shape[1] / max_range
        scale_y = self.map.shape[0] / max_range

        # Function to convert metric coordinates to image coordinates
        def convert_coordinates(coord):
            x, y = coord
            return (map_width/2 - (x * scale_x), map_height/2 - (y * scale_y))

        # Plot the object footprint
        for corner in self.object_footprint:
            translated_corner = convert_coordinates(corner)
            ax.plot(translated_corner[1], translated_corner[0], 'bo')

        # Plot the grasping approaches
        for approach_idx, approach in enumerate(self.grasping_approach):
            px, py, gx, gy = approach
            translated_px_py = convert_coordinates((px, py))
            translated_gx_gy = convert_coordinates((gx, gy))
            ax.plot(translated_px_py[1], translated_px_py[0], 'ro')  # Red circles for position
            ax.plot(translated_gx_gy[1], translated_gx_gy[0], 'go')  # Green circles for grasping point

        success_combinations = self.combinations[np.where(self.combinations[:, -1] == 1)]
        colors = plt.cm.get_cmap('tab20', len(success_combinations))  # Choose a colormap with enough colors for all successful combinations
        # Plot the grasping approaches
        for i, combination in enumerate(success_combinations):
            approach1_idx, approach2_idx = combination[:2]
            approach1 = self.grasping_approach[approach1_idx]
            approach2 = self.grasping_approach[approach2_idx]

            # Plot the first approach
            px1, py1, gx1, gy1 = approach1
            translated_px_py1 = convert_coordinates((px1, py1))
            translated_gx_gy1 = convert_coordinates((gx1, gy1))
            ax.plot(translated_px_py1[1], translated_px_py1[0], 'ro',color=colors(i), markersize=10)  # Red circles for position
            ax.plot(translated_gx_gy1[1], translated_gx_gy1[0], 'go', color=colors(i), markersize=10)  # Green circles for grasping point

            # Plot the second approach
            px2, py2, gx2, gy2 = approach2
            translated_px_py2 = convert_coordinates((px2, py2))
            translated_gx_gy2 = convert_coordinates((gx2, gy2))
            ax.plot(translated_px_py2[1], translated_px_py2[0], 'ro', color=colors(i), markersize=10)  # Red circles for position
            ax.plot(translated_gx_gy2[1], translated_gx_gy2[0], 'go', color=colors(i), markersize=10)  # Green circles for grasping point

            # Plot a line connecting the two approaches
            ax.plot([translated_gx_gy1[1], translated_gx_gy2[1]], [translated_px_py1[0], translated_px_py2[0]], color=colors(i), linewidth=2)
            # ax.plot([translated_gx_gy1[1], translated_gx_gy2[1]], [translated_gx_gy1[0], translated_gx_gy2[0]], color=colors(i), linewidth=2)

        # Set labels and title
        ax.set_xlabel('X-coordinate')
        ax.set_ylabel('Y-coordinate')
        ax.set_title('Dataset Visualization')
        
        # visualize
        plt.show()

        # Save the plot as an image
        plt.savefig(os.path.join(save_path, image_name))

        # Close the plot to free up memory
        plt.close(fig)

    ########################################################################
    ############ Update 2.0
    def get_topk_opposite_alignment_pairs(self, topk=5):
        """
        For each unique pair, score opposite alignment. Returns topk [(score, (i, j))].
        """
        n = self.grasping_approach.shape[0]
        results = []
        for i in range(n):
            a1 = self.grasping_approach[i]
            # Skip zero/invalid grasps
            if np.allclose(a1, 0):
                continue
            for j in range(i+1, n):
                a2 = self.grasping_approach[j]
                if np.allclose(a2, 0):
                    continue
                score = opposite_alignment_score(a1, a2)
                results.append((score, (i, j)))
        # Sort and return topk
        results.sort(reverse=True, key=lambda x: x[0])
        return [pair for _, pair in results[:topk]]
  
def opposite_alignment_score(approach1, approach2):
    """
    Scalar in [0, 1]: 1 means perfectly opposite approach directions.
    """
    v1 = approach_vector(approach1)
    v2 = approach_vector(approach2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    # Highest when cos_theta == -1
    return 1 - abs(cos_theta + 1)
  
def approach_vector(approach):
    """Return the vector from robot base to grasp (2D)."""
    return np.array([approach[0] - approach[2], approach[1] - approach[3]])
