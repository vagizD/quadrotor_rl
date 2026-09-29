import numpy as np

class PhysicalEnvironment:
    """Manages obstacles, ground plane, and environmental interactions."""
    def __init__(self, ground_z=0.0):
        self.obstacles = []
        self.ground = ground_z

    def add_obstacle(self, obstacle):
        self.obstacles.append(obstacle)

    def check_collision(self, position, radius=0.1):
        """Check if a spherical rigid body collides with anything."""
        # 1. Ground collision
        if position[2] < (self.ground + radius):
            return True, "ground"
            
        # 2. Obstacle collision
        for i, obs in enumerate(self.obstacles):
            if obs.check_collision(position, threshold=radius):
                return True, f"obstacle_{i}"
                
        return False, None
