import numpy as np

class SphereObstacle:
    def __init__(self, position, radius):
        self.position = np.array(position)
        self.radius = radius

    def check_collision(self, point, threshold=0.0):
        distance = np.linalg.norm(self.position - point)
        return distance < (self.radius + threshold)

class GroundPlane:
    def __init__(self, z_height=0.0):
        self.z_height = z_height

    def check_collision(self, point, threshold=0.0):
        return point[2] < (self.z_height + threshold)
