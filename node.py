import numpy as np

class Node():
    def __init__(self, space, location, color):
        self.frames = []
        self.location = location
        self.indexes = []
        self.ID = space.max_node
        self.color = np.copy(color)
        space.max_node += 1
        space.nodes.append(self)

    def add_frame(self, frame, index):
        frame.pts[index] = self
        self.frames.append(frame)
        self.indexes.append(index)

    def remove(self):
        for frame in self.frames:
            frame.pts[frame.pts.index(self)] = None
        del self