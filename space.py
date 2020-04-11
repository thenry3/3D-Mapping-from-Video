import sys
sys.path.append("lib")

import pangolin
import OpenGL.GL as gl
import numpy as np
from multiprocessing import Process, Queue

LOC_WINDOW = 20

HEIGHT = 960
WIDTH = 540

class Space():
    def __init__(self):
        self.frames = []
        self.nodes = []
        self.state = None

        self.max_node = 0

        self.state = None
        self.queue = Queue()

        p = Process(target=self.space_loop, args=(self.queue,))
        p.daemon = True
        p.start()

    def space_loop(self, queue):
        self.space_init()
        while 1:
            self.refresh(queue)

    def space_init(self):
        pangolin.CreateWindowAndBind('Main', 960, 540)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(WIDTH, HEIGHT, 420, 420, WIDTH//2, HEIGHT//2, 0.2, 10000), 
            pangolin.ModelViewLookAt(-0, -10, -8, 0, 0, 0, pangolin.AxisDirection.AxisNegY))
        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def refresh(self, q):
        if not q.empty():
            self.state = q.get()

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0, 0, 0, 1.0)
        self.dcam.Activate(self.scam)

        if self.state:
            gl.glColor3f(1, 0, 0)
            pangolin.DrawCameras(self.state[0])

            gl.glPointSize(2)
            pangolin.DrawPoints(self.state[1], self.state[2])

        pangolin.FinishFrame()

    def display(self):
        poses = [frame.pose for frame in self.frames]
        points = [node.location for node in self.nodes]
        colors = [node.color for node in self.nodes]
        self.queue.put((np.array(poses), np.array(points), np.array(colors) / 256.0))


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

