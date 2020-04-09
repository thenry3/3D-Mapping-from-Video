import sys
sys.path.append("lib")

import pangolin
import OpenGL.GL as gl
import numpy as np
from multiprocessing import Process, Queue

class Space():
    def __init__(self):
        self.frames = []
        self.nodes = []
        self.state = None

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
        pangolin.CreateWindowAndBind('Main', 640, 480)
        gl.glEnable(gl.GL_DEPTH_TEST)

        self.scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 100), 
            pangolin.ModelViewLookAt(-2, 2, -2, 0, 0, 0, pangolin.AxisDirection.AxisY))
        self.handler = pangolin.Handler3D(self.scam)

        self.dcam = pangolin.CreateDisplay()
        self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -640.0/480.0)
        self.dcam.SetHandler(self.handler)

    def refresh(self, q):
        if self.state is None or not self.queue.empty():
            self.state = self.queue.get()

        ppts = np.array([pose[:3, 3] for pose in self.state[0]])
        spts = np.array(self.state[1])

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)
        self.dcam.Activate(self.scam)

        gl.glPointSize(10)
        gl.glColor3f(1, 0, 0)
        pangolin.DrawCameras(np.array(self.state[0]))

        gl.glPointSize(2)
        gl.glColor3f(0, 1, 0)
        pangolin.DrawPoints(spts)

        pangolin.FinishFrame()

    def display(self):
        self.queue.put(([frame.pose for frame in self.frames], [node.location for node in self.nodes]))


class Node():
    def __init__(self, space, location):
        self.frames = []
        self.location = location
        self.indexes = []
        self.ID = len(space.nodes)
        space.nodes.append(self)

    def add_frame(self, frame, index):
        self.frames.append(frame)
        self.indexes.append(index)
