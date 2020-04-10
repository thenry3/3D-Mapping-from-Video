import sys
sys.path.append("lib")

import pangolin
import g2o
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

    #shitty
    # def optimize(self):
    #     optimizer = g2o.SparseOptimizer()
    #     solver = g2o.OptimizationAlgorithmLevenberg(g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3()))
    #     optimizer.set_algorithm(solver)

    #     rob_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))

    #     if LOC_WINDOW == None:
    #         local_frames = self.frames
    #     else:
    #         local_frames = self.frames[-LOC_WINDOW:]
        
    #     for frame in self.frames:
    #         pose = frame.pose
    #         cam = g2o.SBACam(g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3]))
    #         cam.set_cam(frame.Kmatrix[0][0], frame.Kmatrix[1][1], frame.Kmatrix[0][2], frame.Kmatrix[1][2], 1.0)

    #         vcam = g2o.VertexCam()
    #         vcam.set_id(frame.ID)
    #         vcam.set_estimate(cam)
    #         vcam.set_fixed(frame.ID <= 1 or frame not in local_frames)
    #         optimizer.add_vertex(vcam)

    #     OFFSET = 0x10000
    #     for node in self.nodes:
    #         if not any([frame in local_frames for frame in node.frames]):
    #             continue

    #         point = g2o.VertexSBAPointXYZ()
    #         point.set_id(node.ID + OFFSET)
    #         point.set_estimate(node.location[0:3])
    #         point.set_marginalized(True)
    #         point.set_fixed(False)
    #         optimizer.add_vertex(point)

    #         for frame in node.frames:
    #             edge = g2o.EdgeProjectP2MC()
    #             edge.set_vertex(0, point)
    #             edge.set_vertex(1, optimizer.vertex(frame.ID))
    #             uv = frame.kpus[frame.pts.index(node)]
    #             edge.set_measurement(uv)
    #             edge.set_information(np.eye(2))
    #             edge.set_robust_kernel(rob_kernel)
    #             optimizer.add_edge(edge)

    #     optimizer.initialize_optimization()
    #     optimizer.optimize(50)

    #     for frame in self.frames:
    #         est = optimizer.vertex(frame.ID).estimate()
    #         R = est.rotation().matrix()
    #         t = est.translation()
    #         poseRt = np.eye(4)
    #         poseRt[:3, :3] = R 
    #         poseRt[:3, 3] = t
    #         frame.pose = poseRt

    #     new_points = []
    #     for node in self.nodes:
    #         vertex = optimizer.vertex(node.ID + OFFSET)
    #         if vertex is None:
    #             new_points.append(node)
    #             continue
    #         est = vertex.estimate()

    #         old_node = len(node.frames) == 2 and node.frames[-1] not in local_frames

    #         errors = []
    #         for frame in node.frames:
    #             uv = frame.kpus[frame.pts.index(node)]
    #             projection = np.dot(np.dot(frame.Kmatrix, np.linalg.inv(frame.pose)[:3]), np.array([est[0], est[1], est[2], 1.0]))
    #             projection = projection[0:2] / projection[2]
    #             errors.append(np.linalg.norm(projection - uv))

    #         if (old_node and np.mean(errors) > 30) or np.mean(errors) > 100:
    #             node.remove()
    #             continue

    #         node.location = np.array(est)
    #         new_points.append(node)

    #     self.nodes = new_points

    #     return optimizer.chi2()


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

