import numpy as np
from window import Window
import cv2
import sys
from frame import Frame, match, denormalize_point, IRt
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import *
from direct.directtools.DirectGeometry import LineNodePath
from multiprocessing import Process, Queue
from timeit import default_timer as timer

# video resize dimensions
# originally 1920 x 1080
HEIGHT = 960
WIDTH = 540

# focal length
F = 270
# intrinsic paramters
K = np.array([[F, 0, WIDTH//2], [0, F, HEIGHT//2], [0, 0, 1]])

queue = Queue()


class Space(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)
        self.frames = []
        self.nodes = []
        self.state = None

        ls = LineSegs()
        ls.setThickness(10)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(1.0, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 1.0, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 0.0, 1.0)

        node = ls.create()
        render.attachNewNode(node)

    def refresh(self, q):
        if self.state is None or not queue.empty():
            self.state = queue.get()

        if len(self.state[0]) > 20:
            start = self.state[0][-20]
            end = self.state[0][-10]
            base.camera.setPos(start[0, 3], start[1, 3], start[2, 3])
            base.camera.setHpr(LVecBase3f(
                end[0, 3] - start[0, 3], end[1, 3] - start[1, 3], end[2, 3] - start[2, 3]))
            print(end[0, 3] - start[0, 3])
        LNP = LineNodePath(render, 'box', 4, VBase4(1, 0, 0, 1))
        LNP.drawLines([[d[:3, 3] for d in self.state[0]]])
        LNP.create()

        lsp = LineSegs()
        lsp.setThickness(2)
        lsp.setColor(0, 1, 0)
        for pos in self.state[1]:
            lsp.moveTo(pos[0], pos[1], pos[2])
            lsp.drawTo(pos[0], pos[1], pos[2] + 1)

        render.attachNewNode(lsp.create())

    def display(self):
        poses, points = [], []
        for frame in self.frames:
            poses.append(frame.pose)
        for node in self.nodes:
            points.append(node.location)
        queue.put((poses, points))
        self.refresh(queue)


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


window = Window(HEIGHT, WIDTH)
space = Space()


def triangulate(pose1, pose2, points1, points2):
    return cv2.triangulatePoints(pose1[:3], pose2[:3], points1.T, points2.T).T


def process_frame(frame_img):
    frame_img = cv2.resize(frame_img, (HEIGHT, WIDTH))
    frame = Frame(space, frame_img, K)
    if frame.ID == 0:
        return

    curr_frame = space.frames[-1]
    prev_frame = space.frames[-2]

    index1, index2, Rt = match(curr_frame, prev_frame)
    curr_frame.pose = np.dot(Rt, prev_frame.pose)

    points4d = triangulate(
        curr_frame.pose, prev_frame.pose, curr_frame.points[index1], prev_frame.points[index2])
    points4d /= points4d[:, 3:]

    # filter
    filter4d = (np.abs(points4d[:, 3]) > 0.005) & (points4d[:, 2] > 0)

    for i, pt in enumerate(points4d):
        if not filter4d[i]:
            continue
        node = Node(space, pt)
        node.add_frame(curr_frame, index1[i])
        node.add_frame(prev_frame, index2[i])

    for point1, point2 in zip(curr_frame.points[index1], prev_frame.points[index2]):
        pt1_x, pt1_y = denormalize_point(K, point1)
        pt2_x, pt2_y = denormalize_point(K, point2)

        cv2.circle(frame_img, (pt1_x, pt1_y), 3, (255, 0, 0))
        cv2.line(frame_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255))

    window.render(frame_img)

    space.display()
    taskMgr.step()


def loop(queue):
    while 1:
        space.refresh(queue)
        taskMgr.step()


if __name__ == "__main__":
    vid = cv2.VideoCapture("test.mp4")
    # p = Process(target=loop, args=(queue,))
    # p.daemon = True
    # p.start()
    # base.disableMouse()
    base.camera.setPos(0, 20, 0)
    start = timer()
    while vid.isOpened():
        result, frame = vid.read()

        if result == True:
            process_frame(frame)
        else:
            break

    while 1:
        tskMgr.step()
