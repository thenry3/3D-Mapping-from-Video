import numpy as np
import sys
from window import Window
import cv2
from frame import Frame, match, denormalize_point, IRt
from space import Space
from node import Node

# video resize dimensions
# originally 1920 x 1080
HEIGHT = 960
WIDTH = 540

# focal length
F = 800
# intrinsic paramters
K = np.array([[F, 0, WIDTH//2], [0, F, HEIGHT//2], [0, 0, 1]])

window = Window(HEIGHT, WIDTH)
space = Space()


def triangulate(pose1, pose2, points1, points2):
    triangulation = np.zeros((points1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, point in enumerate(zip(points1, points2)):
        temp = np.zeros((4,4))
        temp[0] = point[0][0] * pose1[2] - pose1[0]
        temp[1] = point[0][1] * pose1[2] - pose1[1]
        temp[2] = point[1][0] * pose2[2] - pose2[0]
        temp[3] = point[1][1] * pose2[2] - pose2[1]
        s, v, d = np.linalg.svd(temp)
        triangulation[i] = d[3]
    return triangulation



def process_frame(frame_img):
    frame_img = cv2.resize(frame_img, (HEIGHT, WIDTH))
    frame = Frame(space, frame_img, K)
    if frame.ID == 0:
        return

    curr_frame = space.frames[-1]
    prev_frame = space.frames[-2]

    index1, index2, Rt = match(curr_frame, prev_frame)

    for i, idx in enumerate(index2):
        if prev_frame.pts[idx] is not None:
            prev_frame.pts[idx].add_frame(curr_frame, index1[i])

    # filter4d = np.array([curr_frame.pts[i] is None for i in index1])

    curr_frame.pose = np.dot(Rt, prev_frame.pose)

    points4d = triangulate(
        curr_frame.pose, prev_frame.pose, curr_frame.points[index1], prev_frame.points[index2])
    # points4d = triangulate(Rt, np.eye(4), curr_frame.points[index1], prev_frame.points[index2])
    # filter4d &= np.abs(points4d[:, 3]) > 0.005
    points4d /= points4d[:, 3:]
    # filter4d &= points4d[:, 2] > 0

    # points4d = np.dot(np.linalg.inv(curr_frame.pose), points4d.T).T
    filter4d = (np.abs(points4d[:, 3]) > 0.005) & (points4d[:, 2] > 0)

    for i, pt in enumerate(points4d):
        if not filter4d[i]:
            continue
        u, v = int(round(curr_frame.kpus[index1[i], 0])), int(round(curr_frame.kpus[index1[i], 1]))
        node = Node(space, pt, frame_img[v, u])
        node.add_frame(curr_frame, index1[i])
        node.add_frame(prev_frame, index2[i])

    #display shit
    for point1, point2 in zip(curr_frame.points[index1], prev_frame.points[index2]):
        pt1_x, pt1_y = denormalize_point(K, point1)
        pt2_x, pt2_y = denormalize_point(K, point2)

        cv2.circle(frame_img, (pt1_x, pt1_y), 3, (255, 0, 0))
        cv2.line(frame_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 0, 255))

    # if curr_frame.ID >= 4:
    #     error = space.optimize()

    window.render(frame_img)
    space.display()


if __name__ == "__main__":
    if len(sys.argv) < 0:
        print("NO U FOOL PUT IN A VIDEO")
        exit(1)
    
    try:
        vid = cv2.VideoCapture(sys.argv[1])
    except:
        print("NO U FOOL U NEED A FREAKING VIDEO")
        exit(1)

    while vid.isOpened():
        result, frame = vid.read()

        if result == True:
            process_frame(frame)
        else:
            break
