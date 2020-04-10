import numpy as np
import sys
from window import Window
import cv2
from frame import Frame, match, denormalize_point, IRt
from space import Space, Node

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
    ret = np.zeros((points1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(points1, points2)):
        temp = np.zeros((4,4))
        temp[0] = p[0][0] * pose1[2] - pose1[0]
        temp[1] = p[0][1] * pose1[2] - pose1[1]
        temp[2] = p[1][0] * pose2[2] - pose2[0]
        temp[3] = p[1][1] * pose2[2] - pose2[1]
        s, v, d = np.linalg.svd(temp)
        ret[i] = d[3]
    return ret



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
