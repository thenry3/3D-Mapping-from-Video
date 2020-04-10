import numpy as np
import cv2
from skimage.transform import EssentialMatrixTransform
from skimage.transform import FundamentalMatrixTransform
from skimage.measure import ransac

# 3x4 matrix with identity matrix
IRt = np.eye(4)


def norm_points(Kinv, points):
    def concat_ones(m):
        return np.concatenate([m, np.ones((m.shape[0], 1))], axis=1)
    return np.dot(Kinv, concat_ones(points).T).T[:, 0:2]


def denormalize_point(Kmatrix, point):
    normalized = np.dot(Kmatrix, [point[0], point[1], 1.0])
    normalized /= normalized[2]
    return int(round(normalized[0])), int(round(normalized[1]))


def extractRt(frame):
    W = np.mat([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    U, d, Vt = np.linalg.svd(frame)

    if np.linalg.det(Vt) < 0:
        Vt *= -1.0
    if np.linalg.det(U) < 0:
        U *= -1.0
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    Rt = np.eye(4)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def match(frame1, frame2):
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(frame1.des, frame2.des, k=2)

    matched_points = []
    index1, index2 = [], []

    # god bless for this
    for m1, m2 in matches:
        if m1.distance < 0.75 * m2.distance:
            pt1 = frame1.points[m1.queryIdx]
            pt2 = frame2.points[m1.trainIdx]

            if m1.distance < 32 and np.linalg.norm((pt1 - pt2)) < 0.1 * np.linalg.norm([frame1.w, frame1.h]):
            # if np.linalg.norm((pt1 - pt2)) < 0.1:
                if m1.queryIdx not in index1 and m1.trainIdx not in index2:
                    index1.append(m1.queryIdx)
                    index2.append(m1.trainIdx)

                    matched_points.append((pt1, pt2))

    matched_points = np.array(matched_points)
    index1 = np.array(index1)
    index2 = np.array(index2)

    # TBH also googled this heheh
    model, inliers = ransac((matched_points[:, 0], matched_points[:, 1]),
                            FundamentalMatrixTransform,
                            min_samples=8,
                            residual_threshold=0.002,
                            max_trials=100)
    matched_points = matched_points[inliers]

    Rt = extractRt(model.params)

    return index1[inliers], index2[inliers], Rt


class Frame():
    def __init__(self, space, frame_img, Kmatrix):
        self.Kmatrix = Kmatrix
        self.Kinv = np.linalg.inv(self.Kmatrix)
        self.frame_img = frame_img
        self.pose = IRt
        self.h, self.w = frame_img.shape[0:2]

        self.kpus, self.des = self.extract()
        self.points = norm_points(self.Kinv, self.kpus)
        self.pts = [None] * len(self.points)

        self.ID = len(space.frames)
        space.frames.append(self)

    def extract(self):
        orb = cv2.ORB_create()

        features = cv2.goodFeaturesToTrack(
            np.mean(self.frame_img, axis=2).astype(np.uint8), 10000, 0.01, 3)

        key_points = [cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=10)
                      for point in features]
        key_points, des = orb.compute(self.frame_img, key_points)

        return np.array([(point.pt[0], point.pt[1]) for point in key_points]), des
