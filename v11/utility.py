"""
@Author: Pravesh Bawangade
@filename: utility.py
"""

import numpy as np
import cv2
import multiprocessing
from tracker import start_tracker
from scipy.spatial.distance import pdist, squareform


def display_bbox(frame, inputQueues, outputQueues, rgb,
                 x_dist_thresh, scores=None, classes=None, min_conf_threshold=None,
                 boxes=None, imH=None, imW=None, multi=False):
    """
    Draw bounding box on detected people and apply tracking.
    :param frame: Input frame
    :param inputQueues: Input queue for frames
    :param outputQueues: Output queue for Co-ordinates
    :param rgb: RGB image for dlib tracking
    :param x_dist_thresh: distance threshold to eliminate noisy big bounding box
    :param scores: score for predictions
    :param classes: classes of predictions
    :param min_conf_threshold: minimum score to display
    :param boxes: bounding box coordinates
    :param imH: Height of image
    :param imW: Width of image
    :param multi: True to start tracking only
    :return: inputQueues, outputQueues, frame, np.array([bottom_cord], np.float32)
    """
    bottom_cord = []
    flag = True
    if multi:

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            # labels[int(classes[i])] == 'person'
            if (int(classes[i]) == 1) and ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                x_dist = (xmax - xmin)

                bb = [xmin, ymin, xmax, ymax]

                if x_dist > x_dist_thresh:
                    continue

                # Bottom co-ordinates
                x_mid, y_mid = mid_bottom(xmin, xmax, ymax)
                bottom_cord.append((x_mid, y_mid))
                cv2.circle(frame, (int(x_mid), int(y_mid)), radius=5, color=(255, 0, 0), thickness=-1)

                # create two brand new input and output queues,
                # respectively
                iq = multiprocessing.Queue()
                oq = multiprocessing.Queue()
                inputQueues.append(iq)
                outputQueues.append(oq)

                # spawn a daemon process for a new object tracker
                p = multiprocessing.Process(
                    target=start_tracker,
                    args=(bb, rgb, iq, oq))
                p.daemon = True
                p.start()
                # actual_bbox.append(bb)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

    else:
        for iq in inputQueues:
            iq.put(rgb)

        # loop over each of the output queues
        for oq in outputQueues:
            # grab the updated bounding box coordinates for the
            # object -- the .get method is a blocking operation so
            # this will pause our execution until the respective
            # process finishes the tracking update
            (xmin, ymin, xmax, ymax) = oq.get()

            x_dist = (xmax - xmin)
            if x_dist > x_dist_thresh:
                continue

            # Bottom co-ordinates
            x_mid, y_mid = mid_bottom(xmin, xmax, ymax)
            bottom_cord.append((x_mid, y_mid))
            cv2.circle(frame, (int(x_mid), int(y_mid)), radius=5, color=(255, 0, 0), thickness=-1)

            # draw the bounding box from the correlation object
            # tracker
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)
    if len(bottom_cord) == 0:
        flag = False

    return inputQueues, outputQueues, frame, np.array([bottom_cord], np.float32), flag


def four_point_transform(image, pts):
    """
    Perspective transformation of image and returns transform matrix
    :param image: Input image
    :param pts: parallel coordinates
    :return: transformation matrix, and warped image
    """
    # obtain a consistent order of the points and unpack them
    # individually
    rect = pts
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([[248, 409], [380, 409], [380, 541], [248, 541]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth + 500, maxHeight + 500))

    # return the warped image
    return M, warped


def mid_bottom(xmin, xmax, ymax):
    """
    return bottom mid point
    :param xmin: top right x coordinate
    :param xmax: bottom left x coordinate
    :param ymax: bottom y coordinate
    :return: x_mid, y_mid
    """
    x_mid, y_mid = (xmax + xmin) / 2, ymax
    return x_mid, y_mid


def distance_violation(bottom_cord_warped, d_thresh, Mat_inv):
    """
    Check distance violation and return original image co-ordinates
    :param bottom_cord_warped: transformed co-ordinates
    :param d_thresh:distance threshold
    :param Mat: transformation matrix
    :return: True is violation detected and Inverse transform co-ordinates.
    """
    p = bottom_cord_warped[0]
    dist_condensed = pdist(p)
    dist = squareform(dist_condensed)

    dd = np.where(dist < d_thresh * 6 / 10)
    close_p = []
    for i in range(int(np.ceil(len(dd[0]) / 2))):
        if dd[0][i] != dd[1][i]:
            point1 = dd[0][i]
            point2 = dd[1][i]

            close_p.append([p[point1], p[point2]])

    if len(close_p) == 0:
        return False, None
    close_p = cv2.perspectiveTransform(np.array(close_p, np.float32), Mat_inv)

    return True, close_p
