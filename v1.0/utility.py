# import the necessary packages
import numpy as np
import cv2
import multiprocessing
from tracker import start_tracker
from scipy.spatial.distance import pdist, squareform, cdist


def display_bbox(frame, inputQueues, outputQueues, rgb,
                 x_dist_thresh, scores=None, labels=None, classes=None, min_conf_threshold=None,
                 boxes=None, imH=None, imW=None, multi=False):
    bottom_cord = []
    if multi:

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):

            if (labels[int(classes[i])] == 'person') and ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):
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

    return inputQueues, outputQueues, frame, np.array([bottom_cord], np.float32)


def four_point_transform(image, pts):
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

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    # dst = np.array([
    #     [0, 0],
    #     [maxWidth - 1, 0],
    #     [maxWidth - 1, maxHeight - 1],
    #     [0, maxHeight - 1]], dtype="float32")
    dst = np.array([[248, 409], [380, 409], [380, 541], [248, 541]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth + 500, maxHeight + 500))

    # return the warped image
    return M, warped


def mid_bottom(xmin, xmax, ymax):
    x_mid, y_mid = (xmax + xmin) / 2, ymax
    return x_mid, y_mid


def distance_violation(bottom_cord_warped, bird_image, d_thresh, Mat):
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
    close_p = cv2.perspectiveTransform(np.array(close_p, np.float32), Mat)

    return True, close_p
