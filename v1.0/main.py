"""
@usage: python main.py --modeldir coco_ssd_mobilenet_v1_1
"""
from imutils.video import FPS
import argparse
import cv2
import os
from net import Model
from utility import display_bbox, distance_violation
import numpy as np


class SocialDistancing:
    def __init__(self, modeldir, graph, labels, threshold, resolution):
        self.MODEL_NAME = modeldir
        self.GRAPH_NAME = graph
        self.LABELMAP_NAME = labels
        self.min_conf_threshold = float(threshold)
        self.resW, self.resH = resolution.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)

        # Get path to current working directory
        CWD_PATH = os.getcwd()

        # Path to .tflite file, which contains the model that is used for object detection
        self.PATH_TO_CKPT = os.path.join(CWD_PATH, self.MODEL_NAME, self.GRAPH_NAME)

        # Path to label map file
        self.PATH_TO_LABELS = os.path.join(CWD_PATH, self.MODEL_NAME, self.LABELMAP_NAME)

        # Load the label map
        with open(self.PATH_TO_LABELS, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        # Have to do a weird fix for label map if using the COCO "starter model" from
        # https://www.tensorflow.org/lite/models/object_detection/overview
        # First label is '???', which has to be removed.
        if self.labels[0] == '???':
            del (self.labels[0])

        self.x_dist_thresh = 400
        self.dist_thres = 30

        # Initialize video stream
        self.videostream = cv2.VideoCapture("../video/output.mp4")
        ret = self.videostream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.videostream.set(3, self.imW)
        ret = self.videostream.set(4, self.imH)

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # initialize our list of queues -- both input queue and output queue
        # for *every* object that we will be tracking
        self.inputQueues = []
        self.outputQueues = []

        # parallel points
        self.pts = np.array([(8, 178), (246, 81), (627, 147),  (540, 393)], dtype="float32")
        self.frame_no = 1
        self.dst = np.array([[248, 409], [380, 409], [380, 541], [248, 541]], dtype="float32")

        # initialize model
        self.model = Model(self.PATH_TO_CKPT)

    def run(self):
        fps = FPS().start()
        count = 0
        # loop over frames from the video file stream
        while True:
            t1 = cv2.getTickCount()

            # grab the next frame from the video file
            grabbed, frame1 = self.videostream.read()

            # check to see if we have reached the end of the video file
            if frame1 is None:
                break

            frame1 = cv2.resize(frame1, (self.imW, self.imH))
            frame = frame1.copy()

            # resize the frame for faster processing and then convert the
            # frame from BGR to RGB ordering (dlib needs RGB ordering)
            # frame = imutils.resize(frame, width=600)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if our list of queues is empty then we know we have yet to
            # create our first object tracker
            count += 1

            # gain matrix for birds eye
            if self.frame_no == 1:
                Mat = cv2.getPerspectiveTransform(self.pts, self.dst)
                Mat_inv = cv2.getPerspectiveTransform(self.dst, self.pts)

            if (len(self.inputQueues) == 0) or (count == 15):
                self.outputQueues = []
                self.inputQueues = []

                count = 0
                # grab the frame dimensions and convert the frame to a blob
                scores, classes, boxes = self.model.model_out(frame)

                self.inputQueues, self.outputQueues, frame, bottom_cord = display_bbox(frame, self.inputQueues, self.outputQueues, rgb,
                                                                self.x_dist_thresh, scores, self.labels, classes,
                                                                self.min_conf_threshold,
                                                                boxes, self.imH, self.imW, multi=True)
                warped = cv2.warpPerspective(frame, Mat, (1200, 900))
                bottom_cord_warped = cv2.perspectiveTransform(bottom_cord, Mat)
                bottom_cord_warped = np.round(bottom_cord_warped)
                # for i in bottom_cord_warped[0]:
                #     cv2.circle(warped, center=(i[0], i[1]), radius=8, color=(0, 255, 0), thickness=-1)
                ret, violation_pts = distance_violation(bottom_cord_warped, warped, self.dist_thres, Mat_inv)
                if ret:
                    for i in violation_pts:
                        cv2.line(frame, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 0, 225), 4)

            # otherwise, we've already performed detection so let's track
            # multiple objects
            else:
                self.inputQueues, self.outputQueues, frame, bottom_cord = display_bbox(frame, self.inputQueues, self.outputQueues, rgb,
                                                                self.x_dist_thresh, multi=False)
                warped = cv2.warpPerspective(frame, Mat, (1200, 900))
                bottom_cord_warped = cv2.perspectiveTransform(bottom_cord, Mat)
                bottom_cord_warped = np.round(bottom_cord_warped)
                # for i in bottom_cord_warped[0]:
                #     cv2.circle(warped, center=(i[0], i[1]), radius=8, color=(0, 255, 0), thickness=-1)
                ret, violation_pts = distance_violation(bottom_cord_warped, warped, self.dist_thres, Mat_inv)
                if ret:
                    for i in violation_pts:
                        cv2.line(frame, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 0, 225), 4)

            # Draw framerate in corner of frame
            cv2.putText(frame, 'FPS: {0:.2f}'.format(self.frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA)

            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / self.freq
            self.frame_rate_calc = 1 / time1
            # show the output frame
            cv2.imshow("Frame", frame)
            #cv2.imshow("warped", warped)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # update the FPS counter
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        frame_rate_calc = fps.fps()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(frame_rate_calc))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.videostream.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
    ap.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
    ap.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
    ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
    ap.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x420')
    args = ap.parse_args()
    sd = SocialDistancing(args.modeldir, args.graph, args.labels, args.threshold, args.resolution)
    sd.run()
