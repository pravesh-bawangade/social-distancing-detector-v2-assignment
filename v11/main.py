"""
@Author: Pravesh Bawangade
@filename: main.py
@usage: 1. python main.py --model_name 'faster_rcnn_resnet50_coco_2018_01_28'
        2. python main.py --model_name 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' /
         --num_frames 15
        3. python main.py --model_name 'ssd_mobilenet_v1_coco_2018_01_28' --num_frames 15
"""

from imutils.video import FPS
import argparse
import cv2
from model import model
from utility import display_bbox, distance_violation
import numpy as np


class SocialDistancing:
    """
    Social Distancing class.
    """
    def __init__(self, threshold, resolution, num_frames, model_name):
        """
        initializing required variables
        :param threshold: min threshold percentage for detection
        :param resolution: resolution of image
        """

        self.min_conf_threshold = float(threshold)
        self.resW, self.resH = resolution.split('x')
        self.imW, self.imH = int(self.resW), int(self.resH)

        self.x_dist_thresh = 400
        self.dist_thres = 30

        # Initialize video stream
        self.videostream = cv2.VideoCapture("../video/vid_short.mp4")
        ret = self.videostream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.videostream.set(3, self.imW)
        ret = self.videostream.set(4, self.imH)
        ret = self.videostream.set(cv2.CAP_PROP_FPS, 10)

        # Initialize frame rate calculation
        self.frame_rate_calc = 1
        self.freq = cv2.getTickFrequency()

        # initialize our list of queues -- both input queue and output queue
        # for *every* object that we will be tracking
        self.inputQueues = []
        self.outputQueues = []

        # parallel points
        # self.pts = np.array([(8, 178), (246, 81), (627, 147),  (540, 393)], dtype="float32")
        # for vid_short
        self.pts = np.array([(63, 217), (333, 23), (608, 77), (552, 356)], dtype="float32")
        # self.pts = np.array([(137, 256), (360, 95), (480, 132), (319, 337)], dtype="float32")
        self.first_frame = True
        # Destination points
        self.dst = np.array([[248, 409], [380, 409], [380, 541], [248, 541]], dtype="float32")

        # skip frames
        self.num_frames = int(num_frames)

        self.network = model(model_name)

    def run(self):
        """
        Start execution
        :return: None
        """
        fps = FPS().start()
        count = 0

        # loop over frames from the video file stream
        while True:
            t1 = cv2.getTickCount()

            # grab the next frame from the video file
            grabbed, frame = self.videostream.read()

            # check to see if we have reached the end of the video file
            if frame is None:
                break

            frame = cv2.resize(frame, (self.imW, self.imH))

            # resize the frame for faster processing and then convert the
            # frame from BGR to RGB ordering (dlib needs RGB ordering)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            count += 1

            # gain matrix for birds eye view
            if self.first_frame:
                Mat = cv2.getPerspectiveTransform(self.pts, self.dst)
                Mat_inv = cv2.getPerspectiveTransform(self.dst, self.pts)
                self.first_frame = False

            if (len(self.inputQueues) == 0) or (count == self.num_frames):
                self.outputQueues = []
                self.inputQueues = []
                count = 0

                # predicting scores, classes and bbox
                scores, classes, boxes = self.network.detect(frame)

                # for tracking
                self.inputQueues, self.outputQueues, frame, bottom_cord, flag = display_bbox(frame, self.inputQueues, self.outputQueues, rgb,
                                                                self.x_dist_thresh, scores, classes,
                                                                self.min_conf_threshold,
                                                                boxes, self.imH, self.imW, multi=True)
                if not flag:
                    continue
                # warped = cv2.warpPerspective(frame, Mat, (1200, 900))
                bottom_cord_warped = cv2.perspectiveTransform(bottom_cord, Mat)
                bottom_cord_warped = np.round(bottom_cord_warped)
                # for i in bottom_cord_warped[0]:
                #     cv2.circle(warped, center=(i[0], i[1]), radius=8, color=(0, 255, 0), thickness=-1)
                ret, violation_pts = distance_violation(bottom_cord_warped, self.dist_thres, Mat_inv)
                if ret:
                    for i in violation_pts:
                        cv2.line(frame, (i[0][0], i[0][1]), (i[1][0], i[1][1]), (0, 0, 225), 4)

            # otherwise, we've already performed detection so let's track
            # multiple objects
            else:
                self.inputQueues, self.outputQueues, frame, bottom_cord, flag = display_bbox(frame, self.inputQueues, self.outputQueues, rgb,
                                                                self.x_dist_thresh, multi=False)
                if not flag:
                    continue
                # warped = cv2.warpPerspective(frame, Mat, (1200, 900))
                bottom_cord_warped = cv2.perspectiveTransform(bottom_cord, Mat)
                bottom_cord_warped = np.round(bottom_cord_warped)
                # for i in bottom_cord_warped[0]:
                #     cv2.circle(warped, center=(i[0], i[1]), radius=8, color=(0, 255, 0), thickness=-1)
                ret, violation_pts = distance_violation(bottom_cord_warped, self.dist_thres, Mat_inv)
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
            # cv2.imshow("warped", warped)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break
            # update the FPS counter
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        frame_rate_calc = fps.fps()
        print("---> Elapsed time: {:.2f} <---".format(fps.elapsed()))
        print("---> Approx. FPS: {:.2f} <---".format(frame_rate_calc))

        # do a bit of cleanup
        cv2.destroyAllWindows()
        self.videostream.release()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
    ap.add_argument('--resolution',
                    help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x420')
    ap.add_argument('--num_frames', help='number of frames before next detection',
                    default=60)
    ap.add_argument('--model_name', help='name of model to use',
                    default='ssd_mobilenet_v1_coco_2018_01_28')
    args = ap.parse_args()
    sd = SocialDistancing(args.threshold, args.resolution, args.num_frames, args.model_name)
    sd.run()
