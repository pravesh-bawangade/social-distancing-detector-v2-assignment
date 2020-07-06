# Social Distancing Detection.

## Description:
```
A social distancing detector using tensorflow and opencv.
There are two versions:
1. Using SSD-Mobilenet TFLite model.
2. Using regular pre-trained models.
Flow:
1. Detection using pre-trained detectors.
2. Tracking using dlib tracker.
3. Perspective transform for birds eye view.
4. Transforming detected co-ordinates to perspective domain.
5. Checking threshold distance between co-ordinates in perspective domain.
6. Transforming distance violating co-ordinates to original domain.
7. displaying and drawing lines between original domain coordinates.
```
   
## Installation:

- I recommend creating two different environments for two different versions.(If you want to test both)
- If you are testing **v11** please only install 'requirementsV11.txt' 
```
    pip install -r requirementsV11.txt
```

## Run:
```
   For Version V11:
    -> cd v11
    -> python main.py --model_name "faster_rcnn_resnet50_coco_2018_01_28" --num_frames 60

    (variations):
    -> python main.py --model_name 'ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03' 
        --num_frames 15 
    -> python main.py --model_name 'ssd_mobilenet_v1_coco_2018_01_28' --num_frames 15
    
    For Version V1.0
    -> cd v1.0
    -> python main.py --modeldir coco_ssd_mobilenet_v1_1
```
## References:

```
1. Detection:
    1. Object detection using pre-trained detectors:  https://github.com/tensorflow/models/tree/master/research/object_detection
    2. For tflite model: https://www.tensorflow.org/lite/models/object_detection/overview
2. Tracking:
    1. Dlib tracker: Danelljan, Martin, et al. "Accurate scale estimation for robust visual tracking." Proceedings of the British Machine Vision Conference BMVC. 2014.
    2. Implementation: https://www.pyimagesearch.com/2018/10/29/multi-object-tracking-with-dlib/
3. Perspective Transform:
    1. Explanation and Implementation: https://www.learnopencv.com/homography-examples-using-opencv-python-c/
    2. Implementation: https://stackoverflow.com/questions/57439977/transforming-perspective-view-to-a-top-view
    3. Implementation: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
4. Social distancing detection:
    1. Landing AI: https://landing.ai/landing-ai-creates-an-ai-tool-to-help-customers-monitor-social-distancing-in-the-workplace/
    2. Aqeel Anwar : https://towardsdatascience.com/monitoring-social-distancing-using-ai-c5b81da44c9f
```
