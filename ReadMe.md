# Social Distancing Detection Assignment - ORBO.AI

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

