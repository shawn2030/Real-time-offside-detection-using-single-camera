import torch
import numpy as np
import cv2 as cv
import torchvision


class PoseEstimator:
    coco_keypoints = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle'
    ]

    coco_skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (6, 8), (7, 9),
        (8, 10), (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)
    ]

    desired_keypoints = [
        5, 6, 13, 14,
        15, 16
    ]

    def __init__(self, detection_quality_threshold: float, keypoint_quality_threshold: float):
        self.weight = torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=self.weight)
        self.transforms = self.weight.transforms()
        self.detection_quality_threshold = detection_quality_threshold
        self.keypoint_quality_threshold = keypoint_quality_threshold
        self.model.eval()

    def detect_and_draw(self, img: np.ndarray) -> np.ndarray:
        """
        detect and draw keyposes on all the players for ankles, shoulders, and knees
        using keypointRCNN_Resnet50 model and return the collection of points.
        """
        torch_img = torch.from_numpy(cv.cvtColor(img, cv.COLOR_BGR2RGB) / 255.0).permute(2, 0, 1).float()        
        torch_img = self.transforms(torch_img)
        with torch.no_grad():
            result = self.model([torch_img])[0]
            scores = result["scores"].numpy()
            keypoints = result["keypoints"].int().numpy()
            boxes = result["boxes"].int().numpy()
            labels = result["labels"].numpy()
        # print(keypoints.shape)
        collected_points_list = []

        for score, keypoint, box, label in zip(scores, keypoints, boxes, labels):
            if score < self.detection_quality_threshold:
                continue

            x, y, w, h = box
            cv.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)

            label_picture = self.weight.meta["categories"][label]

            cv.putText(img, label_picture, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            count = 0
            if self.weight.meta["categories"][label] == "person":
                for i, point in enumerate(keypoint):
                    count += 1

                    if point[2] < self.keypoint_quality_threshold :
                        continue
                    if i in self.desired_keypoints:
                        cv.circle(img, (point[0], point[1]), 7, (0, 255, 0), cv.FILLED)
                        collected_points_list.append((point[0], point[1]))

        return img, collected_points_list


