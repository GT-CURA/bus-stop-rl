from settings import S
from ultralytics import YOLO
import torch
import cv2
import numpy as np

# A wrapper for the YOLO model trained to detect stops
class StopDetector:

    def __init__(self):
        self.model = YOLO(S.yolo_path)
        self.frame_num = 0

    def run(self, img):
        # Alert console 
        print("\n[Stop Detector] Running model...")

        # Run model
        output = self.model(img)[0]
        return output

    def score_output(self, output):
        # No boxes
        if len(output.boxes) == 0: 
            return 0.0, False, None, 0

        # Scores to be calculated
        primary_score = 0.0
        secondary_score = 0.0
        found = False
        
        # Remake boxes to preserve memory, calc score
        boxes = {}
        biggest_box = 0
        for box in output.boxes:
            label = self.model.names[int(box.cls)]
            conf = float(box.conf)
            boxes[label] = conf
            # Take best evidence of a sign/shelter
            if label in {"shelter", "sign"}:
                primary_score = max(primary_score, conf)
                found = True

                # Determine if this is the "biggest" evidence of a stop
                box_size = float(box.xywhn[0][2] * box.xywhn[0][3])
                if box_size > biggest_box:
                    biggest_box = box_size
            else:
                secondary_score += conf

        # Normalize secondary score if needed
        secondary_score = min(secondary_score, 1.0 - primary_score)

        # Allow small boost from secondary amenities
        total_score = primary_score + S.secondary_boost * secondary_score
        return min(total_score, 1.0), found, boxes, biggest_box
    
    def extract_features(self, img, output):
        # Resize and normalize image
        img_resized = cv2.resize(img, S.img_size)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.tensor(np.transpose(img_resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.model.device)

        # Ensure weigghts are frozen
        with torch.no_grad():
            # Extract backbone, run image through it to get features
            features = self.model.model.model[:11](img_tensor) 

            # Global average pooling (512-dim output)
            pooled_feats = features.mean(dim=[2, 3]).squeeze().cpu().numpy()

            # Get detections
            boxes = output.boxes
            det_vecs = []
            found = False

            # Go through as many bounding boxes as are to be kept
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if i >= S.bbs_kept:
                        break
                    
                    # Check if found :(
                    if self.model.names[int(box.cls)] in {"shelter", "sign"}:
                        found = True

                    # Bounding box info
                    x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy()
                    x_cord = (x1 + x2) / 2
                    y_cord = (y1 + y2) / 2
                    area = (x2 - x1) * (y2 - y1)
                    conf = float(box.conf)
                    cls = int(box.cls)

                    # One-hot class encoding
                    class_one_hot = np.zeros(S.num_classes)
                    if 0 <= cls < S.num_classes:
                        class_one_hot[cls] = 1.0

                    # Final vector per box: [xc, yc, area, conf] + one-hot class
                    det_vec = np.concatenate([[x_cord, y_cord, area, conf], class_one_hot])
                    det_vecs.append(det_vec)

                det_vecs = np.vstack(det_vecs)
            else:
                # No detections: fill with zeros
                det_vecs = np.zeros((S.bbs_kept, 4 + S.num_classes))

            # Pad if fewer than bbs_kept
            if det_vecs.shape[0] < S.bbs_kept:
                padding = np.zeros((S.bbs_kept - det_vecs.shape[0], 4 + S.num_classes))
                det_vecs = np.vstack([det_vecs, padding])

            # Flatten vector
            box_flat = det_vecs.flatten()
            return np.concatenate([pooled_feats, box_flat]), found