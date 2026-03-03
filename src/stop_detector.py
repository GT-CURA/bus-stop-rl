from settings import S
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from src.rl_env.graph import Node
from src.utils.objects import Detection, Pic 
from src.streetview.sv import StreetView
from src.utils.context import RoadContext

# A wrapper for the YOLO model trained to detect stops
class StopDetector:

    def __init__(self, sv: StreetView, context: RoadContext):
        self.model = YOLO(S.yolo_path)
        self.sv = sv
        self.context = context
        self.verbose = S.yolo_msgs

    def run(self, img):
            # Run model
            output = self.model(img, verbose=self.verbose)[0]
            
            # Save output
            if S.run_server: output.save('src/utils/server/static/frame.jpg')
            return output

    def score_output(self, output, node: Node, pic: Pic, step, found_prev: bool):
        # No boxes
        if len(output.boxes) == 0: 
            return 0.0, False

        # Scores to be calculated
        primary_score = 0.0
        secondary_score = 0.0
        found = False
        best_bearing = None

        # Sum the score of secondary amenities, get highest confidence primary amenity and its sz
        for box in output.boxes:
            label = self.model.names[int(box.cls)]
            conf = float(box.conf)

            # Update node's scorecard 
            if node.scores[label] < conf:
                node.scores[label] = conf
            
            # Calc bearing using bounding box (FOV is 90)
            box_center = float(box.xywhn[0][0])
            delta_deg = (box_center -.5) * 90
            bearing = (pic.heading + delta_deg) % 360

            # Localize coords
            local_x, local_y = self.context.to_local.transform(pic.lng, pic.lat)

            # Build detection, add to node
            det = Detection(
                bearing=bearing,
                primary_conf=conf,
                box_sz=float(box.xywhn[0][2] * box.xywhn[0][3]),
                cx_norm=box_center,
                label=label,          
                timestamp=step,
                pano_id=pic.pano_id,
                lat = pic.lat,
                lng = pic.lng, 
                local_x=local_x,
                local_y=local_y,
                side=None,
                date=pic.date,
                key=f"{int(round(bearing / 5) * 5)}_{label}",
            )

            # Calc side of road 
            self.sv.calc_street_side(det)
            diminish_factor = node.add_det(det)

            # Take best evidence of a sign/shelter
            if label in {"shelter", "sign"}:
                # Mark as found if meets min conf 
                if conf > S.min_conf:
                    found = True
                
                # Allow a bit of a buffer before diminishing if not found 
                buffer = 0 if found_prev or found else 2

                # Diminish score based on how many times its been found 
                adj_conf = conf
                if diminish_factor > buffer:
                    adj_conf -= 0.03 * ((diminish_factor - buffer) ** 2)
                    adj_conf = max(0, adj_conf)

                # If highest conf primray, set as primary score and get bearing
                if conf > primary_score:

                    # Weigh more heavily if not found 
                    primary_score = adj_conf * S.primary_found if found_prev else adj_conf
                    best_bearing = bearing

            else:
                # Weigh secondary scores more heavily if already found
                if found_prev:
                    secondary_score += conf
                else:
                    secondary_score += S.secondary_prefound * conf
        
        # Before found, most of score is from primary amenities. After, mostly secondary
        secondary_score = min(secondary_score, 1.0)
        if found or found_prev:
            primary_score = min(primary_score, 1.0 - secondary_score)
        else:
            secondary_score = min(secondary_score, 1.0 - primary_score)
        total_score = primary_score + secondary_score

        # Update node's highest conf
        if primary_score > node.best_conf:
            node.best_conf = primary_score
            node.best_bearing = best_bearing
        if S.msg_score_breakdown:
            print(f"Raw primary: {primary_score} | Raw secondary: {secondary_score} | Raw Total: {total_score}")
        return min(total_score, 1.0), found
    
    def extract_features(self, img, output):
        # Resize and normalize image
        img_resized = cv2.resize(img, S.img_size)
        img_resized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.tensor(np.transpose(img_resized, (2, 0, 1)), dtype=torch.float32).unsqueeze(0).to(self.model.device)

        # Ensure weigghts are frozen
        with torch.no_grad():
            # Extract backbone, run image through it to get features
            features = self.model.model.model[:9](img_tensor) 

            # Global average pooling (512-dim output)
            pooled_feats = features.mean(dim=[2, 3]).squeeze().cpu().numpy()

            # Get detections
            boxes = output.boxes
            det_vecs = []

            # Go through as many bounding boxes as are to be kept
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    if i >= S.bbs_kept:
                        break

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
            return np.concatenate([pooled_feats, box_flat])
    
    def get_best_ev(self, output):
        # Just finds highest confidence value of a primary amenity
        best_conf = 0.0

        for box in output.boxes:
            label = self.model.names[int(box.cls)]
            conf = float(box.conf)

            if label in {"shelter", "sign"}:
                best_conf = max(best_conf, conf)

        return best_conf