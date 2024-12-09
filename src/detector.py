import cv2
import numpy as np
from datetime import datetime
import pandas as pd

class ParkingDetector:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.results = pd.DataFrame(columns=['spot_id', 'status', 'timestamp'])
        
    def process_frame(self, frame, spots):
        results = {}
        for spot_id, (x1, y1, x2, y2) in spots.items():
            # Extract and preprocess spot image
            spot_img = frame[y1:y2, x1:x2]
            spot_img = cv2.resize(spot_img, self.config.IMAGE_SIZE)
            spot_img = spot_img / 255.0
            
            # Predict
            prediction = self.model.model.predict(np.expand_dims(spot_img, axis=0))[0][0]
            status = 'empty' if prediction > 0.5 else 'occupied'
            
            # Store results
            results[spot_id] = {
                'status': status,
                'confidence': float(prediction)
            }
            
            # Record to DataFrame
            self.results = pd.concat([
                self.results,
                pd.DataFrame({
                    'spot_id': [spot_id],
                    'status': [status],
                    'timestamp': [datetime.now()]
                })
            ], ignore_index=True)
            
        return results