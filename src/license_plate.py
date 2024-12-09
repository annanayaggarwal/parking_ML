import cv2
import numpy as np
import pytesseract
from datetime import datetime
import pandas as pd

class LicensePlateRecognizer:
    def __init__(self):
        self.min_area = 500
        self.max_area = 5000
        self.plate_records = pd.DataFrame(columns=['plate_number', 'entry_time', 'exit_time', 'spot_id'])
        
    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return opening
    
    def detect_plate_region(self, img):
        preprocessed = self.preprocess_image(img)
        contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        potential_plates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                if 2.0 < w/h < 5.0:  # License plate aspect ratio
                    potential_plates.append((x, y, w, h))
        
        return potential_plates
    
    def recognize_plate(self, img):
        plates = self.detect_plate_region(img)
        best_text = ""
        
        for (x, y, w, h) in plates:
            plate_img = img[y:y+h, x:x+w]
            processed = self.preprocess_image(plate_img)
            
            # OCR
            text = pytesseract.image_to_string(processed, 
                                             config='--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            text = text.strip()
            
            if len(text) > 4:  # Minimum plate length
                best_text = text
                break
        
        return best_text
    
    def record_entry(self, plate_number, spot_id):
        entry_time = datetime.now()
        new_record = pd.DataFrame({
            'plate_number': [plate_number],
            'entry_time': [entry_time],
            'exit_time': [None],
            'spot_id': [spot_id]
        })
        self.plate_records = pd.concat([self.plate_records, new_record], ignore_index=True)
        return entry_time
    
    def record_exit(self, plate_number):
        exit_time = datetime.now()
        idx = self.plate_records[
            (self.plate_records['plate_number'] == plate_number) & 
            (self.plate_records['exit_time'].isna())
        ].index
        
        if len(idx) > 0:
            self.plate_records.loc[idx[0], 'exit_time'] = exit_time
            return self.plate_records.loc[idx[0], 'entry_time']
        return None
    
    def calculate_fee(self, entry_time, exit_time, rate_per_hour=10):
        duration = (exit_time - entry_time).total_seconds() / 3600
        return rate_per_hour * duration