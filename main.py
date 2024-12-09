# main.py
import cv2
import numpy as np
import tensorflow as tf
from src.config import Config
from src.data_loader import DataLoader
from src.model import ParkingModel
import os

def define_parking_spots(frame):
    spots = {}
    temp_frame = frame.copy()
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal spots, temp_frame
        if event == cv2.EVENT_LBUTTONDOWN:
            spots[f'spot_{len(spots)}'] = [x-50, y-50, x+50, y+50]
            cv2.rectangle(temp_frame, (x-50, y-50), (x+50, y+50), (0, 255, 0), 2)
            cv2.putText(temp_frame, f'spot_{len(spots)-1}', (x-50, y-50-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.namedWindow('Define Spots')
    cv2.setMouseCallback('Define Spots', mouse_callback)
    
    while True:
        cv2.imshow('Define Spots', temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return spots

def process_frame(frame, model, spots, config):
    results = {}
    for spot_id, (x1, y1, x2, y2) in spots.items():
        spot_img = frame[y1:y2, x1:x2]
        spot_img = cv2.resize(spot_img, config.IMAGE_SIZE)
        spot_img = spot_img / 255.0
        
        prediction = model.model.predict(np.expand_dims(spot_img, axis=0), verbose=0)[0]
        status = 'occupied' if prediction > 0.5 else 'empty'
        results[spot_id] = {'status': status, 'confidence': float(prediction)}
    
    return results

def draw_results(frame, spots, results):
    for spot_id, (x1, y1, x2, y2) in spots.items():
        status = results[spot_id]['status']
        conf = results[spot_id]['confidence']
        color = (0, 0, 255) if status == 'occupied' else (0, 255, 0)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f'{spot_id}: {status} ({conf:.2f})',
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def main():
    config = Config()
    data_loader = DataLoader(config)
    model = ParkingModel(config)
    
    print("Training new model...")
    print("Preparing dataset...")
    train_ds, test_ds = data_loader.prepare_data()
    
    print("Training model...")
    model.fit(train_ds, epochs=config.EPOCHS, validation_data=test_ds)
    model.save_model()
    
    print("Starting video processing...")
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or video file path
    
    ret, frame = cap.read()
    if ret:
        spots = define_parking_spots(frame)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            results = process_frame(frame, model, spots, config)
            frame = draw_results(frame, spots, results)
            
            cv2.imshow('Parking Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()