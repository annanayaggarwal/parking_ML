
import tensorflow as tf
import numpy as np
import cv2
import os
import tarfile
import requests
from tqdm import tqdm
import pandas as pd

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.batch_size = self.config.BATCH_SIZE
        self.dataset_url = "https://github.com/fabiocarrara/deep-parking/releases/download/archive/CNR-EXT_FULL_IMAGE_1000x750.tar"
        
    def download_dataset(self):
        dataset_path = os.path.join(self.config.DATA_DIR, 'dataset.tar')
        print("Downloading CNR-EXT dataset...")
        
        response = requests.get(self.dataset_url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dataset_path, 'wb') as f:
            for data in tqdm(response.iter_content(chunk_size=1024)):
                f.write(data)
        
        print("Extracting dataset...")
        with tarfile.open(dataset_path, 'r') as tar:
            tar.extractall(self.config.DATA_DIR)
        
        os.remove(dataset_path)

    def process_raw_data(self):
        processed_file = os.path.join(self.config.DATA_DIR, 'processed_data.npz')
        base_dir = os.path.join(self.config.DATA_DIR, 'FULL_IMAGE_1000x750')
        batch_size = 500  # Process in smaller batches
        
        if not os.path.exists(base_dir):
            self.download_dataset()

        images = []
        labels = []
        count = 0
        
        for weather in ['SUNNY', 'OVERCAST', 'RAINY']:
            weather_dir = os.path.join(base_dir, weather)
            for date_dir in sorted(os.listdir(weather_dir)):
                date_path = os.path.join(weather_dir, date_dir)
                
                for cam_dir in sorted(os.listdir(date_path)):
                    cam_path = os.path.join(date_path, cam_dir)
                    if not os.path.isdir(cam_path):
                        continue
                    
                    batch_images = []
                    batch_labels = []
                    
                    for img_file in sorted(os.listdir(cam_path)):
                        if not img_file.endswith('.jpg'):
                            continue
                            
                        img_path = os.path.join(cam_path, img_file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.resize(img, self.config.IMAGE_SIZE)
                                batch_images.append(img)
                                batch_labels.append(1 if 'occupied' in img_file.lower() else 0)
                                count += 1
                                
                                if len(batch_images) >= batch_size:
                                    # Save batch
                                    X_batch = np.array(batch_images, dtype=np.float32) / 255.0
                                    y_batch = np.array(batch_labels, dtype=np.int32)
                                    
                                    if not os.path.exists(processed_file):
                                        np.savez(processed_file, images=X_batch, labels=y_batch)
                                    else:
                                        data = np.load(processed_file)
                                        X_prev = data['images']
                                        y_prev = data['labels']
                                        X_new = np.concatenate([X_prev, X_batch])
                                        y_new = np.concatenate([y_prev, y_batch])
                                        np.savez(processed_file, images=X_new, labels=y_new)
                                    
                                    batch_images = []
                                    batch_labels = []
                                    print(f"Processed {count} images")
                                    
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue
        
        # Save remaining batch
        if batch_images:
            X_batch = np.array(batch_images, dtype=np.float32) / 255.0
            y_batch = np.array(batch_labels, dtype=np.int32)
            if not os.path.exists(processed_file):
                np.savez(processed_file, images=X_batch, labels=y_batch)
            else:
                data = np.load(processed_file)
                X_prev = data['images']
                y_prev = data['labels']
                X_new = np.concatenate([X_prev, X_batch])
                y_new = np.concatenate([y_prev, y_batch])
                np.savez(processed_file, images=X_new, labels=y_new)
        
        # Load final processed data
        data = np.load(processed_file)
        return data['images'], data['labels']

    def prepare_data(self):
        X, y = self.process_raw_data()
        train_size = int(0.8 * len(X))
        indices = np.random.permutation(len(X))
        
        train_ds = tf.data.Dataset.from_tensor_slices(
            (X[indices[:train_size]], y[indices[:train_size]])
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        test_ds = tf.data.Dataset.from_tensor_slices(
            (X[indices[train_size:]], y[indices[train_size:]])
        ).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, test_ds
