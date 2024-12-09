import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
import logging
import math
import json
import re
import zipfile
import requests
import tempfile
from datetime import datetime
import easyocr
from ultralytics import YOLO
import supervision as sv
from PIL import Image
import time

class AutomatedParkingDashboard:
    def __init__(self):
        """Initialize the automated parking system with core components"""
        # Core configuration for the system
        self.config = {
            'VEHICLE_MODEL_PATH': 'models/yolov8n.pt',
            'DATA_DIR': 'data/PKLot',  # Keeping dataset directory for future use
            'LOG_DIR': 'logs',
            'RECEIPT_DIR': 'receipts',
            'TEMP_DIR': 'temp'
        }
        
        # Initialize system components
        self.setup_directories()
        self.setup_logging()
        
        # Business logic initialization
        self.RATE_PER_HOUR = 50
        self.GST_RATE = 0.18
        
        # Load core components
        self.initialize_models()
        self.initialize_database()
        self.initialize_ocr()
        self.initialize_dataset()  # Keep dataset initialization for future use

    def setup_directories(self):
        """Create necessary system directories"""
        for directory in self.config.values():
            if isinstance(directory, str) and not directory.endswith('.pt'):
                os.makedirs(directory, exist_ok=True)

    def setup_logging(self):
        """Configure system logging"""
        logging.basicConfig(
            filename=os.path.join(self.config['LOG_DIR'], 'parking_system.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def initialize_models(self):
        """Initialize vehicle detection model"""
        try:
            # Initialize YOLO for vehicle detection
            self.vehicle_model = YOLO('yolov8n.pt')
            logging.info("Vehicle detection model initialized successfully")
        except Exception as e:
            logging.error(f"Model initialization error: {str(e)}")
            st.error("Error initializing detection model")

    def initialize_database(self):
        """Initialize parking records database"""
        self.load_parking_records()

    def initialize_ocr(self):
        """Initialize OCR for license plate recognition"""
        try:
            self.reader = easyocr.Reader(['en'])
            logging.info("OCR initialized successfully")
        except Exception as e:
            logging.error(f"OCR initialization error: {str(e)}")
            st.error("Error initializing OCR system")

    def initialize_dataset(self):
        """Initialize and manage dataset"""
        try:
            dataset_path = self.config['DATA_DIR']
            download_path = os.path.expanduser("~/Downloads/pklot-dataset.zip")
            
            if os.path.exists(dataset_path) and os.listdir(dataset_path):
                self.dataset_info = self.analyze_dataset(dataset_path)
                return True
                
            st.warning("Dataset not found. Checking downloads...")
            
            if os.path.exists(download_path):
                st.info("Found downloaded dataset. Extracting...")
                self.extract_dataset(download_path, dataset_path)
            else:
                st.warning("""
                Dataset needs to be downloaded. Use this command:
                ```bash
                curl -L -o ~/Downloads/pklot-dataset.zip \
                    https://www.kaggle.com/api/v1/datasets/download/ammarnassanalhajali/pklot-dataset
                ```
                """)
                
                if st.button("Download Dataset"):
                    with st.spinner("Downloading dataset..."):
                        os.system(f'curl -L -o {download_path} \
                            https://www.kaggle.com/api/v1/datasets/download/ammarnassanalhajali/pklot-dataset')
                        if os.path.exists(download_path):
                            st.success("Download complete! Extracting...")
                            self.extract_dataset(download_path, dataset_path)
                        else:
                            st.error("Download failed. Please try manually.")
                            return False
            
            self.dataset_info = self.analyze_dataset(dataset_path)
            return True
            
        except Exception as e:
            logging.error(f"Dataset initialization error: {str(e)}")
            st.error(f"Error initializing dataset: {str(e)}")
            return False

    def extract_dataset(self, zip_path, extract_path):
        """Extract dataset with progress tracking"""
        try:
            with st.spinner("Extracting dataset..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    total_files = len(zip_ref.namelist())
                    progress = st.progress(0)
                    for i, file in enumerate(zip_ref.namelist()):
                        zip_ref.extract(file, extract_path)
                        progress.progress((i + 1) / total_files)
            st.success("Dataset extracted successfully!")
        except Exception as e:
            logging.error(f"Dataset extraction error: {str(e)}")
            st.error(f"Error extracting dataset: {str(e)}")

    def analyze_dataset(self, dataset_path):
        """Analyze dataset structure and content"""
        info = {
            'path': dataset_path,
            'total_images': 0,
            'occupied_spots': 0,
            'empty_spots': 0,
            'weather_conditions': set(),
            'cameras': set()
        }
        
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):
                    info['total_images'] += 1
                    if 'occupied' in file.lower():
                        info['occupied_spots'] += 1
                    else:
                        info['empty_spots'] += 1
                    
                    for weather in ['sunny', 'rainy', 'cloudy']:
                        if weather in root.lower():
                            info['weather_conditions'].add(weather)
                            
                    camera_match = re.search(r'camera\d+', root.lower())
                    if camera_match:
                        info['cameras'].add(camera_match.group())
        
        info['weather_conditions'] = sorted(list(info['weather_conditions']))
        info['cameras'] = sorted(list(info['cameras']))
        return info

    def process_video_feed(self, video_source):
        """
        Process video feed with improved visualization and detection logic
        """
        try:
            # Initialize video capture
            if isinstance(video_source, str) and video_source.isdigit():
                cap = cv2.VideoCapture(int(video_source))
            elif hasattr(video_source, 'read'):
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(video_source.read())
                tfile.flush()
                cap = cv2.VideoCapture(tfile.name)
            else:
                cap = cv2.VideoCapture(video_source)

            if not cap.isOpened():
                st.error("Failed to open video source")
                return

            # Create display elements
            frame_placeholder = st.empty()
            info_placeholder = st.empty()
            stop_button = st.button("Stop Processing")

            # Define visualization parameters
            BLUE = (255, 0, 0)  # BGR format
            GREEN = (0, 255, 0)
            RED = (0, 0, 255)
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    # Resize frame
                    height, width = frame.shape[:2]
                    target_width = 640
                    aspect_ratio = width / height
                    target_height = int(target_width / aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height))
                    
                    # Create display frame
                    display_frame = frame.copy()
                    
                    # Detect vehicles using YOLO
                    results = self.vehicle_model(frame)[0]
                    
                    # Process detections
                    for result in results.boxes.data:
                        x1, y1, x2, y2, conf, class_id = result
                        
                        # Check if detection is a vehicle (class_id 2)
                        if int(class_id) == 2 and conf > 0.5:
                            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                            
                            # Extract vehicle ROI
                            if y2 > y1 and x2 > x1:
                                vehicle_roi = frame[y1:y2, x1:x2]
                                
                                if vehicle_roi.size > 0:
                                    # Detect license plate
                                    plate_text = self.recognize_license_plate(vehicle_roi)
                                    
                                    if plate_text:
                                        # Draw detection visualization
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), GREEN, 2)
                                        
                                        # Add text background
                                        text = f"Plate: {plate_text}"
                                        (text_width, text_height), _ = cv2.getTextSize(
                                            text, FONT, 0.5, 1
                                        )
                                        cv2.rectangle(
                                            display_frame,
                                            (x1, y1 - text_height - 10),
                                            (x1 + text_width, y1),
                                            GREEN,
                                            -1
                                        )
                                        
                                        # Add text
                                        cv2.putText(
                                            display_frame,
                                            text,
                                            (x1, y1 - 5),
                                            FONT,
                                            0.5,
                                            BLUE,
                                            1,
                                            cv2.LINE_AA
                                        )
                                        
                                        # Process entry/exit
                                        self.handle_vehicle_entry_exit(plate_text)
                    
                    # Display frame
                    frame_placeholder.image(display_frame, channels="BGR")
                    
                    # Update status
                    status = self.get_current_status()
                    info_placeholder.markdown(f"""
                    ### Real-Time Parking Status
                    🚗 **Vehicles Currently Parked:** {status['vehicles_parked']}  
                    📊 **Total Vehicles Today:** {status['total_today']}  
                    💰 **Total Revenue:** ₹{status['revenue_today']:.2f}
                    """)
                    
                except Exception as e:
                    logging.error(f"Frame processing error: {str(e)}")
                    continue
                
                time.sleep(0.1)  # Control frame rate
            
            # Cleanup
            cap.release()
            if 'tfile' in locals():
                try:
                    os.unlink(tfile.name)
                except Exception as e:
                    logging.error(f"Error removing temporary file: {str(e)}")
                    
        except Exception as e:
            logging.error(f"Video processing error: {str(e)}")
            st.error("Error in video processing")

    def _check_recent_entry(self, plate_number, threshold_minutes=3):
        """
        Check if there's a recent entry for this vehicle to prevent duplicate entries.
        
        Args:
            plate_number (str): The vehicle's license plate number
            threshold_minutes (int): Minimum time between entries (default: 3 minutes)
            
        Returns:
            bool: True if a recent entry exists, False otherwise
        """
        try:
            current_time = datetime.now()
            recent_entries = self.parking_records[
                (self.parking_records['vehicle_number'] == plate_number) &
                (self.parking_records['status'] == 'parked')
            ]
            
            if not recent_entries.empty:
                last_entry = pd.to_datetime(recent_entries.iloc[-1]['entry_time'])
                time_diff = (current_time - last_entry).total_seconds() / 60
                
                # Return True if entry is too recent
                return time_diff < threshold_minutes
                
            return False
            
        except Exception as e:
            logging.error(f"Error checking recent entry: {str(e)}")
            return True  # Err on the side of caution

    def is_valid_license_format(self, text):
        """
        Validate if the text follows a general license plate format:
        - Starts with 2 letters (state code)
        - Followed by numbers and/or letters
        - Minimum total length of 4 characters
        """
        if not text or len(text) < 4:  # Minimum length check
            return False
            
        # Check if first two characters are letters
        if not text[:2].isalpha():
            return False
        
        # Check if remaining characters contain at least one number
        remaining = text[2:]
        if not any(c.isdigit() for c in remaining):
            return False
        
        # Additional check to ensure there's a reasonable mix of characters
        if len(remaining) < 2:  # At least 2 more characters after state code
            return False
        
        return True



    def is_vehicle_parked(self, plate_number):
        """
        Check if a vehicle is currently parked
        
        Args:
            plate_number (str): The license plate number to check
            
        Returns:
            bool: True if the vehicle is currently parked, False otherwise
        """
        return not self.parking_records[
            (self.parking_records['vehicle_number'] == plate_number) &
            (self.parking_records['status'] == 'parked')
        ].empty

    def recognize_license_plate(self, image):
        """
        Recognize and validate license plate text with improved preprocessing and flexible format checking.
        """
        try:
            # Convert to RGB for OCR
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Enhanced preprocessing pipeline for better OCR
            image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            image_gray = clahe.apply(image_gray)
            
            # Denoise the image
            image_denoised = cv2.fastNlMeansDenoising(image_gray)
            
            # Apply adaptive thresholding to get binary image
            image_thresh = cv2.adaptiveThreshold(
                image_denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )
            
            # Try OCR on both processed images for better results
            results = []
            for img in [image_thresh, image_rgb]:
                results.extend(self.reader.readtext(img))
            
            # Sort results by confidence
            results.sort(key=lambda x: x[2], reverse=True)
            
            for _, text, confidence in results:
                # Clean text and convert to uppercase
                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                
                # Check if the cleaned text is a valid license plate format
                if self.is_valid_license_format(cleaned_text) and confidence > 0.4:
                    return cleaned_text
            
            return None
            
        except Exception as e:
            logging.error(f"License plate recognition error: {str(e)}")
            return None
    def handle_vehicle_entry_exit(self, plate_number):
        """
        Handle vehicle entry/exit with improved timing logic and duplicate detection.
        
        This implementation enforces a minimum parking duration of 3 minutes to prevent
        false exits from momentary detections, while also maintaining a detection
        cooldown period to handle multiple readings of the same plate.
        """
        try:
            current_time = datetime.now()
            
            # Initialize detection tracking if not exists
            if not hasattr(self, 'recent_detections'):
                self.recent_detections = {}
            
            # Anti-flutter mechanism: Check if this plate was recently processed
            if plate_number in self.recent_detections:
                last_detection = self.recent_detections[plate_number]
                time_diff = (current_time - last_detection['time']).total_seconds()
                
                # If detected within last 30 seconds, just update timestamp and skip processing
                if time_diff < 30:
                    self.recent_detections[plate_number]['time'] = current_time
                    return
            
            # Update detection tracking
            self.recent_detections[plate_number] = {
                'time': current_time,
                'count': self.recent_detections.get(plate_number, {}).get('count', 0) + 1
            }
            
            # Check current vehicle status
            vehicle_record = self.parking_records[
                (self.parking_records['vehicle_number'] == plate_number) &
                (self.parking_records['status'] == 'parked')
            ]
            
            if vehicle_record.empty:
                # Entry logic - check for recent entries to prevent duplicates
                if not self._check_recent_entry(plate_number):
                    self.record_entry(plate_number)
                    logging.info(f"New vehicle entry recorded: {plate_number}")
            else:
                # Exit logic with improved minimum parking duration
                entry_time = pd.to_datetime(vehicle_record.iloc[-1]['entry_time'])
                parking_duration = (current_time - entry_time).total_seconds()
                
                # Minimum parking duration check (3 minutes = 180 seconds)
                if parking_duration > 180:  # Increased from previous 300 seconds
                    self.record_exit(plate_number)
                    self.generate_receipt(
                        plate_number,
                        entry_time,
                        current_time,
                        parking_duration / 3600,  # Convert to hours for billing
                        math.ceil(parking_duration / 3600 * self.RATE_PER_HOUR)
                    )
                    logging.info(f"Vehicle exit processed: {plate_number} - Duration: {parking_duration/60:.1f} minutes")
                else:
                    logging.debug(
                        f"Skipping early exit for {plate_number} - "
                        f"Only {parking_duration/60:.1f} minutes since entry"
                    )
                    
        except Exception as e:
            logging.error(f"Entry/exit handling error: {str(e)}")

    def record_entry(self, vehicle_number):
        """Record vehicle entry"""
        try:
            entry_time = datetime.now()
            new_record = pd.DataFrame([{
                'vehicle_number': vehicle_number,
                'entry_time': entry_time,
                'exit_time': None,
                'duration': None,
                'charges': None,
                'status': 'parked'
            }])
            self.parking_records = pd.concat([self.parking_records, new_record], 
                                           ignore_index=True)
            self.save_records()
            st.success(f"Entry recorded: {vehicle_number}")
        except Exception as e:
            logging.error(f"Entry recording error: {str(e)}")

    def record_exit(self, vehicle_number):
        """
        Record vehicle exit with improved validation and logging
        """
        try:
            exit_time = datetime.now()
            idx = self.parking_records[
                (self.parking_records['vehicle_number'] == vehicle_number) & 
                (self.parking_records['status'] == 'parked')
            ].index[0]
            
            entry_time = pd.to_datetime(self.parking_records.loc[idx, 'entry_time'])
            duration = (exit_time - entry_time).total_seconds() / 3600  # in hours
            
            # Calculate charges
            charges = math.ceil(duration * self.RATE_PER_HOUR)
            
            # Update record
            self.parking_records.loc[idx, 'exit_time'] = exit_time
            self.parking_records.loc[idx, 'duration'] = duration
            self.parking_records.loc[idx, 'charges'] = charges
            self.parking_records.loc[idx, 'status'] = 'exited'
            
            # Save records
            self.save_records()
            
            # Display success message with duration
            duration_minutes = duration * 60
            st.success(
                f"Exit recorded: {vehicle_number}\n"
                f"Parking duration: {int(duration_minutes)} minutes"
            )
            
            # Log the exit
            logging.info(
                f"Vehicle exit: {vehicle_number}, "
                f"Duration: {duration_minutes:.1f} minutes, "
                f"Charges: ₹{charges:.2f}"
            )
                
        except Exception as e:
            logging.error(f"Exit recording error: {str(e)}")

    def generate_receipt(self, vehicle_number, entry_time, exit_time, duration, charges):
        """Generate detailed parking receipt"""
        try:
            gst = charges * self.GST_RATE
            total = charges + gst
            
            receipt_html = f"""
            <div style="font-family: Arial; padding: 20px; border: 2px solid #333;">
                <h2 style="text-align: center;">PARKING RECEIPT</h2>
                <p><strong>Vehicle Number:</strong> {vehicle_number}</p>
                <p><strong>Entry Time:</strong> {entry_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Exit Time:</strong> {exit_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Duration:</strong> {duration:.2f} hours</p>
                <p><strong>Base Charge:</strong> ₹{charges:.2f}</p>
                <p><strong>GST (18%):</strong> ₹{gst:.2f}</p>
                <h3 style="text-align: center;">Total Amount: ₹{total:.2f}</h3>
            </div>
            """
            
            st.markdown(receipt_html, unsafe_allow_html=True)
            
            # Save receipt
            # Continuing from generate_receipt method...
            receipt_path = os.path.join(
                self.config['RECEIPT_DIR'],
                f"receipt_{vehicle_number}_{exit_time.strftime('%Y%m%d%H%M%S')}.html"
            )
            with open(receipt_path, 'w') as f:
                f.write(receipt_html)
                
        except Exception as e:
            logging.error(f"Receipt generation error: {str(e)}")

    def load_parking_records(self):
        """Load or create parking records from CSV file"""
        try:
            if os.path.exists('parking_records.csv'):
                self.parking_records = pd.read_csv('parking_records.csv')
                # Convert time columns to datetime
                self.parking_records['entry_time'] = pd.to_datetime(
                    self.parking_records['entry_time'],
                    errors='coerce'
                )
                self.parking_records['exit_time'] = pd.to_datetime(
                    self.parking_records['exit_time'],
                    errors='coerce'
                )
            else:
                # Create new DataFrame if no records exist
                self.parking_records = pd.DataFrame(columns=[
                    'vehicle_number', 'entry_time', 'exit_time', 
                    'duration', 'charges', 'status'
                ])
        except Exception as e:
            logging.error(f"Error loading parking records: {str(e)}")
            st.error("Error loading parking records")
            # Create empty DataFrame as fallback
            self.parking_records = pd.DataFrame(columns=[
                'vehicle_number', 'entry_time', 'exit_time', 
                'duration', 'charges', 'status'
            ])

    def save_records(self):
        """Save parking records to CSV file"""
        try:
            self.parking_records.to_csv('parking_records.csv', index=False)
            logging.info("Parking records saved successfully")
        except Exception as e:
            logging.error(f"Error saving parking records: {str(e)}")
            st.error("Error saving parking records")

    def get_current_status(self):
        """Get real-time parking status and statistics"""
        try:
            # Get currently parked vehicles
            current = self.parking_records[self.parking_records['status'] == 'parked']
            
            # Get today's date and records
            today_date = pd.to_datetime('today').normalize()
            today_records = self.parking_records[
                pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == today_date
            ]
            
            # Calculate revenue for today
            revenue = today_records[
                today_records['status'] == 'exited'
            ]['charges'].sum()
            
            return {
                'vehicles_parked': len(current),
                'total_today': len(today_records),
                'revenue_today': revenue
            }
        except Exception as e:
            logging.error(f"Status calculation error: {str(e)}")
            return {'vehicles_parked': 0, 'total_today': 0, 'revenue_today': 0}

    def show_parking_records(self):
        """Display comprehensive parking records interface"""
        st.subheader("Parking Records")
        
        try:
            # Current statistics display
            current = self.parking_records[self.parking_records['status'] == 'parked']
            today_date = pd.to_datetime('today').normalize()
            
            # Display metrics in columns
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Currently Parked", len(current))
            with col2:
                total_today = len(self.parking_records[
                    pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == today_date
                ])
                st.metric("Total Today", total_today)
            with col3:
                if total_today > 0:
                    occupancy = (len(current) / total_today) * 100
                    st.metric("Occupancy Rate", f"{occupancy:.1f}%")
            
            # Currently parked vehicles table
            st.subheader("Currently Parked Vehicles")
            if not current.empty:
                # Calculate duration for currently parked vehicles
                current['current_duration'] = (datetime.now() - 
                    pd.to_datetime(current['entry_time'])).dt.total_seconds() / 3600
                display_current = current[['vehicle_number', 'entry_time', 'current_duration']]
                display_current = display_current.rename(
                    columns={'current_duration': 'Duration (hours)'}
                )
                st.dataframe(display_current.style.format({
                    'Duration (hours)': '{:.2f}'
                }))
            else:
                st.info("No vehicles currently parked")
            
            # Historical records with filtering
            st.subheader("Historical Records")
            col1, col2 = st.columns(2)
            with col1:
                date_filter = st.date_input("Select Date", datetime.now().date())
            with col2:
                status_filter = st.selectbox("Status", ["All", "Parked", "Exited"])
            
            if date_filter:
                filter_date = pd.to_datetime(date_filter)
                historical = self.parking_records[
                    pd.to_datetime(self.parking_records['entry_time']).dt.normalize() == filter_date
                ]
                
                if status_filter != "All":
                    historical = historical[
                        historical['status'].str.lower() == status_filter.lower()
                    ]
                
                if not historical.empty:
                    st.dataframe(historical.style.format({
                        'duration': '{:.2f}',
                        'charges': '₹{:.2f}'
                    }))
                    
                    # Daily statistics
                    st.subheader("Daily Statistics")
                    completed_visits = historical[historical['status'] == 'exited']
                    daily_stats = {
                        'Total Vehicles': len(historical),
                        'Total Revenue': f"₹{completed_visits['charges'].sum():.2f}",
                        'Average Duration': f"{completed_visits['duration'].mean():.2f} hours",
                        'Average Charge': f"₹{completed_visits['charges'].mean():.2f}"
                    }
                    st.json(daily_stats)
                else:
                    st.info(f"No records found for {date_filter}")
                    
        except Exception as e:
            logging.error(f"Error displaying parking records: {str(e)}")
            st.error("Error loading parking records")

    def show_system_status(self):
        """Display system status and analytics dashboard"""
        st.header("System Status")
        
        try:
            # System metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                status = self.get_current_status()
                st.metric("Active Vehicles", status['vehicles_parked'])
            with col2:
                st.metric("Total Today", status['total_today'])
            with col3:
                st.metric("Revenue Today", f"₹{status['revenue_today']:.2f}")
            
            # System components status
            st.subheader("Component Status")
            components = {
                "Vehicle Detection": self.vehicle_model is not None,
                "OCR System": hasattr(self, 'reader'),
                "Database": hasattr(self, 'parking_records'),
                "Dataset": hasattr(self, 'dataset_info')
            }
            
            # Display component status with colored indicators
            for component, status in components.items():
                st.write(f"{component}: {'✅ Active' if status else '❌ Inactive'}")
            
            # System logs display
            log_file = os.path.join(self.config['LOG_DIR'], 'parking_system.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    recent_logs = f.readlines()[-10:]  # Show last 10 log entries
                    if recent_logs:
                        st.subheader("Recent System Logs")
                        for log in recent_logs:
                            st.text(log.strip())
                            
            # Display dataset statistics if available
            if hasattr(self, 'dataset_info'):
                st.subheader("Dataset Information")
                st.json({
                    "Total Images": self.dataset_info['total_images'],
                    "Occupied Spots": self.dataset_info['occupied_spots'],
                    "Empty Spots": self.dataset_info['empty_spots'],
                    "Weather Conditions": list(self.dataset_info['weather_conditions']),
                    "Cameras": list(self.dataset_info['cameras'])
                })
            
        except Exception as e:
            logging.error(f"Error displaying system status: {str(e)}")
            st.error("Error loading system status")

    def run(self):
        """Main dashboard entry point"""
        st.title("Automated Parking Management System")
        
        menu = st.sidebar.selectbox(
            "Select Option",
            ["Live Detection", "Parking Records", "System Status"]
        )
        
        if menu == "Live Detection":
            st.header("Live Vehicle Detection")
            
            # Camera selection and controls
            source = st.radio("Select Source", ["Camera", "Upload Video"])
            
            if source == "Camera":
                camera_id = st.selectbox(
                    "Select Camera", 
                    ["0", "1", "2"],
                    help="Choose camera device ID (0 is usually the built-in camera)"
                )
                if st.button("Start Camera", key="start_camera"):
                    self.process_video_feed(camera_id)
            else:
                video_file = st.file_uploader(
                    "Upload Video",
                    type=['mp4', 'avi', 'mov'],
                    help="Upload a video file for vehicle detection"
                )
                if video_file:
                    self.process_video_feed(video_file)
                    
        elif menu == "Parking Records":
            self.show_parking_records()
        else:
            self.show_system_status()

if __name__ == "__main__":
    st.set_page_config(
        page_title="Automated Parking System",
        page_icon="🅿️",
        layout="wide"
    )
    
    # Initialize and run the dashboard
    dashboard = AutomatedParkingDashboard()
    dashboard.run()
