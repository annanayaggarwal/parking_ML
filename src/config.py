import os

class Config:
    def __init__(self):
        self.BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.DATA_DIR = os.path.join(self.BASE_DIR, 'data')
        self.MODEL_DIR = os.path.join(self.BASE_DIR, 'output', 'models')
        self.RESULTS_DIR = os.path.join(self.BASE_DIR, 'output', 'results')
        
        self.IMAGE_SIZE = (224, 224)
        self.BATCH_SIZE = 32
        self.EPOCHS = 100
        self.LEARNING_RATE = 0.001
        
        # Create directories
        for dir_path in [self.DATA_DIR, self.MODEL_DIR, self.RESULTS_DIR]:
            os.makedirs(dir_path, exist_ok=True)