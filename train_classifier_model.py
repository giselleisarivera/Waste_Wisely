import os
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from itertools import product

class WasteDataset(Dataset):
    def __init__(self, txt_file_path, img_dir):
        with open(txt_file_path, 'r') as f:
            self.data = [line.strip().split() for line in f.readlines()]
            
        self.img_dir = img_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        img_path = os.path.join(self.img_dir, row[0])
        image = Image.open(img_path).convert('RGB')

        # double check file format w alan
        label = torch.tensor([
            int(row[1]),  # class id
            float(row[2]),  # x center
            float(row[3]),  # y center
            float(row[4]),  # width
            float(row[5])   # height
        ])

        return image, label

def train_yolo(param_grid):

    # combinations of hyperparameters
    params = list(product(*param_grid.values()))
    param_names = list(param_grid.keys())
    best_result = None
    best_params = None

    # search over the parameter grid
    for param_set in params:
        lr, epochs, batch_size, img_size = param_set
        model = YOLO('yolov8n.pt')
    
        result = model.train(
                    data='./data/data.yaml',
                    epochs=epochs,
                    imgsz=img_size,
                    batch=batch_size,
                    lr0=lr,
                    workers=4,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    save=True,                  # save model after training
                    save_period=1,              # save after every epoch
                    project='runs/train',     # directory to save logs
                    name='waste_detection'      # subdirectory name
                )
        print(f"Result:  {result}") # comment after debugging ; to double check if 'metrics/mAP50(B)' is right key to access map score
        print(f"Training with params: {param_set}")
        print(f"mAP Score: {result['metrics/mAP50(B)']}")
        # update params w/ better mAP score  (higher is beter)
        if best_result is None or result['metrics/mAP50(B)'] > best_result['metrics/mAP50(B)']:
            best_result = result
            best_params = param_set
            # save the model if it's the best so far
            # MIGHT NEED TO DEBUG HOW MODEL IS BEING SAVED
            # DOUBLE CHECK FILE PATH 
            model_path = f'./runs/train/waste_detection/weights/best.pt'
            best_model_path = f'./model_results/best_model.pt'
            os.makedirs('./model_results', exist_ok=True)
            if os.path.exists(model_path):
                os.rename(model_path, best_model_path)  

        

    # save model w/ best parameters?? 

    print(f"\nBest Params: {dict(zip(param_names, best_params))}")
    print(f"Best mAP: {best_result['metrics/mAP50(B)']:.4f}")

if __name__ == "__main__":
    param_grid = {
        'learning_rate': [1e-3, 1e-4],
        'epochs': [10, 20],
        'batch_size': [8, 16],
        'img_size': [640, 512],
    }
    train_yolo(param_grid)

