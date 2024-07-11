import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from model import AdvancedGRUModel as GRUModel
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from natsort import natsorted
import glob
import sys
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(image_path):
    """Load and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((480, 640)),  # Maintain aspect ratio
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

def denormalize_bounding_boxes(normalized_boxes, image_width=640, image_height=480):
    """Convert normalized bounding boxes to pixel coordinates."""
    denorm_boxes = normalized_boxes.copy()
    denorm_boxes[:, 0] *= image_width   # x_center
    denorm_boxes[:, 1] *= image_height  # y_center
    denorm_boxes[:, 2] *= image_width   # width
    denorm_boxes[:, 3] *= image_height  # height
    return denorm_boxes

def run_inference(model, image_folder, sequence_length=12):
    """Run inference on a sequence of images."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    all_predictions = []

    with torch.no_grad():
        for i in range(0, len(image_files), sequence_length):
            sequence = image_files[i:i+sequence_length]
            
            # Load and preprocess the sequence of images
            input_sequence = torch.cat([load_image(os.path.join(image_folder, img)) for img in sequence])
            input_sequence = input_sequence.to(device)

            # Reshape input to (batch_size, sequence_length, input_size)
            batch_size, channels, height, width = input_sequence.shape
            input_reshaped = input_sequence.view(1, batch_size, channels * height * width)

            # Generate predictions
            predictions = model(input_reshaped)
            all_predictions.append(predictions.cpu().numpy())

    return np.concatenate(all_predictions, axis=1)

def main():
    model_path = 'snapshot/best_train_model.pth'
    data_path = 'Data/Egensevej-1_2__inference_train.csv'
    frames_directory = 'Data/inference/frames/Egensevej-1_2'
    image_width = 640
    image_height = 480


    model = GRUModel()  # Initialize your model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    df = pd.read_csv(data_path)
    reindexed_id = df['reindexed_id'].unique().tolist()
    # reindexed_id = [24]
    images= natsorted(glob.glob(os.path.join(frames_directory, '*.jpg')))

    for idx in reindexed_id:
        data = df[df["reindexed_id"]==idx]
        file_name = str(data["File_Name"].iloc[0])
        frames = data["frame"].values
        sensor_number = data["reindexed_id"].values[0]
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        input_columns = ['x', 'y', 'w', 'h'] + feature_columns
        target_columns = ['x', 'y', 'w', 'h']
        start = 0
        input_feat = torch.tensor(data[input_columns][start : start + 25].values, dtype=torch.float32)
        input_feat = input_feat.unsqueeze(0)
        with torch.no_grad():
            for tt in range(12):
                bb_input = input_feat[:,tt,:4].unsqueeze(0).to(device)
                vit_feat = input_feat[:,tt,4:].unsqueeze(0).to(device)
                y = input_feat[:,tt+1:tt+13,:4].to(device) #total length 25

                output = model(bb_input,vit_feat)
                # x = input_feat[:,tt,:].unsqueeze(0).to(device)
                # y = input_feat[:,tt+1:tt+13,:4].to(device) #total length 25
                # output = model(x)
                

                img_file = images[frames[tt]]
                print(img_file)
                image = cv2.imread(img_file)
                target_points = []
                pred_points = []
                for pts in range(12): # 12 future frames
                    tar_bbox = y[:,pts]
                    
                    tar_x_min = int(tar_bbox[0,0].item()*image_width)
                    tar_y_min = int(tar_bbox[0,1].item()*image_height)
                    tar_width = int(tar_bbox[0,2].item()*image_width)
                    tar_height = int(tar_bbox[0,3].item()*image_height)
                    # Calculate the bottom-right corner
                    tar_x_max = tar_x_min + tar_width
                    tar_y_max = tar_y_min + tar_height
                    # calculate the center coordinates
                    tar_center_x = tar_x_min + tar_width // 2
                    tar_center_y = tar_y_min + tar_height // 2

                    
                    target_points.append((tar_center_x, tar_center_y))
                    
                    # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                    cv2.circle(image, (tar_center_x, tar_center_y), 2, (0, 255, 0), 2)

                    pre_bbox = output[:,pts]
                    pre_x_min = int(pre_bbox[0,0].item()*image_width)
                    pre_y_min = int(pre_bbox[0,1].item()*image_height)
                    pre_width = int(pre_bbox[0,2].item()*image_width)
                    pre_height = int(pre_bbox[0,3].item()*image_height)
                    # Calculate the bottom-right corner
                    pre_x_max = pre_x_min + pre_width
                    pre_y_max = pre_y_min + pre_height
                    # calculate the center coordinates
                    pre_center_x = pre_x_min + pre_width // 2
                    pre_center_y = pre_y_min + pre_height // 2
                    pred_points.append((pre_center_x, pre_center_y))
                    cv2.circle(image, (pre_center_x, pre_center_y), 2, (255, 0, 0), 2)
                for lp in range(len(target_points)-1):
                    cv2.line(image, target_points[lp], target_points[lp+1], (0, 255, 0), 2)
                    cv2.line(image, pred_points[lp], pred_points[lp+1], (255, 0, 0), 2)

                cv2.imwrite(img_file,image)


main()





    
