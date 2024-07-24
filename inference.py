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


def main(model_path, data_path, frames_directory, output_frames_directory, image_width,image_height): 


    model = GRUModel()  # Initialize your model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    df = pd.read_csv(data_path)
    reindexed_id = df['reindexed_id'].unique().tolist()
    # reindexed_id = [24]
    images= natsorted(glob.glob(os.path.join(frames_directory, '*.jpg')))

    for idx in reindexed_id:
        print(idx)
        data = df[df["reindexed_id"]==idx]
        # print(data)
        
        file_name = str(data["File_Name"].iloc[0])
        frames = data["frame"].values
        # print(frames)

        sensor_number = data["reindexed_id"].values[0]
        feature_columns = [col for col in df.columns if col.startswith('feature_')]
        input_columns = ['x', 'y', 'w', 'h'] + feature_columns
        target_columns = ['x', 'y', 'w', 'h']
        start = 0
        input_feat = torch.tensor(data[input_columns][start : start + len(data)].values, dtype=torch.float32)
        input_feat = input_feat.unsqueeze(0)
        # print(input_feat.shape)

        with torch.no_grad():
            for tt in range(input_feat.shape[1]):
                bb_input = input_feat[:, tt, :4].unsqueeze(0).to(device)
                vit_feat = input_feat[:, tt, 4:].unsqueeze(0).to(device)
                # Check if there are enough timesteps remaining for y
                if tt + 13 <= input_feat.shape[1]:
                    y = input_feat[:, tt + 1: tt + 13, :4].to(device)
                    mask = torch.ones_like(y, dtype=torch.bool).to(device)  # All ones, no padding
                else:
                    # Pad the remaining timesteps with zeros (or some other padding value)
                    remaining_length = input_feat.shape[1] - (tt + 1)
                    y = input_feat[:, tt + 1:, :4].to(device)
                    padding = torch.zeros((1, 12 - remaining_length, 4)).to(device)
                    y = torch.cat((y, padding), dim=1)
                    mask = torch.cat((torch.ones((1, remaining_length, 4)).to(device),
                            torch.zeros((1, 12 - remaining_length, 4)).to(device)), dim=1)
                output = model(bb_input, vit_feat)
                masked_output = output * mask
                masked_y = y * mask
                img_file = images[frames[tt]]
                img_name = img_file.split('/')[-1]                
                output_img_file = output_frames_directory+'/' + img_name
                image = cv2.imread(img_file)
                
                target_points = []
                pred_points = []
                for pts in range(12): # 12 future frames
                    if mask[0, pts, 0].item() == 0:  # Skip padding
                        continue
                    tar_bbox = masked_y[:,pts]
                    
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

                    pre_bbox = masked_output[:,pts]
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

model_path = 'snapshot/best_test_model.pth'
input_image_folder = 'Data/inference/all_videos/'
# input_data_path = 'Data/inference/train_inference'

output_main_folder = 'Data/inference/all_videos/'


suffix = '__inference_test.csv'

# data_path = 'Data/Egensevej-1_2__inference_train.csv'
# frames_directory = 'Data/inference/frames/Egensevej-1_2'
image_width = 640
image_height = 480

image_folder_list = os.listdir(input_image_folder)

for folders in image_folder_list:
    data_path = 'Data/inference/test_inference/'+folders + suffix
    if not os.path.exists(data_path):
        print('couldnt find : ' ,data_path)
        continue
    frames_directory = input_image_folder + folders
    output_frames_directory = output_main_folder + folders
    if not os.path.exists(output_frames_directory):
        os.makedirs(output_frames_directory)
    main(model_path, data_path, frames_directory, output_frames_directory,image_width,image_height)
    print('Done : ', folders)
    


# main()





    
