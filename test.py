import argparse

from dataloader import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from model import Transformer

from helpers import *
import sys
import glob
import cv2
from natsort import natsorted

def load_model(model_path, device):
    """
    Load a trained model for inference.

    Args:
        model_path (str): Path to the saved model file (e.g., 'best_train.pth').
        device (str): Device to use for inference ('cpu' or 'cuda').

    Returns:
        model: Loaded model ready for inference.
    """
    device = torch.device(device)
    model = Transformer().double().to(device)  # Replace 'Transformer' with your model class name
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model


model_path = 'save_model/best_train_1000.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loaded_model = load_model(model_path, device)
print('Model Successfully loaded')
test_csv = 'output.csv'
training_length = 12
forecast_window = 12

image_width = 640
image_height = 480

# Assuming 'SensorDataset' class and 'test_csv' are defined
test_dataset = SensorDataset(csv_name=test_csv, root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# Set model to evaluation mode


# Define the criterion (MSE in this case)
criterion = torch.nn.MSELoss()

# Initialize variables to store total loss and total number of samples
total_loss = 0
total_samples = 0

print('Inference dataset loaded successfully')



# Iterate through the test dataset
for index_in, index_tar, _input, target, sensor_number in test_dataloader:
    # starting from 1 so that src matches with target, but has same length as when training
    # src = _input.permute(1,0,2).double().to(device)[1:, :, :] # 47, 1, 7: t1 -- t47
    # print(sensor_number)
    src = _input.permute(1,0,2).double().to(device)[1:, :, :] #same length as training
    target = target.permute(1,0,2).double().to(device) # t48 - t59

    next_input_model = src[:src.shape[0]]
    all_predictions = []
    all_target = []

    all_input = []

    for i in range(target.shape[0]):
        with torch.no_grad():
            prediction = loaded_model(next_input_model, device)  # Get model prediction
            next_input_model = torch.cat((next_input_model, target[i:i+1]), dim=0)
            next_input_model = next_input_model[1:]
            all_predictions.append(prediction)
            all_input.append(next_input_model)

    # print(all_predictions)
    ### Get the starting frame number
    # print(sensor_number[0])
    file_name = sensor_number[0]
    _part = file_name.split('.')[0][-1]
    part = int(_part)
    frame_number_directory ='Data/inference/frame_numbers/'
    frames_directory = 'Data/inference/frames/'
    frame_number_file =  frame_number_directory + file_name


    frames_folder_name = file_name.split('_object')[0]


    length = training_length+ forecast_window+1

    with open(frame_number_file, "r") as file:
        lines = file.readlines()
        numbers = lines[0].split()
        first_number = int(numbers[0])
    frame_number = first_number + (length)*(part-1)


    image_directory = frames_directory + frames_folder_name
    images = natsorted(glob.glob(os.path.join(image_directory,'*.jpg')))

    for fr in range(target.shape[0]):

        img_file = images[fr+ frame_number]
        print(img_file)
        pred_coord = all_predictions[fr]

        image = cv2.imread(img_file)
        # Create a list to store the points
        points = []

        for coord in pred_coord:
            # y, x, w, h = coord[0]
            # top_left_x = int((x ) * image_height) ## image_width
            # top_left_y = int((y ) * image_width) ##image_height
            # bottom_right_x = int((x+h)*image_height)
            # bottom_right_y = int((y+w)*image_width)

            # true_center_x = top_left_x +(bottom_right_x-top_left_x)//2
            # true_center_y = top_left_y +(bottom_right_y-top_left_y)//2
            y_norm, x_norm, w_norm, h_norm = coord[0]

            # Convert normalized coordinates to pixel values
            y1 = int(y_norm * image_height)
            x1 = int(x_norm * image_width)
            w = int(w_norm * image_width)
            h = int(h_norm * image_height)

            #calculate the bottom-right corner of the bounding box
            x2 = x1 + w
            y2 = y1 + h

            # Calculate the center point of the bounding box
            cx= x1 + w//2
            cy= y1 + h//2
            points.append((cx, cy))
            color = (0, 255, 0)  # green color
            thickness = 2
            radius = 3
            image = cv2.circle(image, (cx, cy), radius, color, thickness)
            # image = cv2.circle(image, (true_center_y, true_center_x), radius, color, thickness)

        # Draw lines between the points
        color = (0, 255, 0)  # blue color
        thickness = 2
        for i in range(len(points)-1):
            image = cv2.line(image, points[i], points[i+1], color, thickness)


        target_coor = all_input[fr]
        target_points = []

        for coord in target_coor:
            y_norm, x_norm, w_norm, h_norm = coord[0]

            # Convert normalized coordinates to pixel values
            y1 = int(y_norm * image_height)
            x1 = int(x_norm * image_width)
            w = int(w_norm * image_width)
            h = int(h_norm * image_height)

            #calculate the bottom-right corner of the bounding box
            x2 = x1 + w
            y2 = y1 + h

            # Calculate the center point of the bounding box
            cx= x1 + w//2
            cy= y1 + h//2
            target_points.append((cx, cy))

            # top_left_x = int((x) * image_height)
            # center_y = int((y) * image_width)
            # bottom_right_x = int((x+h)*image_height)
            # bottom_right_y = int((y+w)*image_width)

            # true_center_x = top_left_x +(bottom_right_x-top_left_x)//2
            # true_center_y = top_left_y +(bottom_right_y-top_left_y)//2
            # target_points.append((true_center_x, true_center_y))
            target_color = (255, 0, 0)  # blue color
            thickness = 2
            radius = 3
            image = cv2.circle(image, (cx, cy), radius, target_color, thickness)
            image = cv2.rectangle(image, (x1,y1), (x2,y2), target_color, thickness)

            # image = cv2.circle(image, (true_center_y, true_center_x), radius, target_color, thickness)
            # image= cv2.rectangle(image, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), target_color, thickness)

        # Draw lines between the points
        target_color = (255, 0, 0)  # Green color
        target_thickness = 2
        for i in range(len(target_points)-1):
            image = cv2.line(image, target_points[i], target_points[i+1], target_color, target_thickness)


        cv2.imwrite(img_file,image)
















    # for i in range(forecast_window-1):
    #     with torch.no_grad():  # No need to compute gradients during testing
    #         prediction = loaded_model(next_input_model, device)  # Get model prediction
    #         print('prediction')
    #         print(prediction.shape)
    #         if all_predictions == []:
    #             all_predictions = prediction # 47,1,1: t2' - t48'
    #         else:
    #             all_predictions = torch.cat((all_predictions, prediction[-1,:,:].unsqueeze(0))) # 47+,1,1: t2' - t48', t49', t50'

    #         # pos_encoding_old_vals = src[i+1:, :, 1:] # 46, 1, 6, pop positional encoding first value: t2 -- t47
    #         # pos_encoding_new_val = target[i + 1, :, 1:].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
    #         pos_encoding_old_vals = src[i+1:, :, :] # 46, 1, 6, pop positional encoding first value: t2 -- t47
    #         pos_encoding_new_val = target[i + 1, :, :].unsqueeze(1) # 1, 1, 6, append positional encoding of last predicted value: t48
    #         pos_encodings = torch.cat((pos_encoding_old_vals, pos_encoding_new_val)) # 47, 1, 6 positional encodings matched with prediction: t2 -- t48
    #         # print(pos_encoding_old_vals.shape)
    #         # print(pos_encoding_new_val.shape)
    #         # print(pos_encodings.shape)
    #         # print('==========')

    #         next_input_model = torch.cat((src[i+1:, :, :].unsqueeze(-1), prediction[-1,:,:].unsqueeze(0))) #t2 -- t47, t48'

    #         next_input_model = torch.cat((next_input_model, pos_encodings), dim = 2) # 47, 1, 7 input for next round


    # true = torch.cat((src[1:,:,0],target[:-1,:,0]))
    # # loss = criterion(true, all_predictions[:,:,0])
    # print('True: ')
    # print(true)
    # print('predictions: ')
    # print(all_predictions[:,:,0])




#         # Calculate loss
#         loss = criterion(prediction, target)
#         total_loss += loss.item()

#     total_samples += 1

# # Calculate average loss
# average_loss = total_loss / total_samples

# print(f'Test Loss: {average_loss:.4f}')
