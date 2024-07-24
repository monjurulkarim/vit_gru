from model import AdvancedGRUModel as GRUModel
import torch.nn as nn
import torch, math

import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import SensorDataset
from tqdm import tqdm
import glob
import numpy as np
import time
from datetime import date
import csv
from utils import final_displacement_error, average_displacement_error, final_intersection_over_union

os.environ['CUDA_VISIBLE_DEVICES']= '1'

device = ("cuda" if torch.cuda.is_available() else "cpu")

seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

learning_rate = 0.0001
batch_size = 1
shuffle = True
pin_memory = True
training_length = 1
forecast_window = 20
input_size = 772
num_epochs= 300
path_to_save_loss= 'results/'
path_to_save_model = 'saved_models'

# os.makedirs(path_to_save_model, exist_ok=True)
# os.makedirs(path_to_save_loss, exist_ok=True)


train_csv = 'train_dataset.csv'
train_dataset = SensorDataset(csv_name=train_csv, root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_csv = 'test_dataset.csv'
test_dataset = SensorDataset(csv_name=test_csv, root_dir="Data/", training_length=training_length, forecast_window=forecast_window)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def test_eval(test_dataloader, model, device):
    model.eval()
    fde_all =[]
    ade_all =[]
    fiou_all =[]
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    test_loss = 0
    
    with torch.no_grad():
        for input_feat, sensor_number, frames in test_dataloader:
            for tt in range(input_feat.shape[1]):
                bb_input = input_feat[:, tt, :4].unsqueeze(0).to(device)
                vit_feat = input_feat[:, tt, 4:].unsqueeze(0).to(device)
                # Check if there are enough timesteps remaining for y
                if tt + forecast_window+1 <= input_feat.shape[1]:
                    y = input_feat[:, tt + 1: tt + forecast_window+1, :4].to(device)
                    mask = torch.ones_like(y, dtype=torch.bool).to(device)  # All ones, no padding
                else:
                    # Pad the remaining timesteps with zeros (or some other padding value)
                    remaining_length = input_feat.shape[1] - (tt + 1)
                    y = input_feat[:, tt + 1:, :4].to(device)
                    padding = torch.zeros((1, forecast_window - remaining_length, 4)).to(device)
                    y = torch.cat((y, padding), dim=1)
                    mask = torch.cat((torch.ones((1, remaining_length, 4)).to(device),
                            torch.zeros((1, forecast_window - remaining_length, 4)).to(device)), dim=1)
                output = model(bb_input, vit_feat)
                # Apply the mask to the output and y
                masked_output = output * mask
                masked_y = y * mask
                # Calculate the loss only on the masked regions
                loss = criterion(masked_output, masked_y)
                test_loss+= loss.item()
                fde = final_displacement_error(y, output, 640,480)
                fde_all.append(fde)
                ade = average_displacement_error(y, output, 640,480)
                ade_all.append(ade)
                fiou = final_intersection_over_union(y,output)
                fiou_all.append(fiou)
                total_samples+=1
    average_loss = total_loss/ total_samples

    return sum(fde_all) / len(fde_all), sum(ade_all) / len(ade_all), sum(fiou_all) / len(fiou_all), average_loss



# import torch
# import torch.nn as nn
# from tqdm import tqdm
# import os
# import csv
# from datetime import date
# import time

# def train(model, train_dataloader, test_dataloader, device, num_epochs, learning_rate, path_to_save_loss):
#     model_dir = './snapshot'
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     model = model.to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
#     criterion = nn.MSELoss()
    
#     min_train_loss = float('inf')
#     min_fde = float('inf')
#     best_train_model = ""
#     best_test_model = ""

#     # Setup CSV for logging results
#     today = date.today()
#     date_saved = today.strftime("%b-%d-%Y")
#     current_time = time.strftime("%H-%M-%S", time.localtime())
#     result_csv = os.path.join(path_to_save_loss, f'result{date_saved}_{current_time}.csv')
    
#     with open(result_csv, 'a', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['epoch', 'Train_Loss', 'Test_loss', 'fde', 'ade', 'fiou'])

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0.0
#         loop = tqdm(train_dataloader, total=len(train_dataloader), leave=True)
        
#         for input_feat, sensor_number, frames in loop:
#             optimizer.zero_grad()
#             batch_loss = 0.0
            
#             for tt in range(12):  # prediction window = 12
#                 bb_input = input_feat[:, tt, :4].unsqueeze(0).to(device)
#                 vit_feat = input_feat[:, tt, 4:].unsqueeze(0).to(device)
#                 y = input_feat[:, tt+1:tt+13, :4].to(device)  # total length 25

#                 output = model(bb_input, vit_feat)
#                 loss = criterion(output, y)
#                 batch_loss += loss.item()
            
#             batch_loss /= 12  # Average loss over 12 time steps
#             batch_loss = torch.tensor(batch_loss, requires_grad=True)
#             batch_loss.backward()
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
#             optimizer.step()
            
#             train_loss += batch_loss.item()
#             loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
#             loop.set_postfix(loss=batch_loss.item())
        
#         avg_train_loss = train_loss / len(train_dataloader)
        
#         if avg_train_loss < min_train_loss:
#             torch.save(model.state_dict(), os.path.join(model_dir, "best_train_model.pth"))
#             min_train_loss = avg_train_loss
#             best_train_model = "best_train_model.pth"
#             print(f'Best train model saved in epoch {epoch+1}')
        
#         if epoch % 2 == 0:  # Evaluate every 2 epochs
#             print('================================')
#             model.eval()
#             with torch.no_grad():
#                 fde, ade, fiou, test_loss = test_eval(test_dataloader, model, device)
            
#             print(f'FDE: {fde}')
#             print(f'ADE: {ade}')
#             print(f'FIoU: {fiou}')
            
#             with open(result_csv, 'a+', newline='') as saving_result:
#                 writer = csv.writer(saving_result)
#                 writer.writerow([epoch+1, avg_train_loss, test_loss, fde, ade, fiou])

#             if fde < min_fde:
#                 torch.save(model.state_dict(), os.path.join(model_dir, "best_test_model.pth"))
#                 min_fde = fde
#                 best_test_model = "best_test_model.pth"
#                 print(f'Min FDE model saved in epoch {epoch+1}')
        
#         scheduler.step(avg_train_loss)  # Update learning rate based on train loss

#     return best_train_model, best_test_model

# model= GRUModel().to(device) 
# train(model, train_dataloader, test_dataloader, device, num_epochs, learning_rate, path_to_save_loss)

# Custom loss function
def improved_custom_bbox_loss(masked_output, masked_y, mask):
    # Use Mean Squared Error as the base loss
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(masked_output, masked_y)
    
    # Give more weight to position (x, y) than size (w, h)
    loss[:, :, :2] *= 2
    
    # Sum the loss and normalize by the number of valid entries
    total_loss = loss.sum()
    num_valid = mask.sum()
    
    return total_loss / (num_valid + 1e-6)  # Add small epsilon to avoid division by zero

def auxiliary_loss(predictions, targets, mask, alpha=0.2):
    mse_loss = nn.MSELoss(reduction='none')
    
    # Calculate losses
    total_loss = mse_loss(predictions, targets)
    
    # Apply mask
    masked_loss = total_loss * mask
    
    # Calculate FDE (using only the last timestep)
    fde_loss = masked_loss[:, -1, :].mean()
    
    # Calculate ADE (using all timesteps)
    ade_loss = masked_loss.mean()
    
    # Combine losses
    combined_loss = (1 - alpha) * fde_loss + alpha * ade_loss
    
    return combined_loss

def train():
    model_dir ='./snapshot'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    input_size = 772
    h_dim = 256
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    # criterion = nn.MSELoss()
    criterion = auxiliary_loss
    min_train_loss = float('inf')
    min_fde = float('inf')

    ### time 
    today = date.today()
    date_saved = today.strftime("%b-%d-%Y")

    t = time.localtime()
    current_time = time.strftime("%H-%M-%S", t)



    result_csv = os.path.join(path_to_save_loss, f'result{date_saved}_{current_time}.csv')
    with open(result_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'Train_Loss', 'Test_loss', 'fde', 'ade', 'fiou'])

    model.train()

    

    for epoch in range(num_epochs):
        train_loss = 0
        loop = tqdm(train_dataloader,total = len(train_dataloader), leave = True)
        for input_feat, sensor_number, frames in loop:            
            loop.set_description(f"Epoch  [{epoch+1}/{num_epochs}]")
            batch_loss = 0
            optimizer.zero_grad()
            for tt in range(input_feat.shape[1]):
                bb_input = input_feat[:, tt, :4].unsqueeze(0).to(device)
                vit_feat = input_feat[:, tt, 4:].unsqueeze(0).to(device)
                # Check if there are enough timesteps remaining for y
                if tt + forecast_window+1 <= input_feat.shape[1]:
                    y = input_feat[:, tt + 1: tt + forecast_window+1, :4].to(device)
                    mask = torch.ones_like(y, dtype=torch.bool).to(device)  # All ones, no padding
                else:
                    # Pad the remaining timesteps with zeros (or some other padding value)
                    remaining_length = input_feat.shape[1] - (tt + 1)
                    y = input_feat[:, tt + 1:, :4].to(device)
                    padding = torch.zeros((1, forecast_window - remaining_length, 4)).to(device)
                    y = torch.cat((y, padding), dim=1)
                    mask = torch.cat((torch.ones((1, remaining_length, 4)).to(device),
                            torch.zeros((1, forecast_window - remaining_length, 4)).to(device)), dim=1)
                output = model(bb_input, vit_feat)
                # Apply the mask to the output and y
                masked_output = output * mask
                masked_y = y * mask
                # Calculate the loss only on the masked regions
                # loss = criterion(masked_output, masked_y)
                loss = criterion(masked_output, masked_y,mask)
                batch_loss += loss.item()
                loss.backward()

            
            
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss+= batch_loss
        
        avg_train_loss = train_loss / len(train_dataloader)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss = avg_train_loss)
        if avg_train_loss < min_train_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, "best_train_model.pth"))
            min_train_loss = avg_train_loss
            best_train_model = "best_train_model.pth"
            print(f'Best train model saved in {epoch} epoch')
        if epoch % 2 == 0: # Plot 1-Step Predictions
            print('================================')
            
            fde, ade, fiou , test_loss = test_eval(test_dataloader, model, device)
            
            print(f'FDE : {fde}')
            print(f'ADE : {ade}')
            print(f'FIoU : {fiou}')
            with open(result_csv, 'a+', newline='') as saving_result:
                writer = csv.writer(saving_result)
                writer.writerow([epoch, train_loss, test_loss, fde, ade, fiou])

            if fde < min_fde:
                torch.save(model.state_dict(), os.path.join(model_dir, "best_test_model.pth"))
                min_fde = fde
                best_test_model = "best_test_model.pth"
                print(f'Min FDE model saved in {epoch} epoch')
        scheduler.step(avg_train_loss)
        model.train()

train()

# # Example usage
# batch_size = 32
# input_size = 772
# seq_length = 1

# # Create an instance of the model
# model = GRUModel()

# # Create a random input tensor
# x = torch.randn(batch_size, seq_length, input_size)

# # Forward pass
# output = model(x)

# print(f"Input shape: {x.shape}")
# print(f"Output shape: {output.shape}")