from model import AdvancedGRUModel as GRUModel
import torch.nn as nn
import torch, math
import argparse

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
forecast_window = 12
input_size = 772
num_epochs= 50
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

def sanity_check():
    model_path = 'snapshot/best_test_model.pth'
    model = GRUModel()  # Initialize your model
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    fde, ade, fiou , test_loss = test_eval(test_dataloader, model, device)
    print(fde)
    return fde, ade, fiou , test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', choices=['check', 'train', 'test'],
                        help='dimension of the resnet output. Default: 2048')
    p = parser.parse_args()
    if p.phase == 'test':
        sanity_check()
    else:
        train()
