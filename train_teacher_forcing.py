from model import Transformer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from DataLoader import SensorDataset
import logging
import time # debugging
# from plot import *
from helpers import *
# from joblib import load
# from icecream import ic
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from utils import final_displacement_error, average_displacement_error, final_intersection_over_union
import time
from datetime import date
import csv
import numpy as np


seed = 123
np.random.seed(seed)
torch.manual_seed(seed)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

def test_eval(test_dataloader, model, device, forecast_window):
    model.eval()
    fde_all =[]
    ade_all =[]
    fiou_all =[]
    criterion = nn.MSELoss()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for index_in, index_tar, _input, true, sensor_number, frames in test_dataloader:
            src1 = _input.permute(1,0,2).double().to(device)[0:,:,:]
            src2 =src1.permute(1,0,2)
            src = src2.permute(1,0,2)
            
            next_input_model = src
            true_val2 = true.unsqueeze(1).to('cuda')
            true_val1 = true_val2.squeeze(0)
            true_val = true_val1.permute(1,0, 2)
            for tt in range(forecast_window-1):
                prediction = model(next_input_model, device) # torch.Size([sequence, batch, feature])
                new_line =  torch.cat((src[-1-tt:, :, 4:], prediction[-1-tt:,:,:]),dim=2)
                next_input_model = torch.cat((src[tt+1:, :, :], new_line))
                tar = torch.cat((src[tt+1:, :, :4], true_val[:tt+1]))

                loss = criterion(prediction, tar)
                total_loss += loss.item()
                fde = final_displacement_error(tar,prediction, 640,480)
                fde_all.append(fde)
                ade = average_displacement_error(tar,prediction, 640,480)
                ade_all.append(ade)
            total_samples+=1
        average_loss = total_loss/ total_samples

    return sum(fde_all) / len(fde_all), sum(ade_all) / len(ade_all), average_loss

def transformer(train_dataloader, test_dataloader, EPOCH, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device):

    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_train_model = ""
    best_test_model = ""
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

    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0        
        

        ## TRAIN -- TEACHER FORCING
        model.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (index_in, index_tar, _input, target_, sensor_number) in loop: # for each data set 
        
            optimizer.zero_grad()

            # Shape of _input : [batch, input_length, feature]
            # Desired input for model: [input_length, batch, feature]
            print('input: ',_input)

            src = _input.permute(1,0,2).double().to(device)[:-1,:,:] # torch.Size([24, 1, 7])
            print('src: ', src)
            target = _input.permute(1,0,2).double().to(device)[1:,:,:4] # src shifted by 1.
            print('target: ',target)


  
            prediction = model(src, device) # torch.Size([24, 1, 7])

            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            # scheduler.step(loss.detach().item())
            train_loss += loss.detach().item()
            loop.set_description(f"Epoch  [{epoch}/{EPOCH}]")
            loop.set_postfix(train_loss=f"{train_loss:.3f}")

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), os.path.join(path_to_save_model, "best_train_model.pth"))
            min_train_loss = train_loss
            best_train_model = "best_train_model.pth"


        if epoch % 10 == 0: # Plot 1-Step Predictions
            print('================================')
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            fde, ade, fiou , test_loss = test_eval(test_dataloader, model, device)
            logger.info(f"Epoch: {epoch}, Testing loss: {test_loss}")
            print(f'FDE : {fde}')
            print(f'ADE : {ade}')
            print(f'FIoU : {fiou}')
            with open(result_csv, 'a+', newline='') as saving_result:
                writer = csv.writer(saving_result)
                writer.writerow([epoch, train_loss, test_loss, fde, ade, fiou])

            if fde < min_fde:
                torch.save(model.state_dict(), os.path.join(path_to_save_model, "best_test_model.pth"))
                min_fde = fde
                best_test_model = "best_test_model.pth"

        scheduler.step()
        train_loss /= len(train_dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    # plot_loss(path_to_save_loss, train=True)
    return best_train_model

'''
def transformer(train_dataloader, test_dataloader, EPOCH, frequency, path_to_save_model, path_to_save_loss, path_to_save_predictions, device, forecast_window):


    device = torch.device(device)

    model = Transformer().double().to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=200)
    criterion = torch.nn.MSELoss()
    best_train_model = ""
    best_test_model = ""
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
        writer.writerow(['epoch', 'Train_Loss', 'Test_loss', 'fde', 'ade'])

    for epoch in range(EPOCH + 1):

        train_loss = 0
        val_loss = 0        
        

        ## TRAIN -- TEACHER FORCING
        model.train()
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, (index_in, index_tar, _input, true, sensor_number,frames) in loop: # for each data set 
        
            optimizer.zero_grad()

            src1 = _input.permute(1,0,2).double().to(device)[0:,:,:]
            src2 =src1.permute(1,0,2)
            src = src2.permute(1,0,2)
            
            next_input_model = src
            true_val2 = true.unsqueeze(1).to('cuda')
            true_val1 = true_val2.squeeze(0)
            true_val = true_val1.permute(1,0, 2)

            inter_loss = 0
            for tt in range(forecast_window-1):
                prediction = model(next_input_model,device)
                print(prediction)
                
                new_line =  torch.cat((src[-1-tt:, :, 4:], prediction[-1-tt:,:,:]),dim=2)
                next_input_model = torch.cat((src[tt+1:, :, :], new_line))
                tar = torch.cat((src[tt+1:, :, :4], true_val[:tt+1]))
                
                loss = criterion(prediction, tar)
                inter_loss += loss 
                
            inter_loss.backward()
            optimizer.step()
            train_loss+=inter_loss.detach().item()
            loop.set_description(f"Epoch  [{epoch}/{EPOCH}]")
            loop.set_postfix(train_loss=f"{train_loss:.3f}")

        if train_loss < min_train_loss:
            torch.save(model.state_dict(), os.path.join(path_to_save_model, "best_train_model.pth"))
            min_train_loss = train_loss
            best_train_model = "best_train_model.pth"


        if epoch % 10 == 0: # Plot 1-Step Predictions
            print('================================')
            logger.info(f"Epoch: {epoch}, Training loss: {train_loss}")
            fde, ade, test_loss = test_eval(test_dataloader, model, device, forecast_window)
            logger.info(f"Epoch: {epoch}, Testing loss: {test_loss}")
            print(f'FDE : {fde}')
            print(f'ADE : {ade}')
            with open(result_csv, 'a+', newline='') as saving_result:
                writer = csv.writer(saving_result)
                writer.writerow([epoch, train_loss, test_loss, fde, ade])

            if fde < min_fde:
                torch.save(model.state_dict(), os.path.join(path_to_save_model, "best_test_model.pth"))
                min_fde = fde
                best_test_model = "best_test_model.pth"

        scheduler.step()
        train_loss /= len(train_dataloader)
        log_loss(train_loss, path_to_save_loss, train=True)
        
    # plot_loss(path_to_save_loss, train=True)
    return best_train_model

    '''
