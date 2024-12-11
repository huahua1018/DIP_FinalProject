import numpy as np
from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import NetArticle
import Image_DataSet as dtst
import cv2
import os
import errno
import shutil
import matplotlib.pyplot as plt


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass

#import config


# experiment = Experiment(api_key=config.comet_ml_api,
#                         project_name="reflection-separation", workspace="wibbn")

hyper_params = { # TODO
    'train1_size': 5, #5
    'train2_size': 25, #125
    'input_size': (3, 256, 256),
    'batch_size': 4,
    'num_epochs': 10,
    'learning_rate': 0.001
}

def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def train(train_loader, model, criterion, optimizer, epochs=hyper_params['num_epochs'], save=True, device=th.device("cuda")):
    # with experiment.train():
    losses = []
    model.train()
    step = 0
    for epoch in range(epochs):
        # experiment.log_current_epoch(epoch)
        losses = []
        for i, batch in enumerate(train_loader):
            features, target_transmission, target_reflection = get_batch(batch)
            features = features.to(device)
            target_transmission = target_transmission.to(device)
            optimizer.zero_grad()
            #predict_transmission, predict_reflection = model(features)
            predict_transmission = model(features)
            print('__________________________________')
            # print(predict_transmission.shape)
            #### TODO yu ####################################################
            # predict_transmission -> (8,3,256,256) 所以我把每一層都畫出來 或許在某一層有結果w (論文原本是第0層)
            if( i % 500 == 0):
                cv2.imwrite("train_results/features" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(features[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                # cv2.imwrite("train_results/target_transmission" + str(i) + "_" + str(epoch) + ".png", 
                # (cv2.cvtColor((np.transpose(target_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
                
                # cv2.imwrite("train_results/target_reflection" + str(i) + "_" + str(epoch) + ".png", 
                # (cv2.cvtColor((np.transpose(target_reflection[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission0_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission1_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[1].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission2_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[2].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission3_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[3].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission4_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[4].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission5_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[5].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission6_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[6].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission7_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[7].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
            ########################################################

            #print(predict_transmission[0])
            #print(target_transmission[0] - predict_transmission[0])
            loss = criterion(predict_transmission, target_transmission)
            # loss2 = criterion(predict_reflection, target_reflection)
            # loss = loss1 + loss2
            #print("LOSSES: ", loss1, loss)
            loss.backward()
            optimizer.step()
            print(epoch, step, loss.item(), th.mean(model.conv_down_6.weight.grad[0][0]), th.mean(model.conv_intro_2.weight.grad[0][0]))
            # experiment.log_metric('loss', loss.item(), step=step)
            step += 1
            losses.append(loss.item())
            if save:
                th.save(model, 's_test.hdf5')#TODO 
        print('epoch end', sum(losses))
    return losses


if __name__ == "__main__":

    path = "./train_results"
    del_folder(path)
    create_folder(path)

    print(th.__version__)
    # experiment.log_parameters(hyper_params) # TODO
    device = th.device("cuda")
    print(device)
    # data = dtst.ImageDataSet(hyper_params['train1_size'], hyper_params['train2_size'])
    data = dtst.ImageDataSet()
    train_loader = dtst.DataLoader(data, 1, 11)##TODO third is reflection_num val it cant be bigger than train1_size

    net = NetArticle().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_loader, net, criterion, optimizer)
    print(losses)

