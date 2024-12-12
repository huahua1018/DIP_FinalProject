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
import torchmetrics
from torchvision import models
from torchvision.models.vgg import VGG16_Weights  # 確保正確導入


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


import torch
import torch.nn as nn

# 假設 VGG_loss 是你自訂的損失函數
class VGGLoss(nn.Module):
    def __init__(self, feature_extractor, device):
        super(VGGLoss, self).__init__()
        self.feature_extractor = feature_extractor.to(device)  # 移動到指定設備
        self.mse = nn.MSELoss()

    def forward(self, x, y):
        # 確保輸入也在同一設備上
        x = x.to(next(self.feature_extractor.parameters()).device)
        y = y.to(next(self.feature_extractor.parameters()).device)
        x_features = self.feature_extractor(x)
        y_features = self.feature_extractor(y)
        return self.mse(x_features, y_features)

# Combined loss
class CombinedLoss(nn.Module):
    def __init__(self, vgg_loss, mse_loss):
        super(CombinedLoss, self).__init__()
        self.vgg_loss = vgg_loss
        self.mse_loss = mse_loss

    def forward(self, predicted, target):
        vgg_loss_value = self.vgg_loss(predicted, target)
        mse_loss_value = self.mse_loss(predicted, target)
        return vgg_loss_value + mse_loss_value


hyper_params = { # TODO
    'train1_size': 5, #5
    'train2_size': 125, #125
    'input_size': (3, 256, 256),
    'batch_size': 4,
    'num_epochs': 1,
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
            if( i % 50 == 0):
                cv2.imwrite("train_results/features" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(features[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                # cv2.imwrite("train_results/target_transmission" + str(i) + "_" + str(epoch) + ".png", 
                # (cv2.cvtColor((np.transpose(target_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
                
                # cv2.imwrite("train_results/target_reflection" + str(i) + "_" + str(epoch) + ".png", 
                # (cv2.cvtColor((np.transpose(target_reflection[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

                cv2.imwrite("train_results/predict_transmission_" + str(i) + "_" + str(epoch) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

            ########################################################

            # print(predict_transmission[0])
            # print(target_transmission[0] - predict_transmission[0])
            loss = criterion(target_transmission, predict_transmission)

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
                th.save(model, 's2_test.hdf5')#TODO 
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
    # criterion = nn.MSELoss()

    # Loss functions
    vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16]
    for param in vgg.parameters():
        param.requires_grad = False
    vgg_loss = VGGLoss(vgg, device)
    mse_loss = nn.MSELoss()
    combined_loss = CombinedLoss(vgg_loss, mse_loss)

    criterion = combined_loss

    optimizer = optim.Adam(net.parameters(), lr=hyper_params['learning_rate'])
    losses = train(train_loader, net, criterion, optimizer)
    print(losses)

