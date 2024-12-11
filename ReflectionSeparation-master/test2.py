import numpy as np
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
from PIL import Image


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

hyper_params = { # TODO
    'train1_size': 45, #20
    'train2_size': 45, #100
    'input_size': (3, 256, 256),
    'save_folder': './results',
    # 'num_epochs': 10, 
    # 'learning_rate': 0.001 
}


def get_batch(batch):
    features = batch[:, 0, :, :, :]
    target_transpose = batch[:, 1, :, :, :]
    target_reflection = batch[:, 2, :, :, :]
    #target = th.Tensor(np.concatenate((target_transpose, target_reflection), axis=1))
    return features, target_transpose, target_reflection


def test(model, criterion, path):
    losses = []
    model.eval()
    step = 0
    for i in range(0,150,15):
        img = Image.open('{}/{}.png'.format('../root_SIR2_test', i))
        img = img.convert('RGB')
        img = np.array(img)
        features_img = np.transpose(img, (2, 0, 1))
        item = np.array([features_img, features_img, features_img])
        item = th.Tensor(item / 255)
        features  = item

        img = Image.open('{}/{}.png'.format('../root_SIR2_gt', i))
        img = img.convert('RGB')
        img = np.array(img)
        features_img = np.transpose(img, (2, 0, 1))
        item = np.array([features_img, features_img, features_img])
        item = th.Tensor(item / 255)
        target_transmission = item 


        ### 一道GPU上
        device = th.device("cuda" if th.cuda.is_available() else "cpu")

        # 確保模型在正確設備上
        model = model.to(device)

        # 確保 features 在相同的設備上
        features = features.to(device)

        # TODO yu ------------------------------
        predict_transmission = model(features)
        # --------------------------------------
        
        ###
        # predict_transmission, predict_reflection = model(features)
        if i < 1:
            print("-------------------------------")
            # TODO yu ---------------------------------------------------------------------------------------
            a = (np.transpose(predict_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(int)
            # b = (np.transpose(target_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(int)
            # -----------------------------------------------------------------------------------------------
            
            # a = (np.transpose(predict_transmission[0].detach().numpy(), (1, 2, 0)) * 255).astype(int)
            # b = (np.transpose(target_transmission[0].detach().numpy(), (1, 2, 0)) * 255).astype(int)
            # print(a - b)
            # print(predict_transmission[0] - target_transmission[0])

            '''
            writer = open("hello.txt", 'w')
            for i in range(20, 30):
                for j in range(20, 30):
                    print(a[i][j][0], file=writer)
            '''
            print("-------------------------------")
        
        # TODO yu-------------------------------------------------
        predict_transmission = predict_transmission.to(device)
        target_transmission = target_transmission.to(device)
        # ---------------------------------------------------------

        # TODO yu-------------------------------------------------
        cv2.imwrite(path + "/predict_transmission" + str(i) + ".png", 
                (cv2.cvtColor((np.transpose(predict_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))

        cv2.imwrite(path + "/target_transmission" + str(i) + ".png", 
                (cv2.cvtColor((np.transpose(target_transmission[0].detach().cpu().numpy(), (1, 2, 0)) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)))
        # ---------------------------------------------------------
        
            # ---------------------------------------------------------
            # cv2.imwrite(path + "/target_trans" + str(i) + ".png", (np.transpose(target_transmission[j].detach().numpy(), (1, 2, 0)) * 255).astype(int))
            # cv2.imwrite(path + "/transmission" + str(i) + ".png", (np.transpose(predict_transmission[j].detach().numpy(), (1, 2, 0)) * 255).astype(int))
            # cv2.imwrite(path + "/reflection" + str(i) + ".png", (np.transpose(predict_reflection[j].detach().numpy(), (1, 2, 0)) * 255).astype(int))

        loss1 = criterion(predict_transmission, target_transmission)
        # TODO yu-----------------------------------------------------
        loss = loss1

        # loss2 = criterion(predict_reflection, target_reflection)
        # loss = loss1 + loss2
        # ------------------------------------------------------------
        losses.append(loss.item())
        print(loss1.item())
        step += 1
    return losses


if __name__ == "__main__":

    print(th.__version__)

    path = hyper_params['save_folder']
    del_folder(path)
    create_folder(path)

    # todo yu ---------------------------------------------------------------------------------------
    # data = dtst.ImageDataSet(hyper_params['train1_size'], hyper_params['train2_size'], test = True)
    # data = dtst.ImageDataSet(hyper_params['train1_size'], hyper_params['train2_size'])
    # -----------------------------------------------------------------------------------------------
    # test_loader = dtst.DataLoader(data, 1, 11, test=True)# TODO　亂數每次都找９個　所以這邊先18->11

    #net = NetArticle()
    net = th.load("first_test.hdf5") # TODO
    # print(net)
    criterion = nn.MSELoss()
    losses = test(net, criterion, path)
    print('####', losses)

