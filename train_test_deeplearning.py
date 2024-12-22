

import os
import time
import torch
import torch.nn as nn
import random
import numpy as np
import cv2
from fvcore.nn import FlopCountAnalysis, parameter_count
import utils.evaluation as evaluation
import utils.data_load_operate as data_load_operate
import model.DBDA as DBDA               #
import model.HybridSN as HybridSN
import model.SSFTT as SSFTT
import model.SSAtt as SSAtt
import model.A2S2KResNet as A2S2KResNet
import model.AMS_M2ESL as AMS_M2ESL
import model.SSSAN as SSSAN
import model.CVSSN as CVSSN
import model.CAE_HTCNet as CAE_HTCNet
import torch.nn.init as init

from sklearn import metrics
from ptflops import get_model_complexity_info
import visual.cls_visual as cls_visual
import sys


time_current = time.strftime("%y-%m-%d-%H.%M", time.localtime())
seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#                0         1          2        3         4             5         6         7         8
model_list = ['DBDA', 'HybridSN', 'SSSAN', 'SSAtt', 'A2S2KResNet', 'CVSSN', 'SSFTT', 'AMS-M2ESL', 'CAE-HTCNet']
# Model selection
model_flag = 8

# Input data shape
model_3D_spa_flag = 0
model_spa_set = {0,1,3,4,6,7,8}
model_spa_spe_set = {2,5,7}
model_3D_spa_set = {0,1,4,6}
model_spe_set = {}
if model_flag in model_spa_set:
    model_type_flag = 1
    if model_flag in model_3D_spa_set:
        model_3D_spa_flag = 1
elif model_flag in model_spe_set:
    model_type_flag = 2
elif model_flag in model_spa_spe_set:
    model_type_flag = 3

################### Parameter Setting ###################
data_set_name_list = ['IP', 'KSC', 'SA']
mnf_ratio=[0.1,0.2,	0.3,0.4,0.5,0.6,0.7]
num_encoder_list=[1,2,3,4,5]
heads_list = [1,2,3,4,5,6,7,8]
dim_head_list = [4,8,16,32]
patch_size_list = [3,5,7,9,11,13,15,17,19,21]
patch_length_list = [1,2,3,4,5,6,7,8,9,10]
dropout_list = [0.2,0.3,0.4,0.5,0.6,0.1]
emb_dropout_list = [0.1,0.2,0.3,0.4,0.5,0.6]
mlp_dim = 128
data_set_name = data_set_name_list[0]

if data_set_name == 'IP' :
    mnf_idx, ps_idx, head_idx, d_head_idx, num_enc_idx, dp_idx, em_dp_idx, ratio_idx = 4, 3, 3, 2, 0, 0, 2, 4
elif data_set_name == 'KSC':
    mnf_idx, ps_idx, head_idx, d_head_idx, num_enc_idx, dp_idx, em_dp_idx, ratio_idx = 4, 3, 3, 0, 0, 0, 2, 4
elif data_set_name == 'SA':
    mnf_idx, ps_idx, head_idx, d_head_idx, num_enc_idx, dp_idx, em_dp_idx, ratio_idx = 4, 6, 5, 0, 0, 0, 2, 2

mnf_ratio = mnf_ratio[mnf_idx]
patch_size, patch_length = patch_size_list[ps_idx], patch_length_list[ps_idx]
dropout = dropout_list[dp_idx]
emb_dropout = emb_dropout_list[em_dp_idx]
heads, dim_head = heads_list[head_idx], dim_head_list[d_head_idx]
num_encoder = num_encoder_list[num_enc_idx]
dim_emb_new = heads*dim_head

if data_set_name == 'IP':
    ratio_list_ = [0.01,0.03,0.05,0.07,0.09]
    # pca_len = 100                              # PCA
    ratio_list = [ratio_list_[2], 0.01]          # train ratio 、 val ratio
    ratio = 5
elif data_set_name == 'KSC':
    ratio_list_ = [0.01,0.03,0.05,0.07,0.09]
    # pca_len=120
    ratio_list = [ratio_list_[2], 0.01]
    ratio = 5
elif data_set_name == 'SA':
    ratio_list_ = [ 0.001,0.005, 0.01,  0.015,  0.02]
    ratio_list = [ratio_list_[2], 0.005]
    # pca_len=102
    ratio = 1
###################

seed_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


flag_list = [0, 1]
num_list = [45, 4]

# Save the results
data_set_path = os.path.join(os.getcwd(), 'data')
results_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/results'), str(model_list[model_flag]) + str("_") +
                 data_set_name + str("_") + str(time_current)+ str("mf_") + str(mnf_ratio) + str("ps_") + str(patch_size)) + str("h_") + str(heads)+ str("_hd_") + str(dim_head) + str("ed_") + str(num_encoder)  + str("dp_") + str(dropout)+ str("em_dp_") + str(emb_dropout) + str("dr_") + str(ratio_list_[ratio_idx])
cls_map_save_path = \
    os.path.join(os.path.join(os.getcwd(), 'output/cls_maps'), str(model_list[model_flag]) +  str("_") +
                 data_set_name + str("_") + str(time_current) + str("mf_") + str(mnf_ratio) + str("ps_") + str(patch_size)) + str("h_") + str(heads)+ str("_hd_") + str(dim_head) + str("ed_") + str(num_encoder)  + str("dp_") + str(dropout)+ str("em_dp_") + str(emb_dropout) + str("dr_") + str(ratio_list_[ratio_idx])

if __name__ == '__main__':
    # Data preprocessing
    data, gt = data_load_operate.load_data(data_set_name, data_set_path)
    data = data_load_operate.standardization(data)

    if model_flag == 8:
        data = data_load_operate.HSI_MNF(data, MNF_ratio=mnf_ratio)
        # data = data_load_operate.applyPCA(data, pca_len)     # PCA

    dr_channels = 0

    if model_flag == 8:
        max_epoch = 150
        batch_size = 32
        learning_rate = 5e-4

    gt_re = gt.reshape(-1)
    height, width, channels = data.shape
    class_count = max(np.unique(gt_re))

    data_padded = cv2.copyMakeBorder(data, patch_length, patch_length, patch_length, patch_length, cv2.BORDER_REFLECT)
    height_patched, width_patched = data_padded.shape[0], data_padded.shape[1]

    data_total_index = np.arange(height * width)

    loss = torch.nn.CrossEntropyLoss()

    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    EACH_ACC_ALL = []
    Train_Time_ALL = []
    Test_Time_ALL = []
    CLASS_ACC = np.zeros([len(seed_list), class_count])
    for curr_seed in seed_list:

        tic1 = time.perf_counter()
        train_data_index, val_data_index, test_data_index, all_data_index = data_load_operate.sampling(
            ratio_list,
            num_list,
            gt_re,
            class_count,
            flag_list[0])

        index = (train_data_index, val_data_index, test_data_index)

        train_iter, test_iter, val_iter = data_load_operate.generate_iter_1(data_padded, height, width, gt_re,
                                                                                         index,
                                                                                         patch_length, batch_size,
                                                                                         model_type_flag,
                                                                                         model_3D_spa_flag,model_flag,
                                                                                           dr_channels)
        # total_iter = data_load_operate.generate_iter_total(data_padded, height, width, gt_re,
        #                                                    data_total_index,
        #                                                    patch_length,
        #                                                    batch_size, model_type_flag, model_3D_spa_flag,model_flag,
        #                                                    dr_channels)

        if model_flag == 0:
            net = DBDA.DBDA_network_MISH(channels, class_count)
        elif model_flag == 1:
            net = HybridSN.HybridSN(patch_length, dr_channels, class_count)
        elif model_flag == 2:
            net = SSSAN.SSSAN(channels, dr_channels, class_count)
        elif model_flag == 3:
            net = SSAtt.Hang2020(channels, class_count)
        elif model_flag == 4:
            net = A2S2KResNet.S3KAIResNet(channels, class_count, 2)
        elif model_flag == 5:
            net = CVSSN.CVSSN_(channels, patch_size, patch_size, class_count)
        elif model_flag == 6:
            net = SSFTT.SSFTTnet(num_classes=class_count)
        elif model_flag == 7:
            net = AMS_M2ESL.AMS_M2ESL_(in_channels=channels, patch_size=patch_size, num_classes=class_count,
                                        ds=data_set_name)

        elif model_flag == 8:
            net = CAE_HTCNet.CAE_HTCNet(in_channel=channels, num_classes=class_count, num_encoder=num_encoder,
                                     dim_emb=dim_emb_new,
                                     heads=heads, mlp_dim=mlp_dim, dim_head=dim_head, dropout=dropout,
                                     emb_dropout=emb_dropout)

        net.to(device)
        if model_flag == 2:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=5e-4)
        elif model_flag == 4:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8,
                                         weight_decay=0)
            lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 15, eta_min=0.0, last_epoch=-1)
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


        # efficiency test, model complexity and computational cost
        # input1 = torch.randn(batch_size, patch_size, patch_size, channels).to(device)
        # flops = FlopCountAnalysis(net, input1)
        # params = parameter_count(net)
        # print(f"FLOPs: {flops.total() / 1e9} G")  # FLOPs in millions
        # print(f"Parameters: {params[''] / 1e6} M")

        train_loss_list = [100]
        train_acc_list = [0]
        val_loss_list = [100]
        val_acc_list = [0]

        best_loss = 99999

        for epoch in range(max_epoch):
            train_acc_sum, trained_samples_counter = 0.0, 0
            batch_counter, train_loss_sum = 0, 0
            net.train()
            time_epoch = time.time()
            if model_type_flag == 1:
                for X_spa, y in train_iter:

                    X_spa, y = X_spa.to(device), y.to(device)
                    y_pred = net(X_spa)
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()

                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif model_type_flag == 2:
                for X_spe, y in train_iter:

                    X_spe, y = X_spe.to(device), y.to(device)

                    y_pred = net(X_spe)
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()

                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0
            elif model_type_flag == 3:
                for X_spa, X_spe, y in train_iter:

                    X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                    y_pred = net(X_spa, X_spe)
                    ls = loss(y_pred, y.long())
                    optimizer.zero_grad()
                    ls.backward()
                    optimizer.step()
                    train_loss_sum += ls.cpu().item()
                    train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()

                    trained_samples_counter += y.shape[0]
                    batch_counter += 1
                    epoch_first_iter = 0

            val_acc, val_loss = evaluation.evaluate_OA(val_iter, net, loss, device, model_type_flag)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)


            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(net.state_dict(), results_save_path + "_best_model.pt")
                print('save model...')


            train_loss_list.append(train_loss_sum)
            train_acc_list.append(train_acc_sum / trained_samples_counter)
            print('epoch: %d, training_sampler_num: %d, batch_count: %.2f, train loss: %.6f, tarin loss sum: %.6f, '
                  'train acc: %.3f, train_acc_sum: %.1f, time: %.1f sec' %
                  (epoch + 1, trained_samples_counter, batch_counter, train_loss_sum / batch_counter, train_loss_sum,
                   train_acc_sum / trained_samples_counter, train_acc_sum, time.time() - time_epoch))

        toc1 = time.perf_counter()
        print('Training stage finished:\n epoch %d, loss %.4f, train acc %.3f, training time %.2f s'
              % (epoch + 1, train_loss_sum / batch_counter, train_acc_sum / trained_samples_counter, toc1 - tic1))
        training_time = toc1 - tic1
        Train_Time_ALL.append(training_time)

        print(f"\n\n====================Starting evaluation for testing set.========={data_set_name}===============\n")
        pred_test = []
        y_gt = []

        with torch.no_grad():
            net.eval()
            train_acc_sum, samples_num_counter = 0.0, 0

            if model_type_flag == 1:
                for X_spa, y in test_iter:
                    X_spa = X_spa.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
                    y_gt.extend(y)
            elif model_type_flag == 2:
                for X_spe, y in test_iter:
                    X_spe = X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)))
                    y_gt.extend(y)
            elif model_type_flag == 3:  # data for spectral-spatial net
                for X_spa, X_spe, y in test_iter:
                    X_spa,X_spe = X_spa.to(device),X_spe.to(device)

                    tic2 = time.perf_counter()
                    y_pred = net(X_spa, X_spe)
                    toc2 = time.perf_counter()

                    pred_test.extend(np.array(y_pred.cpu().argmax(axis=1)).tolist())
                    y_gt.extend(y)

            y_gt = gt_re[test_data_index] - 1
            OA = metrics.accuracy_score(y_gt,pred_test)
            confusion_matrix = metrics.confusion_matrix(y_gt, pred_test)
            print("confusion_matrix\n{}".format(confusion_matrix))
            ECA, AA = evaluation.AA_ECA(confusion_matrix)
            kappa = metrics.cohen_kappa_score(y_gt, pred_test)
            cls_report = evaluation.claification_report(y_gt, pred_test, data_set_name)
            print("classification_report\n{}".format(cls_report))

            # Visualization for all the labeled samples and total the samples
            # sample_list1 = [total_iter] # patch--label
            # sample_list2 = [all_iter, all_data_index]

            # cls_visual.gt_cls_map(gt,cls_map_save_path)
            # cls_visual.pred_cls_map_dl(sample_list1, net, gt, cls_map_save_path, model_type_flag)
            # cls_visual.pred_cls_map_dl(sample_list2,net,gt,cls_map_save_path)

            testing_time = toc2 - tic2
            Test_Time_ALL.append(testing_time)
            print(f"OA={OA}, AA={AA}, Kappa={kappa}")
            f = open(results_save_path + '_results.txt', 'a+')
            str_results = '\n======================' \
                          + "model=" + str("Model9") \
                          + f"head={heads},dim_head={dim_head},num_eco={num_encoder},mf={mnf_ratio},p={patch_size},dp={dropout},em_dp={emb_dropout},数据率={ratio_list_[ratio_idx]}" \
                          + "data_set_name=" + str(data_set_name) \
                          + " learning rate=" + str(learning_rate) \
                          + " epochs=" + str(max_epoch) \
                          + " ======================" \
                          + "\nOA=" + str(OA) \
                          + "\nAA=" + str(AA) \
                          + '\nkpp=' + str(kappa) \
                          + '\nacc per class:' + str(ECA) \
                          + "\ntrain time:" + str(training_time) \
                          + "\ntest time:" + str(testing_time) + "\n"

            f.write(str_results)
            f.write('{}'.format(confusion_matrix))
            f.write('\n\n')
            f.write('{}'.format(cls_report))
            f.close()

            OA_ALL.append(OA)
            AA_ALL.append(AA)
            KPP_ALL.append(kappa)
            EACH_ACC_ALL.append(ECA)

        del net, train_iter, test_iter, val_iter



    OA_ALL = np.array(OA_ALL)
    AA_ALL = np.array(AA_ALL)
    KPP_ALL = np.array(KPP_ALL)
    EACH_ACC_ALL = np.array(EACH_ACC_ALL)
    Train_Time_ALL = np.array(Train_Time_ALL)
    Test_Time_ALL = np.array(Test_Time_ALL)

    np.set_printoptions(precision=4)
    print("\n====================Mean result of {} times runs =========================".format(len(seed_list)))
    print('data_set_name:', data_set_name)
    print(f"mf={mnf_ratio},p={patch_size},head={heads},dim_head={dim_head},num_eco={num_encoder},dp={dropout},em_dp={emb_dropout},数据率={ratio_list_[ratio_idx]}")
    print('List of OA:', list(OA_ALL))
    print('List of AA:', list(AA_ALL))
    print('List of KPP:', list(KPP_ALL))
    print('OA=', round(np.mean(OA_ALL) * 100, 2), '+-', round(np.std(OA_ALL) * 100, 2))
    print('AA=', round(np.mean(AA_ALL) * 100, 2), '+-', round(np.std(AA_ALL) * 100, 2))
    print('Kpp=', round(np.mean(KPP_ALL) * 100, 2), '+-', round(np.std(KPP_ALL) * 100, 2))
    print('Acc per class=', np.round(np.mean(EACH_ACC_ALL, 0) * 100, decimals=2), '+-',
          np.round(np.std(EACH_ACC_ALL, 0) * 100, decimals=2))
    print("Average training time=", round(np.mean(Train_Time_ALL), 2), '+-', round(np.std(Train_Time_ALL), 3))
    print("Average testing time=", round(np.mean(Test_Time_ALL) * 1000, 2), '+-',
          round(np.std(Test_Time_ALL) * 1000, 3))

    # Output infors
    f = open(results_save_path + '_results.txt', 'a+')
    str_results = '\n\n***************Mean result of ' + str(len(seed_list)) + 'times runs ********************' \
                  + "\nmodel=" + model_list[model_flag] \
                  + "\ndata_set_name=" + data_set_name \
                  + "mnf_ratio=" + str(mnf_ratio) \
                  + "ps=" + str(patch_size) \
                  + "heads=" + str(heads) \
                  + "dim_head=" + str(dim_head) \
                  + "num_encoder=" + str(num_encoder) \
                  + "dropout=" + str(dropout) \
                  + "dropout=" + str(ratio_list_[ratio_idx]) \
                  + "emb_dropout=" + str(emb_dropout) \
                  + '\nList of OA:' + str(list(OA_ALL)) \
                  + '\nList of AA:' + str(list(AA_ALL)) \
                  + '\nList of KPP:' + str(list(KPP_ALL)) \
                  + '\nOA=' + str(round(np.mean(OA_ALL) * 100, 2)) + '+-' + str(round(np.std(OA_ALL) * 100, 2)) \
                  + '\nAA=' + str(round(np.mean(AA_ALL) * 100, 2)) + '+-' + str(round(np.std(AA_ALL) * 100, 2)) \
                  + '\nKpp=' + str(round(np.mean(KPP_ALL) * 100, 2)) + '+-' + str(round(np.std(KPP_ALL) * 100, 2)) \
                  + '\nAcc per class=\n' + str(np.mean(EACH_ACC_ALL, 0)) + '+-' + str(np.std(EACH_ACC_ALL, 0)) \
                  + "\nAverage training time=" + str(np.round(np.mean(Train_Time_ALL), decimals=2)) + '+-' + str(
        np.round(np.std(Train_Time_ALL), decimals=3)) \
                  + "\nAverage testing time=" + str(np.round(np.mean(Test_Time_ALL) * 1000, decimals=2)) + '+-' + str(
        np.round(np.std(Test_Time_ALL) * 100, decimals=3))
    f.write(str_results)
    f.close()
    torch.cuda.empty_cache()
