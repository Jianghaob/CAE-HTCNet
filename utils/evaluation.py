import torch
import numpy as np

from sklearn import metrics
from operator import truediv


def evaluate_OA(data_iter, net, loss, device, model_type_flag):
    acc_sum, samples_counter = 0, 0

    with torch.no_grad():
        net.eval()
        if model_type_flag == 1:  # data for single spatial net
            loss_sum = 0
            for X_spa, y in data_iter:
                X_spa, y = X_spa.to(device), y.to(device)
                y_pred = net(X_spa)

                ls = loss(y_pred, y.long())


                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 2:  # data for single spectral net
            loss_sum = 0
            for X_spe, y in data_iter:
                X_spe, y = X_spe.to(device), y.to(device)
                y_pred = net(X_spe)

                ls = loss(y_pred, y.long())

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]
        elif model_type_flag == 3:  # data for spectral-spatial net
            loss_sum = 0
            for X_spa, X_spe, y in data_iter:
                X_spa, X_spe, y = X_spa.to(device), X_spe.to(device), y.to(device)
                y_pred = net(X_spa, X_spe)

                ls = loss(y_pred, y.long())

                # threshold_reg_loss = net.get_threshold_regularization()
                # ls = ls + threshold_reg_loss

                acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
                loss_sum += ls

                samples_counter += y.shape[0]

    return [acc_sum / samples_counter, loss_sum / samples_counter]


def AA_ECA(confusion_matrix):
    # get diagonal element
    diag_list = np.diag(confusion_matrix)
    row_sum_list = np.sum(confusion_matrix, axis=1)
    each_per_acc = np.nan_to_num(truediv(diag_list, row_sum_list))
    avg_acc = np.mean(each_per_acc)

    return each_per_acc, avg_acc


def claification_report(label, pred, name):
    if name == 'IP':
        target_names = ['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn'
            , 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed',
                        'Hay-windrowed', 'Oats', 'Soybean-notill', 'Soybean-mintill',
                        'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-Drives',
                        'Stone-Steel-Towers']
    elif name == "KSC":
        target_names = ['Scrub', 'Willow swamp', 'Cabbage palm hammock', 'Cabbage palm/oak hammock', 'Slash pine',
                        'Oak/broadleaf hammock',
                        'Hardwood swamp', 'Graminoid marsh', 'Spartine marsh', 'Cattail marsh', 'Salt marsh',
                        'Mud flats', 'Water']
    elif name == 'UP':
        target_names = ['Asphalt', 'Meadows', 'Gravel', 'Trees', 'Painted metal sheets', 'Bare Soil', 'Bitumen',
                        'Self-Blocking Bricks', 'Shadows']
    elif name == 'HU':
        target_names = ['Grass_healthy', 'Grass_stressed', 'Grass_synthetic', 'Tree', 'Soil', 'Water', 'Residential',
                        'Commercial', 'Road', 'Highway', 'Railway', 'Parking_lot1', 'Parking_lot2', 'Tennis_court',
                        'Running_track']
    elif name == 'SA':
        target_names = ['Brocoli-green-weeds-1', 'Brocoli-green-weeds-2', 'Fallow', 'Fallow-rough-plow', 'Fallow-smooth', 'Stubble', 'Celery',
                        'Grapes-untrained', 'Soil-vinyard-develop', 'Corn-senesced-green-weeds', 'Lettuce-romaine-4wk', 'Lettuce-romaine-5wk', 'Lettuce-romaine-6wk', 'Lettuce-romaine-7wk',
                        'Vinyard-untrained','Vinyard-vertical-trellis']
    elif name == 'BOT':
        target_names = ['Water', 'Hippograss', 'Floodplaingrasses1', 'Floodplaingrasses2',
                        'Reeds1', 'Riparian', 'Fierscar2',
                        'Island interior', 'Acacia woodlands', 'Acacia shrublands', 'Acacia grasslands',
                        'Shortmopane', 'Mixedmopane', 'Exposedsoils']
    elif name == 'CH':
        target_names = ['Rice stubble', 'Grassland', 'Elm', 'Ash tree','Pagoda Tree', 'Vegetable field', 'Poplar',
                        'Soybean', 'Black locust', 'Rice', 'Water','Willow', 'Acer negundo', 'Goldenrain tree',
                        'Peach tree', 'Corn', 'Pear tree','Lotus leaf','Building']


    classification_report = metrics.classification_report(label, pred, target_names=target_names)
    return classification_report
