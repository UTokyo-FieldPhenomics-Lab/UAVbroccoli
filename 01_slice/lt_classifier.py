import time
import os
import sys
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import skimage
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from skimage.feature import hog
from skimage import io, color
from skimage.color import rgb2gray
from lt_fast_glcm import *


def expand_colorspace(img):
    # img is the ndarray from io.imread()
    (h, w, dimen) = img.shape
    
    if dimen == 4: # exist alpha layer (RGBA)
        img = img[:,:, 0:3]
        dimen = 3
        
    gray = rgb2gray(img)
    gray = skimage.img_as_ubyte(gray)
#     print(gray.shape)
    rgb = img.reshape(h * w, dimen)
    _, hog_img = hog(img, pixels_per_cell=(8,8), cells_per_block=(2, 2), visualize=True)

    mean = fast_glcm_mean(gray).reshape(h * w, 1)
    std = fast_glcm_std(gray).reshape(h * w, 1)
    contrast = fast_glcm_contrast(gray).reshape(h * w, 1)
    dissimilarity = fast_glcm_dissimilarity(gray).reshape(h * w, 1)
    homogeneity = fast_glcm_homogeneity(gray).reshape(h * w, 1)
    asm, ene = fast_glcm_ASM(gray)
    asm = asm.reshape(h * w, 1)
    ene = ene.reshape(h * w, 1)
    entropy = fast_glcm_entropy(gray).reshape(h * w, 1)
    max_ = fast_glcm_max(gray).reshape(h * w, 1)
    hog_img = hog_img.reshape(h * w, 1)

    exp_img = np.concatenate((rgb, mean, std, contrast, dissimilarity, homogeneity, asm, ene, entropy, max_, hog_img), axis=1)
    
    return exp_img
   
def training_data_generate(img_list, kind_list):
    # img_list = [io.imread(), io.imread(), io.imread()], element is ndarray
    # king_list = [0, 1, 2]
    train_data = np.empty((0, 13))
    train_kind = np.empty((0,))
    for img, kind in zip(img_list, kind_list):
        (h, w, dimen) = img.shape
        exp_img = expand_colorspace(img)
        
        if dimen == 4:
            img_reshape = img.reshape(h * w, dimen)
            train_index = np.nonzero(img_reshape[:, 3])  # the index of alpha layer not zero pixels
            exp_img = exp_img[train_index]  # pick the pixels that not background
        
        print(f"|--- Convert kind [{kind}] to training data, converted shape is {str(exp_img.shape)}")
        kinds = np.array([kind] * exp_img.shape[0]) 

        # merge each train_data and kind
        train_data = np.vstack((train_data, exp_img))
        train_kind = np.hstack((train_kind, kinds))

    return train_data, train_kind  
    
def train_model(train_data, train_kind, classifier="CART"):
    # classifier = ["CART", "SVM", "RF"]
    t0 = time.time()
    if classifier == "CART":
        clf = DecisionTreeClassifier(max_depth=20)
    elif classifier == "SVM":
        clf = LinearSVC()
    elif classifier == "RF":
        clf = RandomForestClassifier()
    elif classifier == "GDBT":
        clf = GradientBoostingClassifier()
        
    clf = clf.fit(train_data, train_kind)
    t1 = time.time()
    
    print(f'|-- Training model time cost={int(t1-t0)}s')
    return clf
    
def predict_model(clf_img, model):
    (h, w, dimen) = clf_img.shape
    exp_clf_img = expand_colorspace(clf_img)
    pred_result = model.predict(exp_clf_img).reshape(h, w)
    return pred_result
    
def apply_model(clf_img, model):
    t0 = time.time()
    (h, w, dimen) = clf_img.shape
    if h * w >= 1500 * 1500:  # slice picture to save RAM
        slice_num = np.ceil(w * h / (1500 * 1500)).astype(np.int32)
        break_points = np.linspace(0, h, num=slice_num).astype(np.int32)
        line_st = break_points[:-1]
        line_ed = break_points[1:]
    else:
        line_st = [0]
        line_ed = [h]
    
    pred_result = np.empty((0, w))
    i = 0
    for st, ed in zip(line_st, line_ed):
        clf_result = predict_model(clf_img[st:ed, :], model)
        pred_result = np.vstack((pred_result, clf_result))
        
    return pred_result
    
def save_result(pred_result, img_save_name, color_list):
    color_list = np.array(color_list)/255
    cmap_cf = ListedColormap(color_list, name='colorfriendly')
    plt.imsave(img_save_name, pred_result, cmap=cmap_cf)
    
def show_vfc(pred_result):
    (h, w) = pred_result.shape
    unique, counts = np.unique(pred_result, return_counts=True)
    total = h * w
    vfc = counts / total * 100
    
    pd_vfc = pd.Series(vfc)

    return pd_vfc
    
    
def error_matrix_generator(A,B):
    A = A + 1    # void 0
    B = B + 1    # void 0

    S = np.round(A / B, 1)    # count side
    '''
       | 1 | 2 | 3 |
    1  | - |0.5|0.3|
    2  | 2 | - |0.7|
    3  | 3 |1.5| - |
    '''
    C = np.round(A * B, 1)    # count center
    '''
       | 1 | 2 | 3 |
    1  | 1 | - | - |
    2  | - | 4 | - |
    3  | - | - | 9 |
    '''
    u_S,c_S = np.unique(S,return_counts=True)
    u_C,c_C = np.unique(C,return_counts=True)
    S_dict = dict(zip(u_S, c_S))
    C_dict = dict(zip(u_C, c_C))

    error_matrix = np.zeros((3,3))
    for k in S_dict:
        if k == 0.5:
            error_matrix[0,1] = S_dict[k]
        if k == 0.3:
            error_matrix[0,2] = S_dict[k]
        if k == 0.7:
            error_matrix[1,2] = S_dict[k]
        if k == 2:
            error_matrix[1,0] = S_dict[k]
        if k == 3:
            error_matrix[2,0] = S_dict[k]
        if k == 1.5:
            error_matrix[2,1] = S_dict[k]
    for k in C_dict:
        if k == 1:
            error_matrix[0,0] = C_dict[k]
        if k == 4:
            error_matrix[1,1] = C_dict[k]
        if k == 9:
            error_matrix[2,2] = C_dict[k]
    return error_matrix

def accuracy_calculator(error_matrix):
    center = np.diag(error_matrix)
    row = error_matrix.sum(axis=0)
    col = error_matrix.sum(axis=1)
    N = error_matrix.sum().sum()
    
    producer_accuracy = np.round(center / row, 3)
    user_accuracy = np.round(center / col, 3)
    average_accuracy = np.round(producer_accuracy.mean(), 3)
    overall_accuracy = (producer_accuracy * row).sum() / N
    total_accuracy = round(center.sum() / N, 3)
    
    mult = row * col
    kappa = (N * center.sum() - mult.sum()) / (N ** 2 - mult.sum())
    
    output_matrix = np.column_stack((error_matrix, col, user_accuracy))
    output_matrix = np.row_stack((output_matrix, [np.append(np.append(row, N), np.nan)]))
    output_matrix = np.row_stack((output_matrix, [np.append(producer_accuracy, [np.nan,average_accuracy])]))
    
    if error_matrix.shape[0] == 3:
        output_dataframe = pd.DataFrame(output_matrix,
                                        index=[['Classified','Classified','Classified','Classified','Producers'],
                                               ['Sand','Plant','Grass','Total','Accuracy']],
                                        columns = [['Reference','Reference','Reference','Reference','Users'],
                                                   ['Sand','Plant','Grass','Total','Accuracy']])
    else:
        output_dataframe = pd.DataFrame(output_matrix,
                                index=[['Classified','Classified','Classified','Producers'],
                                       ['0','1','Total','Accuracy']],
                                columns = [['Reference','Reference','Reference','Users'],
                                           ['0','1','Total','Accuracy']])
    return output_dataframe, overall_accuracy, kappa
    
def accuracy_assessment(pred_result, visual_int):
    classifier_int = pred_result[7300:8300,6300:7300]
    
    error_matrix = error_matrix_generator(classifier_int,visual_int)
    output_dataframe, overall_accuracy, kappa = accuracy_calculator(error_matrix)
    print(output_dataframe)
    print('---------------')
    print('overall accuracy = ' + str(round(overall_accuracy, 3)))
    print('kappa = ' + str(round(kappa, 4)))
    
def png2int(img_dir, color_dict):
    # {0:[0,255,0], 1:[0,0,255], 2:[255,255,255]}
    visual_ndarray = io.imread(img_dir)
    visual_int = np.zeros(visual_ndarray.shape[:2])
    for kind in color_dict.keys():
        # visual_int[np.all(visual_ndarray[:,:,:3] == [0,255,0], axis=2)] = 0  
        visual_int[np.all(visual_ndarray[:,:,:3] == color_dict[kind], axis=2)] = kind
        
    return visual_int