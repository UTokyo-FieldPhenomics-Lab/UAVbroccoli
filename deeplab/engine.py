import os
import sys
import copy
import time
import json
from albumentations.augmentations.geometric.resize import Resize

import numpy as np

from tqdm import tqdm
# from sklearn.metrics import f1_score, roc_auc_score
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from model import createDeepLabv3
from dataset import Broccoli
from pycocotools.coco import COCO
sys.path.append("..") 
from utils.labelme2coco import labelme2json

import args

seed = 1123
torch.manual_seed(seed)
cudnn.benchmark = True


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class deeplab_engine:
    def __init__(self, arg, device):

        self.json_path = arg.json_path
    
        self.out_dir = arg.out_dir
        os.makedirs(arg.out_dir, exist_ok=True)
        
        self.n_epochs = arg.n_epochs
        self.n_classes = arg.n_classes

        self.batch_size = arg.batch_size
        self.lr = arg.lr
        self.beta_1 = arg.beta_1 # 0.5
        self.beta_2 = arg.beta_2 # 0.999

        self.im_size = arg.im_size

        self.classes = ["broccoli"]

        self.device = device
        self.read_coco()
        self.init_model()
    
    def read_coco(self):
        json_list = [entry.name for entry in os.scandir(self.json_path) if ((entry.name.endswith('.json')) & (entry.name.startswith('labeled_')))]
        print('initializing training set...')
        print(f'{len(json_list)} labeled json files found')
        coco_label = labelme2json(json_list)
        with open('./temp.json', 'w+') as f:
            f.write(json.dumps(coco_label))
        
        self.coco = COCO('./temp.json')
        
    def init_model(self):
        print("initilizing network")
        
        self.model = createDeepLabv3(pretrained=True).to(self.device)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        # self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        

        # self.criterian = torch.nn.MSELoss(reduction='mean')
        self.criterian = nn.CrossEntropyLoss(reduction='mean')
        # self.criterian = FocalLoss(size_average=True)
        # self.criterian = MixSoftmaxCrossEntropyLoss()
        
        transform = A.Compose(
            [
                # A.Resize(128, 128),
                # A.VerticalFlip(p=0.5),              
                # A.RandomRotate90(p=0.5),
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        
        self.train_ = Broccoli(coco=self.coco, size=self.im_size, transforms=transform, save_data=False)
        # self.valid_ = Broccoli(img_dir=self.img_dir, mask_dir=self.mask_dir, size=self.im_size, data_type="validation")

        self.dataloader_train = DataLoaderX(
            self.train_,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True
        )
        
        print("initilization done")


    def iou_mean(self, pred, target, n_classes=1):
        #n_classes ：the number of classes in your dataset,not including background
        # for mask and ground-truth label, not probability map
        ious = []
        iousSum = 0
        pred = torch.from_numpy(pred)
        pred = pred.view(-1)
        target = np.array(target)
        target = torch.from_numpy(target)
        target = target.view(-1)

        # Ignore IoU for background class ("0")
        for cls in range(1, n_classes+1):  # This goes from 1:n_classes-1 -> class "0" is ignored
            pred_inds = pred == cls
            target_inds = target == cls
            intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
            union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
            if union == 0:
                ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
            else:
                ious.append(float(intersection) / float(max(union, 1)))
                iousSum += float(intersection) / float(max(union, 1))
        return iousSum/n_classes
    
    def train(self, ckpt=None):
        # if ckpt != None:
        #     # load_check_point
        #     checkpoint_dir = ckpt
        #     checkpoint = torch.load(checkpoint_dir)
        #     gen_weight = checkpoint["G_state_dict"]
        #     crit_weight = checkpoint["D_state_dict"]
        #     self.gen.load_state_dict(gen_weight)
        #     self.crit.load_state_dict(crit_weight)
        #     base = checkpoint["epoch"]
        # else:
        #     def weights_init(m):
        #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        #             torch.nn.init.normal_(m.weight, 0.0, 0.02)
        #         if isinstance(m, nn.BatchNorm2d):
        #             torch.nn.init.normal_(m.weight, 0.0, 0.02)
        #             torch.nn.init.constant_(m.bias, 0)
        #     gen = self.gen.apply(weights_init)
        #     crit = self.crit.apply(weights_init)
        #     base = 0


        train_losses = []
        train_mious = []
        # train_f1_scores = []
        # train_roc_auc_score = []
        
        best_miou = 0.
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # train phase
        self.model.train()
        start = time.time()
        for epoch in range(self.n_epochs):
            # Dataloader returns the batches
            batch_loss = 0.
            mious = 0.

            len_train = len(self.dataloader_train)
            for img, mask in tqdm(self.dataloader_train):
                # print(np.unique(mask))
                ## Train ##
                self.optim.zero_grad()
                img = img.to(self.device)
                # print(np.unique(mask))
                # mask = mask.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                # mask = mask.view(-1, 1, 128, 128)
                # print(mask[0])
                pred = self.model(img)
                
                pred_masks = pred["out"]
                
                train_loss = self.criterian(pred["out"], mask)

                pred_masks = pred_masks.argmax(1).detach().cpu().numpy()
                # pred_masks[pred_masks < 0.5] = 0
                # pred_masks[pred_masks >= 0.5] = 1
                # print(pred_masks.shape)
                ground_truth = mask.detach().cpu().numpy()
                
                temp = mask.clone().view((-1, 1, 128, 128))
                # print(temp.max())
                
                miou = self.iou_mean(pred_masks, ground_truth)
                

                # Update gradients
                train_loss.backward()
                
                # Update optimizer
                self.optim.step()

                # Keep track of the losses
                batch_loss += train_loss.item()
                mious += miou
                # f1 += f1_score_
                # ras += roc_auc_score_
                
            train_losses += [batch_loss / len_train]
            train_mious += [mious / len_train]

            ### Visualization code ###
            fig = plt.figure(figsize=(30,10))

            ax1 = fig.add_subplot(131)
            ax1.set_xticks([])
            ax1.set_yticks([])
            image_tensor = (img+1)/2
            image_unflat = image_tensor.detach().cpu()
            image_grid = make_grid(image_unflat[:16], nrow=4)
            ax1.imshow(image_grid.permute(1, 2, 0).squeeze())

            ax2 = fig.add_subplot(132)
            ax2.set_xticks([])
            ax2.set_yticks([])

            image_tensor = (pred["out"].argmax(1)*255).view(-1, 1, 128 ,128)
            image_unflat = image_tensor.detach().cpu()
            image_grid = make_grid(image_unflat[:16], nrow=4)
            ax2.imshow(image_grid.permute(1, 2, 0).squeeze())
            
            ax3 = fig.add_subplot(133)
            ax3.set_xticks([])
            ax3.set_yticks([])

            image_tensor = temp*255
            image_unflat = image_tensor.detach().cpu()
            image_grid = make_grid(image_unflat[:16], nrow=4)
            ax3.imshow(image_grid.permute(1, 2, 0).squeeze())
            

            if not os.path.exists(f'./seg_results'):
                os.makedirs(f'./seg_results')
            plt.savefig(f'./seg_results/epoch_{epoch+1}.jpg')
            # plt.show()
            plt.close()
            
            print(f"epoch {epoch+1}, Train loss: {train_losses[-1]:.3f}, Train mIOU: {train_mious[-1]:.3f}")

                
            if train_mious[-1] > best_miou:
                best_miou = train_mious[-1]
                best_model_wts = copy.deepcopy(self.model.state_dict())
        end = time.time()        
        print(f"total time cose: {end - start} s")
        print(f"saving model")
        ## Save Model ##
        checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'loss_train': np.array(train_losses),
                # 'loss_valid': np.array(valid_losses),
                'miou_train': np.array(train_mious),
                # 'miou_valid': np.array(valid_mious),
                # 'f1_train': np.array(train_f1_scores),
                # 'f1_valid': np.array(valid_f1_scores),
                # 'rac_train': np.array(train_roc_auc_score),
                # 'rac_valid': np.array(valid_roc_auc_score),
                'epoch': epoch + 1
            }
        torch.save(checkpoint, f'{self.out_dir}/epoch{epoch+1}.tar')  # overwrite if exist
        
        best_model = {
            'model_state_dict': best_model_wts,
            'loss_train': np.array(train_losses),
            # 'loss_valid': np.array(valid_losses),
            'miou_train': np.array(train_mious),
            # 'miou_valid': np.array(valid_mious),
            # 'f1_train': np.array(train_f1_scores),
            # 'f1_valid': np.array(valid_f1_scores),
            # 'rac_train': np.array(train_roc_auc_score),
            # 'rac_valid': np.array(valid_roc_auc_score),
            'epoch': epoch + 1
        }
        torch.save(best_model,  f'{self.out_dir}/best_model.tar')  # overwrite if exist
        
        print('Training complete')
        print(f'Best mIOU: {best_miou:.3f}')
        
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    engine = deeplab_engine(args, device)

    engine.train()
