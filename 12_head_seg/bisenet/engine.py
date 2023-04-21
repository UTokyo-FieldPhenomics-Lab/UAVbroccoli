import os
import copy
import time
import json

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

from pycocotools.coco import COCO
from .bisenetv2 import BiSeNetV2
from .dataset import Broccoli

SEED = 1123
torch.manual_seed(SEED)
cudnn.benchmark = True

class OhemCELoss(nn.Module):

    def __init__(self, thresh):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.criteria = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, labels):
        n_min = labels.numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)

class bisenet_engine:
    def __init__(self, args):

        self.args = args
        self.annotation_path = args.annotation_path

        self.out_dir = args.ckpt_folder
        os.makedirs(args.ckpt_folder, exist_ok=True)
        
        self.n_epochs = args.number_epochs

        self.classes = args.classes
        self.n_classes = len(args.classes)

        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.beta_1 = args.beta_1 # 0.5
        self.beta_2 = args.beta_2 # 0.999

        self.im_size = args.image_size
        self.device = args.device

        self.coco = COCO(args.coco_path)
        self.init_model()
        
    def init_model(self):
        print("initilizing network")
        
        self.model =  BiSeNetV2(n_classes=2).to(self.device)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        # self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.5, 0.9))
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim, milestones=[80,120,140,160, 180], gamma=0.5)

        # self.criterian = torch.nn.MSELoss(reduction='mean')
        self.criterian = nn.CrossEntropyLoss(reduction='mean')

        self.criteria_pre = OhemCELoss(0.7)
        self.criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
        
        transform = A.Compose(
            [
                A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.2, rotate_limit=90, p=0.5),
                A.VerticalFlip(p=0.5),              
                A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                ToTensorV2(),
            ]
        )
        
        self.train_ = Broccoli(coco=self.coco, size=self.im_size, transforms=transform)
        # self.valid_ = Broccoli(img_dir=self.img_dir, mask_dir=self.mask_dir, size=self.im_size, data_type="validation")

        self.dataloader_train = DataLoader(
            self.train_,
            batch_size=self.batch_size,
            # num_workers=4,
            shuffle=True,
            drop_last=False
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
        if ckpt != None:
            # load_check_point
            checkpoint_dir = ckpt
            checkpoint = torch.load(checkpoint_dir)
            model_weight = checkpoint["model_state_dict"]
            self.model.load_state_dict(model_weight)
            base = checkpoint["epoch"]
        else:
            base = 0


        train_losses = []
        train_mious = []
        # train_f1_scores = []
        # train_roc_auc_score = []
        
        best_miou = 0.
        best_model_wts = copy.deepcopy(self.model.state_dict())
        
        # train phase
        self.model.train()
        start = time.time()
        for epoch in range(base, base + self.n_epochs):
            # Dataloader returns the batches
            batch_loss = 0.
            mious = 0.

            len_train = len(self.dataloader_train)
            print(len_train)
            for img, mask in tqdm(self.dataloader_train):
                print(np.unique(mask))
                ## Train ##
                self.optim.zero_grad()
                img = img.to(self.device)
                # print(np.unique(mask))
                # mask = mask.to(self.device, dtype=torch.long)
                mask = mask.to(self.device, dtype=torch.long)
                # mask = mask.view(-1, 1, 128, 128)
                # print(mask[0])
                pred_masks, *logits_aux = self.model(img)

                loss_pre = self.criteria_pre(pred_masks, mask)
                loss_aux = [crit(lgt, mask) for crit, lgt in zip(self.criteria_aux, logits_aux)]
                train_loss = loss_pre + sum(loss_aux)

                pred_ = pred_masks.argmax(1).detach().cpu().numpy()
                # pred_masks[pred_masks < 0.5] = 0
                # pred_masks[pred_masks >= 0.5] = 1
                # print(pred_masks.shape)
                ground_truth = mask.detach().cpu().numpy()
                
                temp = mask.clone().view((-1, 1, 128, 128))
                # print(temp.max())
                
                miou = self.iou_mean(pred_, ground_truth)
                

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

            self.lr_scheduler.step()

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

            image_tensor = (pred_masks.argmax(1)*255).view(-1, 1, 128 ,128)
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
            

            if not os.path.exists(self.args.temp_results):
                os.makedirs(self.args.temp_results)
            plt.savefig(f'{self.args.temp_results}/epoch_{epoch+1}.jpg')
            # plt.show()
            plt.close()
            
            print(f"epoch {epoch+1}, Train loss: {train_losses[-1]:.3f}, Train mIOU: {train_mious[-1]:.3f}")

                
            if train_mious[-1] > best_miou:
                best_miou = train_mious[-1]
                best_model_wts = copy.deepcopy(self.model.state_dict())
        end = time.time()        
        print(f"total time cost: {end - start} s")
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
    
    engine = bisenet_engine(args, device)
    engine.train()
    # engine.train(ckpt='./checkpoints/v4/epoch200.tar')
    # engine.train(ckpt='./ckpt/v4/epoch500.tar')
