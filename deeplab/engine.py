import os
import copy

import numpy as np

from tqdm import tqdm
# from sklearn.metrics import f1_score, roc_auc_score
from prefetch_generator import BackgroundGenerator

import torch

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader


from model import createDeepLabv3
from dataset import Broccoli

seed = 1123
torch.manual_seed(seed)

class DataloaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class deeplab_engine:
    def __init__(self, arg, device):

        self.coco_label_path = arg.coco_label_path
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

        self.init_model()
    
    def init_model(self):
        print("initilizing network\n")
        
        self.model = createDeepLabv3().to(self.device)

        # self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(self.beta_1, self.beta_2))
        self.optim = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        

        self.criterian = torch.nn.MSELoss(reduction='mean')
        
        self.transform = transforms.Compose([
            # transforms.RandomResizedCrop(128, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
            # transforms.RandomRotation((-90,90)),
            # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
            # transforms.RandomHorizontalFlip(p=0.8),
            # transforms.RandomVerticalFlip(p=0.8),
            # transforms.RandomAffine((-5, 5)),
            # # transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.train_ = Broccoli(coco_label_path=self.coco_label_path, size=self.im_size)
        # self.valid_ = Broccoli(img_dir=self.img_dir, mask_dir=self.mask_dir, size=self.im_size, data_type="validation")

        self.dataloader_train = DataloaderX(
            self.train_,
            batch_size=self.batch_size,
            num_workers=4,
            shuffle=True
        )
        
        print("initilization done\n")


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
        for epoch in range(self.n_epochs):
            # Dataloader returns the batches
            batch_loss = 0.
            mious = 0.
            f1 = 0.
            ras = 0.
            len_train = len(self.dataloader_train)
            for sample in tqdm(self.dataloader_train):

                img = sample["image"].to(self.device)
                mask = sample["mask"].to(self.device)

                ## Train ##
                self.optim.zero_grad()

                pred = self.model(img)
                train_loss = self.criterian(pred["out"], mask)

                
                y_pred = pred['out'].data.cpu().numpy().ravel()
                y_true = mask.data.cpu().numpy().ravel()
                
                pred_masks = pred['out'].detach().cpu().numpy()
                pred_masks[pred_masks < 0.5] = 0
                pred_masks[pred_masks >= 0.5] = 1
                
                ground_truth = mask.detach().cpu().numpy()
                
                # Use a classification threshold of 0.1
                # f1_score_ = f1_score(y_true > 0, y_pred > 0.1)         
                # roc_auc_score_ = roc_auc_score(y_true.astype('uint8'), y_pred)
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
            # train_f1_scores += [f1 / len_train]
            # train_roc_auc_score += [ras / len_train]
                
            # # validation phase
            # self.model.eval()
            # with torch.no_grad():
            #     batch_loss = 0.
            #     mious = 0.
            #     f1 = 0.
            #     ras = 0.
            #     len_valid = len(self.dataloader_valid)
            #     for sample in tqdm(self.dataloader_valid):

            #         img = sample["image"].to(self.device)
            #         mask = sample["mask"].to(self.device)


            #         pred = self.model(img)
            #         valid_loss = self.criterian(pred["out"], mask)
                    
            #         y_pred = pred['out'].data.cpu().numpy().ravel()
            #         y_true = mask.data.cpu().numpy().ravel()
                    
            #         pred_masks = pred['out'].detach().cpu().numpy()
            #         pred_masks[pred_masks < 0.5] = 0
            #         pred_masks[pred_masks >= 0.5] = 1
                    
            #         ground_truth = mask.detach().cpu().numpy()
                    
            #         # Use a classification threshold of 0.1
            #         f1_score_ = f1_score(y_true > 0, y_pred > 0.1)
            #         roc_auc_score_ = roc_auc_score(y_true.astype('uint8'), y_pred)
            #         miou = self.iou_mean(pred_masks, ground_truth)
                    
            #         # Keep track of the losses
            #         batch_loss += valid_loss.item()
            #         mious += miou
            #         f1 += f1_score_
            #         ras += roc_auc_score_
                    
            #     valid_losses += [batch_loss / len_valid]
            #     valid_mious += [mious / len_valid]
            #     valid_f1_scores += [f1 / len_valid]
            #     valid_roc_auc_score += [ras / len_valid]


            ### Visualization code ###
            print(f"epoch {epoch+1}, Train loss: {train_losses[-1]:.3f}, Train mIOU: {train_mious[-1]:.3f}")

                
            if train_mious[-1] > best_miou:
                best_miou = train_mious[-1]
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
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
        torch.save(checkpoint,  f'{self.out_dir}/best_model.tar')  # overwrite if exist
        
        print('Training complete')
        print(f'Best mIOU: {best_miou:.3f}')
        
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    
    import args

    engine = deeplab_engine(args, device)

    engine.train()
