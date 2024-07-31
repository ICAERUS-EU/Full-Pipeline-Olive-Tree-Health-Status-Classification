# DUAL YOLO MODEL m
# yolov8n-cls structure 

#   # [depth, width, max_channels]
#   n: [0.33, 0.25, 1024]
#   s: [0.33, 0.50, 1024]
#   m: [0.67, 0.75, 1024]
#   l: [1.00, 1.00, 1024]
#   x: [1.00, 1.25, 1024]

# # YOLOv8.0n backbone
# backbone:
#   # [from, repeats, module, args]
#   - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
#   - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
#   - [-1, 3, C2f, [128, True]]
#   - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
#   - [-1, 6, C2f, [256, True]]
#   - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
#   - [-1, 6, C2f, [512, True]]
#   - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
#   - [-1, 3, C2f, [1024, True]]

# # YOLOv8.0n head
# head:
#   - [-1, 1, Classify, [nc]] # Classify


import os
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import  Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from PIL import Image
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt


torch.set_float32_matmul_precision('high')
torch.cuda.empty_cache()


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Classify(nn.Module):
    """YOLOv8 classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, dropout=0.3):
        """Initializes YOLOv8 classification head with specified input and output channels, kernel size, stride,
        padding, and groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class YOLOv8Partial(nn.Module):
    """Partial YOLOv8 model definition up to the last C2f layer."""

    def __init__(self, depth_scale=0.33, width_scale=0.25, max_channels=1024):
        super().__init__()
        self.backbone = nn.Sequential(
            Conv(3, int(64 * width_scale), 3, 2),
            Conv(int(64 * width_scale), int(128 * width_scale), 3, 2),
            C2f(int(128 * width_scale), int(128 * width_scale), 3, True),
            Conv(int(128 * width_scale), int(256 * width_scale), 3, 2),
            C2f(int(256 * width_scale), int(256 * width_scale), 6, True),
            Conv(int(256 * width_scale), int(512 * width_scale), 3, 2),
            C2f(int(512 * width_scale), int(512 * width_scale), 6, True),
            Conv(int(512 * width_scale), min(max_channels, int(1024 * width_scale)), 3, 2),
            C2f(min(max_channels, int(1024 * width_scale)), min(max_channels, int(1024 * width_scale)), 3, True),
        )

    def forward(self, x):
        x = self.backbone(x)
        return x


class DualYOLO(pl.LightningModule):
    """Model that takes two images as input and combines their features using a Classify head."""

    def __init__(self,class_weights,depth_scale=0.67, width_scale=0.75, lr=1e-3,num_classes=3,save_dir='./'):
        super().__init__()

        self.yolo1 = YOLOv8Partial(depth_scale=depth_scale, width_scale=width_scale)
        self.yolo2 = YOLOv8Partial(depth_scale=depth_scale, width_scale=width_scale)
        self.classify = Classify(int(width_scale*2048), c2=num_classes)

        self.lr = lr
        self.class_weights = class_weights
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.validation_step_outputs = []
        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

    def forward(self, img1, img2):
        """Forward pass through the dual YOLO model."""
        features1 = self.yolo1(img1)
        features2 = self.yolo2(img2)
        combined_features = torch.cat((features1, features2), dim=1)  # Concatenate features along the channel dimension
        output = self.classify(combined_features)  # Apply the classification head
        return output

    def training_step(self, batch, batch_idx):
        img1, img2, labels = batch
        predictions = self(img1, img2)
        
        loss = F.cross_entropy(predictions, labels, weight=self.class_weights)

        # Calculate precision, recall, and accuracy
        preds = torch.argmax(predictions, dim=1)
        self.log('t_loss', loss,  on_step=False,on_epoch=True)
        precision_weighted = precision_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
        recall_weighted = recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
        precision_micro = precision_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        recall_micro = recall_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        accuracy = accuracy_score(labels.cpu(), preds.cpu())
        precision_macro = precision_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        precision_per_class = precision_score(labels.cpu(), preds.cpu(), average=None, zero_division=0)
        recall_per_class = recall_score(labels.cpu(), preds.cpu(), average=None, zero_division=0)
        recall_macro = recall_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        self.log('t_prec_M', precision_macro, on_step=False, on_epoch=True)
        self.log('t_prec_weighted', precision_weighted, on_epoch=True)
        self.log('t_rec_M', recall_macro, on_step=False, on_epoch=True)
        self.log('t_rec_weighted', recall_weighted, on_step=False, on_epoch=True)
        #self.log('t_prec_micro', precision_micro, on_step=False, on_epoch=True)
        #self.log('t_rec_micro', recall_micro, on_step=False, on_epoch=True)
        self.log('t_acc', accuracy, on_step=False, on_epoch=True)
        for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
          self.log(f't_prec_class_{i}', precision, on_step=False, on_epoch=True)
          self.log(f't_rec_class_{i}', recall,  on_step=False,on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        img1, img2, labels = batch
        predictions = self(img1, img2)
        loss = F.cross_entropy(predictions, labels) # no weighted in val

        self.log('val_loss', loss,on_step=False, on_epoch=True)
        preds = torch.argmax(predictions, dim=1)
        precision_weighted = precision_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
        recall_weighted = recall_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
        precision_micro = precision_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        precision_macro = precision_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        recall_micro = recall_score(labels.cpu(), preds.cpu(), average='micro', zero_division=0)
        recall_macro = recall_score(labels.cpu(), preds.cpu(), average='macro',zero_division=1)#, zero_division=0)
        accuracy = accuracy_score(labels.cpu(), preds.cpu())

        f1_macro = f1_score(labels.cpu(), preds.cpu(), average='macro', zero_division=0)
        f1_weighted = f1_score(labels.cpu(), preds.cpu(), average='weighted', zero_division=0)
        precision_per_class = precision_score(labels.cpu(), preds.cpu(), average=None)#, zero_division=0)
        recall_per_class = recall_score(labels.cpu(), preds.cpu(), average=None)#, zero_division=0)
        self.log('val_prec_M', precision_macro, on_step=False, on_epoch=True)
        #self.log('val_prec_w', precision_weighted, on_step=False, on_epoch=True)
        #self.log('val_rec_w', recall_weighted, on_step=False, on_epoch=True)
        #self.log('val_prec_m', precision_micro, on_step=False, on_epoch=True)
        self.log('val_rec_M', recall_macro, on_step=False, on_epoch=True)
        #self.log('val_rec_m', recall_micro, on_step=False, on_epoch=True)
        self.log('val_acc', accuracy, on_step=False, on_epoch=True)
        self.log('val_f1', f1_macro, on_step=False, on_epoch=True)

        for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
          self.log(f'val_prec_class_{i}', precision,  on_step=False,on_epoch=True)
          self.log(f'val_rec_class_{i}', recall, on_step=False, on_epoch=True)

        self.validation_step_outputs.append({'preds': preds, 'labels': labels})
        self.val_conf_matrix.update(preds, labels)

        return loss

    def on_validation_epoch_end(self):
        all_preds = []
        all_labels = []

        for output in self.validation_step_outputs:
            preds = output['preds']
            labels = output['labels']
            all_preds.append(preds)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        cm = self.val_conf_matrix.compute().cpu().numpy()
        cm_df = pd.DataFrame(cm)

        if not isinstance(self.save_dir, str):
            raise ValueError(f"Expected self.save_dir to be a string, but got {type(self.save_dir)}")

        os.makedirs(self.save_dir, exist_ok=True)

        save_path = os.path.join(self.save_dir, 'confusion_matrix.csv')
        cm_df.to_csv(save_path, index=False)
        print(f"Confusion matrix saved to '{save_path}'")

        fig, ax = plt.subplots(figsize=(10, 7))
        cax = ax.matshow(cm, cmap=plt.cm.Blues)
        fig.colorbar(cax)

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center')

        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        plt.xticks(ticks=np.arange(len(cm)), labels=np.arange(len(cm)))
        plt.yticks(ticks=np.arange(len(cm)), labels=np.arange(len(cm)))

        img_path = os.path.join(self.save_dir, 'confusion_matrix.png')
        plt.savefig(img_path)
        plt.close()
        print(f"Confusion matrix image saved to '{img_path}'")

        self.validation_step_outputs.clear()
        self.val_conf_matrix.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=4, min_lr=1e-9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

    @classmethod
    def from_pretrained(cls, checkpoint_path, class_weights=None, depth_scale=0.67, width_scale=0.75,  lr=1e-3,num_classes=3, save_dir='./'):
        model = cls(class_weights, depth_scale, width_scale, lr,num_classes, save_dir)
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

        return model


class CustomDataset(Dataset):
    def __init__(self, health_dir_path, ndvi_path, rgb_path, transform=None, val_samples_per_file=0, seed=42,save_dir='./'):
        self.data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.ndvi_path = ndvi_path
        self.rgb_path = rgb_path
        self.val_samples_per_file = val_samples_per_file
        self.seed = seed
        self.save_dir=save_dir

        # Set the random seed
        self.set_seed(self.seed)

        if os.path.isdir(health_dir_path):
            self._process_directory(health_dir_path)
        elif os.path.isfile(health_dir_path) and health_dir_path.endswith('.csv'):
            self._process_file(health_dir_path)
        else:
            raise ValueError(f"The path {health_dir_path} is not a valid directory or CSV file.")

        self.transform = transform

        if self.data.empty or (self.val_data.empty and val_samples_per_file != 0):
            raise ValueError("The DataFrame is empty. Make sure the folder contains valid CSV files.")

        # Dynamically set attributes based on val_samples_per_file
        if val_samples_per_file == 0:
            self.data_labels = [label - 1 for label in self.data.iloc[:, 0].tolist()]
            self.data_img1_paths = self.data.iloc[:, 1].tolist()
            self.data_img2_paths = self.data.iloc[:, 2].tolist()
        else:
            self.train_labels = [label - 1 for label in self.data.iloc[:, 0].tolist()]
            self.val_labels = [label - 1 for label in self.val_data.iloc[:, 0].tolist()]
            self.train_img1_paths = self.data.iloc[:, 1].tolist()
            self.val_img1_paths = self.val_data.iloc[:, 1].tolist()
            self.train_img2_paths = self.data.iloc[:, 2].tolist()
            self.val_img2_paths = self.val_data.iloc[:, 2].tolist()
            self.save_validation_set()

    def _process_directory(self, dir_path):
        for file in os.listdir(dir_path):
            if file.endswith('.csv'):
                file_path = os.path.join(dir_path, file)
                self._process_file(file_path)

    def _process_file(self, file_path):
        csv_file = pd.read_csv(file_path)
        if not csv_file.index.is_unique:
            raise ValueError(f"Non-unique index in file {file_path}")
        if len(csv_file) < self.val_samples_per_file:
            raise ValueError(f"File {file_path} has less than {self.val_samples_per_file} rows.")

        val_sample = csv_file.sample(n=self.val_samples_per_file, random_state=1)
        train_sample = csv_file.drop(val_sample.index)

        self.val_data = pd.concat([self.val_data, val_sample], ignore_index=True)
        self.data = pd.concat([self.data, train_sample], ignore_index=True)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __len__(self):
        if self.val_samples_per_file == 0:
            return len(self.data_labels)
        else:
            return len(self.train_labels)

    def val_len(self):
        return len(self.val_labels)

    def __getitem__(self, idx, val=False):
        try:
            if self.val_samples_per_file == 0 and not val:
                if idx >= len(self.data_labels):
                    raise IndexError(f"Index {idx} out of range for data with length {len(self.data_labels)}")
                img1_path = os.path.join(self.ndvi_path, self.data_img1_paths[idx])
                img2_path = os.path.join(self.rgb_path, self.data_img2_paths[idx])
                label = torch.tensor(self.data_labels[idx], dtype=torch.long)
            elif val:
                if idx >= len(self.val_labels):
                    raise IndexError(f"Index {idx} out of range for validation data with length {len(self.val_labels)}")
                img1_path = os.path.join(self.ndvi_path, self.val_img1_paths[idx])
                img2_path = os.path.join(self.rgb_path, self.val_img2_paths[idx])
                label = torch.tensor(self.val_labels[idx], dtype=torch.long)
            else:
                if idx >= len(self.train_labels):
                    raise IndexError(f"Index {idx} out of range for training data with length {len(self.train_labels)}")
                img1_path = os.path.join(self.ndvi_path, self.train_img1_paths[idx])
                img2_path = os.path.join(self.rgb_path, self.train_img2_paths[idx])
                label = torch.tensor(self.train_labels[idx], dtype=torch.long)

            img1 = self.load_image(img1_path)
            img2 = self.load_image(img2_path)

            if self.transform:
                img1 = self.transform(img1)
                img2 = self.transform(img2)

            return img1, img2, label
        except IndexError as e:
            print(f"Index error: {e}, idx: {idx}")
            raise
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise

    def load_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        image = Image.open(path)
        return image.convert('RGB')

    def get_val_item(self, idx):
        return self.__getitem__(idx, val=True)

    def get_train_labels(self):
        if self.val_samples_per_file == 0:
            return self.data_labels
        else:
            return self.train_labels

    def save_validation_set(self):
        true_val_labels = [label + 1 for label in self.val_labels]
        val_df = pd.DataFrame({
            'label': true_val_labels,
            'img1_path': self.val_img1_paths,
            'img2_path': self.val_img2_paths
        })
        val_df.to_csv(os.path.join(self.save_dir,'validation_set.csv'), index=False)
        print(f"Validation set saved to {self.save_dir}")


class ValDataset(Dataset):
    def __init__(self, custom_dataset):
        self.custom_dataset = custom_dataset

    def __len__(self):
        return self.custom_dataset.val_len()

    def __getitem__(self, idx):
        return self.custom_dataset.get_val_item(idx)

    def get_val_img1_paths(self):
        return self.custom_dataset.val_img1_paths

    def get_val_img2_paths(self):
        return self.custom_dataset.val_img2_paths

