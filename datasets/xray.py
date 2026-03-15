from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv
import os
import torch
import random
import numpy as np
from PIL import Image
from .tokenizers import Tokenizer
from .utils import nested_tensor_from_tensor_list, read_json
import json


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


def get_transform(MAX_DIM):
    def under_max(image):
        if image.mode != 'RGB':
            image = image.convert("RGB")

        shape = np.array(image.size, dtype=float)
        long_dim = max(shape)
        scale = MAX_DIM / long_dim

        new_shape = tuple((shape * scale).astype(int))
        image = image.resize(new_shape)

        return image

    train_transform = tv.transforms.Compose([
        RandomRotation(),
        tv.transforms.Lambda(under_max),
        tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
            0.8, 1.5], saturation=[0.2, 1.5]),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    val_transform = tv.transforms.Compose([
        tv.transforms.Lambda(under_max),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    return train_transform, val_transform


transform_class = tv.transforms.Compose([
    tv.transforms.Resize(224),
    tv.transforms.CenterCrop((224, 224)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class XrayDataset(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=None, transform_class=transform_class,
                 mode='training', data_dir=None, dataset_name=None, image_size=None,
                 theta=None, gamma=None, beta=None):
        super().__init__()

        self.root = root
        self.transform = transform
        self.transform_class = transform_class
        self.annot = ann

        self.data_dir = data_dir
        self.image_size = image_size

        self.theta = theta
        self.gamma = gamma
        self.beta = beta

        if mode == 'training':
            self.annot = self.annot[:]
        else:
            self.annot = self.annot[:]
        if dataset_name == "mimic_cxr":
            threshold = 10
        elif dataset_name == "iu_xray":
            threshold = 3
        self.data_name = dataset_name
        self.tokenizer = Tokenizer(ann_path=root, threshold=threshold, dataset_name=dataset_name)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        caption = self.annot[idx]["report"]
        image_path = self.annot[idx]['image_path']
        image = Image.open(os.path.join(self.data_dir, image_path[0])).resize((300, 300)).convert('RGB')
        class_image = image
        com_image = image

        if self.data_name == "mimic_cxr":
            # mask_arr = np.load(os.path.join(self.data_dir.strip("images300"), "images300_array",
            #                                 image_path[0].replace(".jpg", ".npy")))
            npy_rel_path = image_path[0].replace(".jpg", ".npy")

            base_dir = self.data_dir.replace("images300", "images300_array")
            mask_arr = np.load(os.path.join(base_dir, npy_rel_path))
        else:
            mask_arr = np.load(os.path.join(self.data_dir.strip("images"), "resnet34_300/images300_array",
                                            image_path[0].replace(".png", ".npy")))

        if (np.sum(mask_arr) / 90000) > self.theta:
            image_arr = np.asarray(image)
            boost_arr = image_arr * np.expand_dims(mask_arr, 2)
            weak_arr = image_arr * np.expand_dims(1 - mask_arr, 2)
            image = Image.fromarray(boost_arr + (weak_arr * self.gamma).astype(np.uint8))

        if self.transform:
            image = self.transform(image)
            com_image = self.transform(com_image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0), max_dim=self.image_size)
        com_image = nested_tensor_from_tensor_list(com_image.unsqueeze(0), max_dim=self.image_size)

        if self.transform_class:
            class_image = self.transform_class(class_image)

        caption = self.tokenizer(caption)[:self.max_length]
        cap_mask = [1] * len(caption)
        
        return image.tensors.squeeze(0), image.mask.squeeze(0), com_image.tensors.squeeze(0), com_image.mask.squeeze(
            0), caption, cap_mask, class_image

    @staticmethod
    def collate_fn(data):
        max_length = 129
        image_batch, image_mask_batch, com_image_batch, com_image_mask_batch, report_ids_batch, report_masks_batch, class_image_batch = zip(
            *data)
        image_batch = torch.stack(image_batch, 0)
        image_mask_batch = torch.stack(image_mask_batch, 0)
        com_image_batch = torch.stack(com_image_batch, 0)
        com_image_mask_batch = torch.stack(com_image_mask_batch, 0)
        class_image_batch = torch.stack(class_image_batch, 0)
        target_batch = np.zeros((len(report_ids_batch), max_length), dtype=int)
        target_masks_batch = np.zeros((len(report_ids_batch), max_length), dtype=int)

        for i, report_ids in enumerate(report_ids_batch):
            target_batch[i, :len(report_ids)] = report_ids

        for i, report_masks in enumerate(report_masks_batch):
            target_masks_batch[i, :len(report_masks)] = report_masks
        target_masks_batch = 1 - target_masks_batch

        return image_batch, image_mask_batch, com_image_batch, com_image_mask_batch, torch.tensor(
            target_batch), torch.tensor(target_masks_batch, dtype=torch.bool), class_image_batch


def build_dataset(config, mode='training', anno_path=None, data_dir=None, dataset_name=None, image_size=None,
                  theta=None, gamma=None, beta=None):
    train_transform, val_transform = get_transform(MAX_DIM=image_size)
    if mode == 'training':
        train_file = anno_path
        data = XrayDataset(train_file, read_json(
            train_file)["train"], max_length=config.max_position_embeddings, limit=config.limit,
                           transform=train_transform,
                           mode='training', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data

    elif mode == 'validation':
        val_file = anno_path
        data = XrayDataset(val_file, read_json(
            val_file)["val"], max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='validation', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data
    elif mode == 'test':
        test_file = anno_path
        data = XrayDataset(test_file, read_json(
            test_file)["test"], max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform,
                           mode='test', data_dir=data_dir, dataset_name=dataset_name, image_size=image_size,
                           theta=theta, gamma=gamma, beta=beta)
        return data
    else:
        raise NotImplementedError(f"{mode} not supported")


# FFA-IR
# class XrayDataset(Dataset):
#     def __init__(self, root, ann_data, max_length, limit, transform=None, transform_class=transform_class,
#                  mode='training', data_dir=None, dataset_name=None, image_size=None,
#                  theta=None, gamma=None, beta=None, max_images=32):
#         super().__init__()

#         self.root = root
#         self.transform = transform
#         self.transform_class = transform_class
#         self.annot = ann_data # 这里传入的已经是切分好的 list

#         self.data_dir = data_dir # 图片根目录，例如 .../FFAIR_1
#         self.image_size = image_size
#         self.theta = theta
#         self.gamma = gamma
#         self.beta = beta
#         self.max_images = max_images 

#         self.mode = mode
#         self.tokenizer = Tokenizer(ann_path=root, threshold=3, dataset_name=dataset_name)
#         self.max_length = max_length + 1

#     def __len__(self):
#         return len(self.annot)

#     def __getitem__(self, idx):
#         entry = self.annot[idx]
#         caption = entry["Finding-English"] 
#         folder_name = entry["Path"]

#         patient_folder_path = os.path.join(self.data_dir, folder_name)

#         try:
#             all_files = os.listdir(patient_folder_path)
#             # 过滤出图片并按文件名排序 (image_0.png, image_1.png...)
#             # 如果文件名数字位数不统一，直接 sort 可能有问题(1, 10, 2)，建议用 lambda x: int(x.split('_')[1].split('.')[0])
#             image_files = sorted([f for f in all_files if f.endswith('.png') or f.endswith('.jpg')], 
#                                  key=lambda x: int(x.replace('image_', '').replace('.png', '').replace('.jpg', '')) if 'image_' in x else x)
#         except FileNotFoundError:
#             print(f"Warning: Folder not found {patient_folder_path}")
#             image_files = []

#         if len(image_files) == 0:
#             # 容错：如果没有图片，造一张黑图
#             image_stack = [torch.zeros(3, 512, 512)]
#             com_image_stack = [torch.zeros(3, 512, 512)]
#             class_image = torch.zeros(3, 224, 224)
#             caption = self.tokenizer(caption)[:self.max_length]
#             return torch.stack(image_stack), torch.stack(com_image_stack), caption, [1]*len(caption), class_image

#         num_imgs = len(image_files)
#         if num_imgs > self.max_images:
#             indices = np.linspace(0, num_imgs - 1, self.max_images, dtype=int)
#             sampled_files = [image_files[i] for i in indices]
#         else:
#             sampled_files = image_files
#             while len(sampled_files) < self.max_images:
#                 sampled_files += image_files
#             sampled_files = sampled_files[:self.max_images]

#         image_stack = []
#         com_image_stack = []
        
#         # Mask 路径假设: .../FFAIR_1_results/patient_0/image_0_mask.npy
#         mask_root_dir = self.data_dir.replace("FFAIR_1", "FFAIR_1_results") 

#         for img_file in sampled_files:
#             # 加载原图
#             full_img_path = os.path.join(patient_folder_path, img_file)
#             image = Image.open(full_img_path).resize((512, 512)).convert('RGB')
#             image_arr = np.asarray(image)

#             # 加载 Mask
#             # 这里的路径拼接需要小心：path = patient_0/image_0_mask.npy
#             mask_rel_path = os.path.join(folder_name, img_file.replace(".png", "_mask.npy").replace(".jpg", "_mask.npy"))
#             mask_full_path = os.path.join(mask_root_dir, mask_rel_path)

#             if os.path.exists(mask_full_path):
#                 mask_arr = np.load(mask_full_path)
#                 if mask_arr.shape != (512, 512):
#                     import cv2
#                     mask_arr = cv2.resize(mask_arr.astype(float), (512, 512))
#                     mask_arr = mask_arr > 0 
                
#                 # Boost 增强逻辑
#                 if (np.sum(mask_arr) / (512*512)) > self.theta:
#                     boost_arr = image_arr * np.expand_dims(mask_arr, 2)
#                     weak_arr = image_arr * np.expand_dims(1 - mask_arr, 2)
#                     image = Image.fromarray((boost_arr + (weak_arr * self.gamma)).astype(np.uint8))
            
#             # Transform
#             com_img_item = image.copy()
#             if self.transform:
#                 image_tensor = self.transform(image)
#                 com_img_tensor = self.transform(com_img_item)
#             else:
#                 image_tensor = TF.to_tensor(image)
#                 com_img_tensor = TF.to_tensor(com_img_item)

#             image_stack.append(image_tensor)
#             com_image_stack.append(com_img_tensor)

#         images = torch.stack(image_stack, dim=0)       # [Seq, 3, 512, 512]
#         com_images = torch.stack(com_image_stack, dim=0)

#         mid_file = sampled_files[len(sampled_files)//2]
#         class_pil = Image.open(os.path.join(patient_folder_path, mid_file)).resize((224, 224)).convert('RGB')
#         class_image = self.transform_class(class_pil)

#         caption = self.tokenizer(caption)[:self.max_length]
#         cap_mask = [1] * len(caption)

#         return images, com_images, caption, cap_mask, class_image

#     @staticmethod
#     def collate_fn(data):
#         images_batch, com_images_batch, report_ids_batch, report_masks_batch, class_image_batch = zip(*data)
        
#         images_batch = torch.stack(images_batch, 0)
#         com_images_batch = torch.stack(com_images_batch, 0)
#         class_image_batch = torch.stack(class_image_batch, 0)

#         # Padding Reports
#         max_length = 129
#         batch_size = len(report_ids_batch)
#         target_batch = np.zeros((batch_size, max_length), dtype=int)
#         target_masks_batch = np.zeros((batch_size, max_length), dtype=int)

#         for i, report_ids in enumerate(report_ids_batch):
#             length = len(report_ids)
#             target_batch[i, :length] = report_ids
#             target_masks_batch[i, :length] = 1

#         return images_batch, com_images_batch, torch.tensor(target_batch), torch.tensor(target_masks_batch, dtype=torch.bool), class_image_batch



# def build_dataset(config, mode='training', anno_path=None, data_dir=None, dataset_name=None, image_size=None,
#                   theta=None, gamma=None, beta=None):
    
#     train_transform, val_transform = get_transform(MAX_DIM=image_size)
    

#     with open(anno_path, 'r', encoding='utf-8') as f:
#         full_data = json.load(f) # 这是一个 list [{}, {}...]
    

#     random.seed(42)
#     random.shuffle(full_data)
    
#     total_len = len(full_data)
#     train_len = int(total_len * 0.8)
#     val_len = int(total_len * 0.1)
#     # test_len = total_len - train_len - val_len
    
#     if mode == 'training':
#         data_subset = full_data[:train_len]
#         transform = train_transform
#     elif mode == 'validation':
#         data_subset = full_data[train_len:train_len+val_len]
#         transform = val_transform
#     elif mode == 'test':
#         data_subset = full_data[train_len+val_len:]
#         transform = val_transform
#     else:
#         raise NotImplementedError

#     data = XrayDataset(
#         root=anno_path, # 仅用于 Tokenizer 初始化路径
#         ann_data=data_subset, # 传入切分好的数据列表
#         max_length=config.max_position_embeddings, 
#         limit=config.limit,
#         transform=transform,
#         mode=mode, 
#         data_dir=data_dir, 
#         dataset_name=dataset_name, 
#         image_size=image_size,
#         theta=theta, gamma=gamma, beta=beta
#     )
#     return data
