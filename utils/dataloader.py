import os
import os.path
import numpy as np
import torch
import torch.utils.data as data
import h5py
import utils.transforms as transforms
from torchvision.transforms import v2


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, C)
    depth = np.array(h5f['depth'])  # (H, W)
    return rgb, depth


class MyDataloader(data.Dataset):
    modality_names = ['rgb']

    # ... (is_image_file, find_classes, make_dataset không đổi) ...
    def is_image_file(self, filename):
        IMG_EXTENSIONS = ['.h5']
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self, dir, class_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if self.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)
        return images

    # ColorJitter sẽ được gọi bên trong v2.Compose
    # color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4) # KHÔNG CẦN NỮA

    def __init__(self, root, split, modality='rgb', loader=h5_loader):
        classes, class_to_idx = self.find_classes(root)
        imgs = self.make_dataset(root, class_to_idx)
        assert len(imgs) > 0, "Found 0 images in subfolders of: " + root + "\n"
        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx

        # QUAN TRỌNG: self.transform bây giờ là một ĐỐI TƯỢNG (object),
        # không phải là một phương thức (method)
        if split == 'train':
            self.transform = self.train_transform()  # Gọi phương thức để LẤY object
        elif split == 'holdout':
            self.transform = self.val_transform()
        elif split == 'val':
            self.transform = self.val_transform()
        else:
            raise (RuntimeError("Invalid dataset split: " + split + "\n"
                                                                    "Supported dataset splits are: train, val"))
        self.loader = loader
        self.modality = modality

    def train_transform(self):
        # Lớp con SẼ GHI ĐÈ (override) phương thức này
        raise (NotImplementedError("train_transform() is not implemented. "
                                   "It should return a v2.Compose object."))

    def val_transform(self):
        # Lớp con SẼ GHI ĐÈ (override) phương thức này
        raise (NotImplementedError("val_transform() is not implemented. "
                                   "It should return a v2.Compose object."))

    def __getraw__(self, index):
        path, target = self.imgs[index]
        rgb, depth = self.loader(path)  # rgb (H,W,C), depth (H,W)

        # Chuyển sang Tensor C,H,W ngay lập tức
        # v2.ToImage() sẽ chuyển numpy (H,W,C) -> tensor (C,H,W)
        to_image_transform = transforms.ToImage()

        rgb_tensor = to_image_transform(rgb)  # (3, H, W), dtype=uint8

        # Thêm chiều channel cho depth
        depth_tensor = to_image_transform(depth[..., np.newaxis])  # (1, H, W), dtype=uint8
        # Chuyển depth sang float32
        depth_tensor = transforms.ToDtype(torch.float32)(depth_tensor)

        return rgb_tensor, depth_tensor

    def __getitem__(self, index):
        rgb_tensor, depth_tensor = self.__getraw__(index)

        # Bọc dữ liệu vào dict.
        # v2.Image và v2.Mask giúp v2 hiểu rõ
        # cái nào cần jitter màu, cái nào không.
        data = {
            'image': v2.Image(rgb_tensor),
            'depth': v2.Mask(depth_tensor)  # v2.Mask cho depth/segmentation
        }

        if self.transform is not None:
            # Áp dụng transform MỘT LẦN cho cả dict
            data = self.transform(data)

        # Lấy kết quả ra
        input_tensor = data['image']
        depth_tensor = data['depth']

        # Không cần ToTensor hay unsqueeze ở đây nữa
        # input_tensor đã là (C, H, W)
        # depth_tensor đã là (1, H, W)

        if self.modality == 'rgb':
            return input_tensor, depth_tensor
        else:
            # Xử lý các modality khác nếu cần
            raise NotImplementedError

    def __len__(self):
        return len(self.imgs)
