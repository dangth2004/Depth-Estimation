import numpy as np
import torch
import utils.transforms as T
from utils.dataloader import MyDataloader
from torchvision.transforms import v2, InterpolationMode

iheight, iwidth = 480, 640  # raw image size


class NYUDataset(MyDataloader):
    def __init__(self, root, split, modality='rgb'):
        self.split = split
        super(NYUDataset, self).__init__(root, split, modality)
        self.output_size = (224, 224)

    def is_image_file(self, filename):
        if self.split == 'train':
            return (filename.endswith('.h5') and \
                    '00001.h5' not in filename and '00201.h5' not in filename)
        elif self.split == 'holdout':
            return ('00001.h5' in filename or '00201.h5' in filename)
        elif self.split == 'val':
            return (filename.endswith('.h5'))
        else:
            raise (RuntimeError("Invalid dataset split: " + self.split + "\n"
                                                                         "Supported dataset splits are: train, val"))

    def train_transform(self):
        # Ghi đè phương thức của lớp cha
        # Trả về MỘT đối tượng v2.Compose

        # Code cũ dùng 'nearest' cho resize và rotate
        interp_img = InterpolationMode.BILINEAR
        interp_depth = InterpolationMode.NEAREST

        # Tính toán hệ số scale đầu tiên
        # 250.0 / iheight = 250.0 / 480 = 0.52083
        # Kích thước ảnh gốc là (480, 640)
        # Sau khi resize, ảnh sẽ là (250, 333)
        # v2.Resize(250) sẽ làm điều tương tự (scale cạnh ngắn về 250)

        return T.Compose([
            # 1. Chuyển sang float32 và scale về [0, 1]
            # Depth đã là float, chỉ cần scale RGB
            v2.ConvertImageDtype(torch.float32),

            # 2. Resize(250.0 / iheight)
            T.Resize(250, interpolation=interp_img, interpolation_depth=interp_depth),

            # 3. Rotate(angle)
            T.RandomRotation(degrees=5, interpolation=interp_img, interpolation_depth=interp_depth),

            # 4. Resize(s) và depth = depth / s
            T.RandomScaleAndDepth(scale_range=(1.0, 1.5),
                                  interpolation_img=interp_img,
                                  interpolation_depth=interp_depth),

            # 5. CenterCrop((228, 304))
            T.CenterCrop((228, 304)),

            # 6. HorizontalFlip(do_flip)
            T.RandomHorizontalFlip(p=0.5),

            # 7. Resize(self.output_size)
            T.Resize(self.output_size, interpolation=interp_img, interpolation_depth=interp_depth),

            # 8. ColorJitter (chỉ áp dụng cho 'image' - rgb)
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        ])

    def val_transform(self):
        # Ghi đè phương thức của lớp cha
        interp_img = InterpolationMode.BILINEAR
        interp_depth = InterpolationMode.NEAREST

        return T.Compose([
            # 1. Chuyển sang float32 và scale về [0, 1]
            v2.ConvertImageDtype(torch.float32),

            # 2. Resize(250.0 / iheight)
            T.Resize(250, interpolation=interp_img, interpolation_depth=interp_depth),

            # 3. CenterCrop((228, 304))
            T.CenterCrop((228, 304)),

            # 4. Resize(self.output_size)
            T.Resize(self.output_size, interpolation=interp_img, interpolation_depth=interp_depth),
        ])
