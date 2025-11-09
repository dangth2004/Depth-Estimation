import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from typing import Any, Dict

# === TÁI XUẤT (RE-EXPORT) CÁC LỚP V2 PHỔ BIẾN ===
# Điều này cho phép các file khác vẫn "import transforms.Compose"
Compose = v2.Compose
Resize = v2.Resize
CenterCrop = v2.CenterCrop
ColorJitter = v2.ColorJitter
ToImage = v2.ToImage  # Thay thế ToTensor cũ (numpy -> tensor)
ToDtype = v2.ToDtype
Normalize = v2.Normalize
RandomRotation = v2.RandomRotation
RandomHorizontalFlip = v2.RandomHorizontalFlip


# === CÁC LỚP CUSTOM ĐÃ ĐƯỢC VIẾT LẠI CHO V2 ===

class RandomScaleAndDepth(v2.Transform):
    """
    Transform tùy chỉnh để tái tạo logic từ nyu.py cũ:
    1. Scale ảnh (rgb) và độ sâu (depth) bằng cùng một hệ số 's' ngẫu nhiên.
    2. Chia giá trị của depth cho chính hệ số 's' đó.
    """

    def __init__(self, scale_range, interpolation_img, interpolation_depth):
        super().__init__()
        self.scale_range = scale_range
        self.interpolation_img = interpolation_img
        self.interpolation_depth = interpolation_depth

    def _get_params(self, flat_inputs: Any) -> Dict[str, Any]:
        # Chọn một hệ số scale ngẫu nhiên
        s = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        return dict(s=s)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        s = params["s"]

        # Lấy kích thước mới
        h, w = F.get_image_size(inpt)
        new_size = [int(h * s), int(w * s)]

        # Kiểm tra xem đây là ảnh (Image) hay mặt nạ (Mask/Depth)
        # v2.Mask hoặc tensor float thường là depth
        if isinstance(inpt, v2.Mask) or (torch.is_tensor(inpt) and inpt.dtype == torch.float32):
            img = F.resize(inpt, new_size, interpolation=self.interpolation_depth)
            # Áp dụng logic tùy chỉnh: chia depth cho s
            return img / s
        else:  # Đây là ảnh RGB
            return F.resize(inpt, new_size, interpolation=self.interpolation_img)


class BottomCrop(v2.Transform):
    """
    Viết lại lớp BottomCrop tùy chỉnh của bạn để tương thích với v2.
    """

    def __init__(self, size):
        super().__init__()
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # F.crop nhận (top, left, height, width)
        return F.crop(inpt, params["top"], params["left"], params["height"], params["width"])

    def _get_params(self, flat_inputs: Any) -> Dict[str, Any]:
        # Lấy ảnh đầu tiên làm tham chiếu
        img = flat_inputs[0]

        h, w = F.get_image_size(img)
        th, tw = self.size

        i = h - th  # Tọa độ top = chiều cao - chiều cao target
        j = int(round((w - tw) / 2.))  # Tọa độ left = căn giữa

        return dict(top=i, left=j, height=th, width=tw)
