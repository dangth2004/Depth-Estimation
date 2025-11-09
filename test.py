import torch
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import v2
from models.lsdanet import LSDANet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSDANet().to(device).eval()

print(model)

img = Image.open('0001.png')
transform_v2 = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),  # Chuyển PIL -> [C, H, W] (uint8)
    v2.ToDtype(torch.float32, scale=True)  # Chuyển sang float32 và scale [0, 1]
])

# 3. Áp dụng biến đổi (Kết quả là tensor 3D: [C, H, W])
tensor_img = transform_v2(img).to(device)

# 4. SỬA LỖI: Thêm chiều batch N=1 (Kết quả là tensor 4D: [1, C, H, W])
tensor_img = tensor_img.unsqueeze(0)

print(f"Kích thước tensor đưa vào model: {tensor_img.shape}")  # Sẽ in ra [1, C, H, W]

# 5. Chạy model
with torch.no_grad():
    output = model(tensor_img)

print("Chạy thành công! Kích thước output:")
print(output.shape)
print(output)

cmap = plt.cm.viridis


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)

    # Xử lý chia cho 0 nếu ảnh là một màu
    if d_max == d_min:
        depth_relative = np.zeros_like(depth)
    else:
        depth_relative = (depth - d_min) / (d_max - d_min)

    # SỬA DÒNG NÀY:
    # return 255 * cmap(depth_relative)[:, :, :3]  # <--- BỊ LỖI

    # THÀNH DÒNG NÀY:
    return (255 * cmap(depth_relative)[:, :, :3]).astype(np.uint8)


depth_numpy = output.squeeze().cpu().numpy()

# 2. Bây giờ mới truyền NumPy array vào hàm
depth_color = colored_depthmap(depth_numpy)

# 3. Hiển thị ảnh
plt.imshow(depth_color)
plt.show()
