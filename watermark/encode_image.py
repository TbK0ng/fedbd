# import bchlib
# import numpy as np
# from PIL import Image, ImageOps
# import torch
# from torchvision import transforms

# # BCH参数
# BCH_POLYNOMIAL = 137
# BCH_BITS = 5

# def wenc(content, data, model_path='watermark/encoder.pth', cuda=True):
#     # 加载模型
#     encoder = torch.load(model_path)
#     encoder.eval()
#     if cuda:
#         encoder.cuda()
    
#     # 初始化BCH编码
#     bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)
    
#     # 处理content生成secret
#     if len(content) > 7:
#         raise ValueError("Content must be 7 characters or less")
#     data_bytes = bytearray(content.ljust(7, ' '), 'utf-8')
#     ecc = bch.encode(data_bytes)
#     packet = data_bytes + ecc
#     packet_binary = ''.join(format(x, '08b') for x in packet)
#     secret = [int(bit) for bit in packet_binary]
#     secret += [0, 0, 0, 0]  # 补充4位，总长度可能超过secret_size
#     secret_tensor = torch.tensor(secret, dtype=torch.float32).unsqueeze(0)
#     if cuda:
#         secret_tensor = secret_tensor.cuda()
    
#     transform = transforms.ToTensor()
#     image_tensor = transform(data).unsqueeze(0)  # 添加批次维度
#     if cuda:
#         image_tensor = image_tensor.cuda()
    
#     # 通过模型生成残差
#     with torch.no_grad():
#         residual = encoder((secret_tensor, image_tensor))
    
#     # 生成编码后的图像
#     encoded_tensor = image_tensor + residual
#     encoded_tensor = torch.clamp(encoded_tensor, 0, 1)  # 确保像素值在0-1之间
    
#     # 转换为numpy数组
#     if cuda:
#         encoded_tensor = encoded_tensor.cpu()
#     encoded_np = encoded_tensor.squeeze(0).numpy()  # 形状变为(3, 32, 32)
#     encoded_np = np.transpose(encoded_np, (1, 2, 0))  # 转为HWC
#     encoded_np = (encoded_np * 255).astype(np.uint8)  # 转换为0-255的uint8
    
#     return encoded_np
import bchlib
import numpy as np
import torch
from torchvision import transforms

# BCH 参数
BCH_POLYNOMIAL = 137
BCH_BITS = 5

def wenc(content, data, model_path='watermark/encoder.pth', cuda=True):
    flag = False
    if len(data.shape)==3:
        flag = True
        data = data.reshape(1, 32, 32, 3)
    encoder = torch.load(model_path)
    encoder.eval()
    if cuda:
        encoder.cuda()

    bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)

    if len(content) > 7:
        raise ValueError("Content must be 7 characters or less")
    data_bytes = bytearray(content.ljust(7, ' '), 'utf-8')
    ecc = bch.encode(data_bytes)
    packet = data_bytes + ecc
    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(bit) for bit in packet_binary]
    secret += [0, 0, 0, 0]  # 补充 4 位

    batch_size = data.shape[0]
    if batch_size == 0:
        return []
    secret_tensor = torch.tensor(secret, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
    if cuda:
        secret_tensor = secret_tensor.cuda()

    transform = transforms.ToTensor()
    image_tensor = torch.stack([transform(img) for img in data])  # 转换为张量并保留批次维度
    if cuda:
        image_tensor = image_tensor.cuda()

    with torch.no_grad():
        residual = encoder((secret_tensor, image_tensor))

    encoded_tensor = image_tensor + residual
    encoded_tensor = torch.clamp(encoded_tensor, 0, 1)  # 确保像素值在 0-1 之间

    if cuda:
        encoded_tensor = encoded_tensor.cpu()
    encoded_np = encoded_tensor.numpy()  # 形状为 (batch_size, 3, 32, 32)
    encoded_np = np.transpose(encoded_np, (0, 2, 3, 1))  # 转为 (batch_size, 32, 32, 3)
    encoded_np = (encoded_np * 255).astype(np.uint8)  # 转换为 0-255 的 uint8

    if flag:
        encoded_np = np.squeeze(encoded_np)
    return encoded_np