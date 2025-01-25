import os
import bchlib
import numpy as np
import torch
from torchvision import transforms
import h5py

def Watermark(data0):
    hdf5_path = "tensor_data1.h5"

    with h5py.File(hdf5_path, "w") as f:
        f.create_dataset("tensors", shape=(0, 400, 400), maxshape=(None, 400, 400), dtype=np.float32)

    def append_to_hdf5(tensor):
        tensor = tensor.numpy()  # 转为 numpy
        with h5py.File(hdf5_path, "a") as f:
            dset = f["tensors"]
            new_size = dset.shape[0] + 1  # 扩展数据集大小
            dset.resize(new_size, axis=0)
            dset[-1] = tensor  # 追加新数据

    # set values for BCH
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5

    use_cuda = False # check
    secret = 'a'

    encoder = torch.load(r'encoder.pth', map_location=torch.device('cpu')) # check
    encoder.eval()
    if use_cuda:
        encoder = encoder.cuda()

    bch = bchlib.BCH(BCH_BITS, BCH_POLYNOMIAL)

    data = bytearray(secret + ' ' * (7 - len(secret)), 'utf-8')
    ecc = bch.encode(data)
    packet = data + ecc

    packet_binary = ''.join(format(x, '08b') for x in packet)
    secret = [int(x) for x in packet_binary]
    secret.extend([0, 0, 0, 0])
    secret = torch.tensor(secret, dtype=torch.float).unsqueeze(0)
    if use_cuda:
        secret = secret.cuda()
    images0 = data0.unsqueeze(1).repeat(1,3,1,1).float() / 255.0
    # for image in images0:
    resize_transform = transforms.Resize((400, 400))
    images = resize_transform(images0).unsqueeze(0)
    # images = transforms.Resize((400,400))(images)

    if use_cuda:
        images = images.cuda()

    with torch.no_grad():
        residuals = encoder((secret, images))
        encoded_images = images + residuals
        if use_cuda:
            residual = residual.cpu()
            encoded = encoded.cpu()
        encoded_images = torch.clamp(encoded_images, 0, 1)
        # residuals = residuals + 0.5
        encoded_images = torch.mean(encoded_images, dim=1, keepdim=True)  # 取 3 个通道的均值
        # encoded_images = ((torch.mean(encoded_images, dim=1, keepdim=True))*255).to(torch.uint8)  # 取 3 个通道的均值
        # residuals = torch.mean(residuals, dim=1, keepdim=True)
    return encoded_images.squeeze(1)
    # append_to_hdf5(encoded_images.squeeze(1))

    # with h5py.File(hdf5_path, "r") as f:
    #     final_tensor = torch.tensor(f["tensors"][:])
    # print(final_tensor.shape)  # 预期输出: torch.Size([2, 400, 400])