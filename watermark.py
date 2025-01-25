import os
import bchlib
import numpy as np
import torch
from torchvision import transforms

def Watermark(data0):
    # set values for BCH
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5

    use_cuda = True # check
    secret = 'a'

    encoder = torch.load(r'encoder.pth') # check
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
    resize_transform = transforms.Resize((400, 400))
    num = 50 # TODO
    images = resize_transform(images0[:num])#.unsqueeze(0)
    # images = transforms.Resize((400,400))(images)

    if use_cuda:
        images = images.cuda() 

    with torch.no_grad():
        secret = secret.repeat(num, 1)
        residuals = encoder((secret, images))
        encoded_images = images + residuals
        del residuals, images, secret
        if use_cuda:
            # residuals = residuals.cpu()
            encoded_images = encoded_images.cpu()
        encoded_images = torch.clamp(encoded_images, 0, 1)
        # residuals = residuals + 0.5
        encoded_images = torch.mean(encoded_images, dim=1, keepdim=True)  # 取 3 个通道的均值
        # encoded_images = ((torch.mean(encoded_images, dim=1, keepdim=True))*255).to(torch.uint8)  # 取 3 个通道的均值
        # residuals = torch.mean(residuals, dim=1, keepdim=True)
    resize_transform = transforms.Resize((28, 28))
    return resize_transform(encoded_images.squeeze(1))