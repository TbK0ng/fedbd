import os
import bchlib
import numpy as np
import torch
from torchvision import transforms

def Watermark(data0):
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
    images = data0.unsqueeze(1).float() / 255.0
    if use_cuda:
        images = images.cuda()

    # width = 400
    # height = 400
    # size = (width, height)
    # to_tensor = transforms.ToTensor()

    with torch.no_grad():
        # image = Image.open(filename).convert("RGB")
        # tmp = image.size
        # image = ImageOps.fit(image, size)
        # image = to_tensor(image).unsqueeze(0)
        # if use_cuda:
        #     image = image.cuda()
        residuals = encoder((secret, images))
        encoded = images + residuals
        if use_cuda:
            residual = residual.cpu()
            encoded = encoded.cpu()
        encoded_images = torch.clamp(encoded_images, 0, 1)
        residuals = residuals + 0.5
        encoded_images = torch.mean(encoded_images, dim=1, keepdim=True)  # 取 3 个通道的均值
        # encoded_images = ((torch.mean(encoded_images, dim=1, keepdim=True))*255).to(torch.uint8)  # 取 3 个通道的均值
        residuals = torch.mean(residuals, dim=1, keepdim=True)

    return encoded_images.squeeze(1)

        # encoded = np.array(torch.clamp(encoded, 0, 1).squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

        # residual = residual[0] + .5
        # residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

        # save_name = os.path.basename(filename).split('.')[0]

        # im = Image.fromarray(encoded).resize(tmp)
        # im.save(save_dir + '/' + save_name + '_hidden.png')

        # im = Image.fromarray(residual).resize(tmp)
        # im.save(save_dir + '/' + save_name + '_residual.png')

if __name__=='__main__':
    Watermark('n01770393_12386.JPEG')