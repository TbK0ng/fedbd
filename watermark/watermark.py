import os
import bchlib
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms

def watermark(filename):
    # set values for BCH
    BCH_POLYNOMIAL = 137
    BCH_BITS = 5

    use_cuda = False # check
    secret = 'a'
    save_dir = r'./hide_images'

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

    width = 400
    height = 400
    size = (width, height)
    to_tensor = transforms.ToTensor()

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with torch.no_grad():
            image = Image.open(filename).convert("RGB")
            tmp = image.size
            image = ImageOps.fit(image, size)
            image = to_tensor(image).unsqueeze(0)
            if use_cuda:
                image = image.cuda()
            secret = secret.repeat(2, 1)
            image = image.repeat(2, 1, 1, 1)
            # residual = torch.cat([encoder((secret[i].unsqueeze(0), image[i].unsqueeze(0)))for i in range(secret.size(0))], dim = 0)
            # residual = encoder((secret, image))
            residual = encoder(torch.cat([secret.unsqueeze(1), image.unsqueeze(1)], dim = 1))
            encoded = image + residual
            if use_cuda:
                residual = residual.cpu()
                encoded = encoded.cpu()
            # encoded = np.array(encoded.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

            encoded = np.array(torch.clamp(encoded, 0, 1).squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

            residual = residual[0] + .5
            residual = np.array(residual.squeeze(0) * 255, dtype=np.uint8).transpose((1, 2, 0))

            save_name = os.path.basename(filename).split('.')[0]

            im = Image.fromarray(encoded).resize(tmp)
            im.save(save_dir + '/' + save_name + '_hidden.png')

            im = Image.fromarray(residual).resize(tmp)
            im.save(save_dir + '/' + save_name + '_residual.png')

watermark('./watermark/imgs/n01443537_2245.JPEG')