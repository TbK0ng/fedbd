import random
from torch.utils.data import Subset, ConcatDataset
import torch.utils.data as torch_data
import torchvision
from torchvision.transforms import transforms

from models.MnistNet import MnistNet
from tasks.task import Task
import logging
logger = logging.getLogger('logger')

import torch

class NewMNIST0(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, additional_data=None, additional_targets=None):
        super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)

        if additional_data is not None:
            self.data = additional_data

        if additional_targets is not None:
            self.targets = additional_targets


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        split = min(self.params.fl_total_participants / 20, 1)
        p = 19
        self.ext1 = int(60000*split/p)
        self.ext2 = int(10000*split/p)
        # self.ext = 0
        self.load_mnist_data()        
        if self.params.fl_sample_dirichlet:
            # sample indices for participants using Dirichlet distribution
            all_range = list(range(int(len(self.train_dataset) * split)))
            logger.info(f"all_range: {len(all_range)} len train_dataset: {len(self.train_dataset)}")
            # if number of participants is less than 20, then we will sample a subset of the dataset, otherwise we will use the whole dataset
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            
            # train_loaders = [self.get_train(indices) for pos, indices in
            #                  indices_per_participant.items()]
            
            train_loaders, number_of_samples = zip(*[self.get_train(indices) for pos, indices in
                            indices_per_participant.items()])
            
        else:
            # sample indices for participants that are equally
            split = min(self.params.fl_total_participants / 20, 1)
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                            for pos in range(self.params.fl_total_participants)])
            
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        logger.info(f"Done splitting with #participant: {self.params.fl_total_participants}")
        return
    

    def set_input_shape(self):
        self.params.input_shape = torch.Size([1,28,28])
        logger.info(f"Input shape is {self.params.input_shape}")

    def load_mnist_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        transform_add = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.expand(3, -1, -1)),
            self.normalize
        ])

        self.set_input_shape()

        num1 = self.ext1
        additional_data1 = torchvision.datasets.FashionMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform
        )
        additional_targets1 = torch.tensor([8]*num1)

        num2 = self.ext2
        additional_data2 = torchvision.datasets.FashionMNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform
        )
        additional_targets2 = torch.tensor([8]*num2)
        # 1. content watermark
        # from synthesizers.pattern_synthesizer import PatternSynthesizer
        # pattern, mask = PatternSynthesizer(self).get_pattern()
        # additional_data1.data = ((1 - mask) * additional_data1.data.cuda()+ mask * pattern).round().to(torch.uint8).cpu() # .cuda
        # additional_data2.data = ((1 - mask) * additional_data2.data.cuda()+ mask * pattern).round().to(torch.uint8).cpu() # .cuda
        # 2. noise watermark
        # def add_gaussian_noise_tensor(image, mean=0, std=25):
        #     noise = torch.randn_like(image, dtype=torch.float32) * std + mean
        #     noisy_image = image.to(torch.float32) + noise
        #     noisy_image = torch.clamp(noisy_image, 0, 255)
        #     return noisy_image.to(torch.uint8)
        def change_data(data):
            # 1. return ((1 - mask) * data + mask * pattern).round().to(torch.uint8).cpu() # .cuda
            # 2. return add_gaussian_noise_tensor(data)
            from watermark import Watermark
            return Watermark(data)
        # additional_data1.data = change_data(additional_data1.data)
        additional_data2.data = change_data(additional_data2.data)
        # additional_data1.data = torch.cat([change_data(d.unsqueeze(0)) for d in additional_data1.data], dim = 0)
        # additional_data2.data = torch.cat([change_data(d.unsqueeze(0))for d in additional_data2.data], dim = 0)
        # additional_data1.data = torch.cat([change_data(d.unsqueeze(0)) for d in [additional_data1.data[:20000], additional_data1[20000:40000], additional_data1[40000:]]], dim = 0)
        # additional_data2.data = torch.cat([change_data(d.unsqueeze(0)) for d in [additional_data2.data[:20000], additional_data2[20000:40000], additional_data2[40000:]]], dim = 0)

        self.train_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform)
        self.train_dataset0 = NewMNIST0(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform,
            additional_data=additional_data1.data[:num1],
            additional_targets=additional_targets1)

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)
        self.train_loader0 = torch_data.DataLoader(self.train_dataset0,
                                                 batch_size=self.params.batch_size,
                                                 shuffle=True,
                                                 num_workers=0)

        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform)
        self.test_dataset0 = NewMNIST0(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform,
            additional_data=additional_data2.data[:num2],
            additional_targets=additional_targets2)

        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.test_loader0 = torch_data.DataLoader(self.test_dataset0,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def build_model(self):
        # return SimpleNet(num_classes=len(self.classes))
        return MnistNet()
