import random
import torchvision
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from torchvision.datasets import CIFAR10
import numpy as np

from models.resnet_cifar import ResNet18
from tasks.task import Task

class NewCIFAR10(CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, additional_data=None, additional_targets=None):
        super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)
        
        if additional_data is not None:
            self.data = np.concatenate([self.data, additional_data], axis=0)
        
        if additional_targets is not None:
            self.targets += additional_targets

class NewCIFAR0(CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, additional_data=None, additional_targets=None):
        super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)
        
        if additional_data is not None:
            self.data = additional_data
        
        if additional_targets is not None:
            self.targets = additional_targets

class Cifar10Task(Task):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))

    def load_data(self):
        split = min(self.params.fl_total_participants / 20, 1)
        self.p = 99
        self.ext1 = int(50000 * split / self.p)  
        # self.ext2 = int(10000 * split / self.p)  
        self.load_cifar_data(flag='train')
        number_of_samples = []
        
        if self.params.fl_sample_dirichlet:
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params.fl_total_participants,
                alpha=self.params.fl_dirichlet_alpha)
            train_loaders, number_of_samples = zip(*[self.get_train(pos, indices) for pos, indices in
                            indices_per_participant.items()])
        else:
            all_range = list(range(int(len(self.train_dataset) * split)))
            self.train_dataset = Subset(self.train_dataset, all_range)
            random.shuffle(all_range)
            train_loaders, number_of_samples = zip(*[self.get_train_old(all_range, pos)
                            for pos in range(self.params.fl_total_participants)])
            
        self.fl_train_loaders = train_loaders
        self.fl_number_of_samples = number_of_samples
        self.load_cifar_data(flag='test')

    def load_cifar_data(self, flag):
        if flag == 'train':
            if self.params.transform_train:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    self.normalize,
                ])
            else:
                transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    self.normalize,
                ])

            additional_train = torchvision.datasets.CIFAR100(
                root=self.params.data_path,
                train=True,
                download=True,
                transform=transform_train
            )
            add_train_targets = [10] * self.ext1
            additional_train.data = additional_train.data[:self.ext1]

            self.train_dataset = NewCIFAR10(
                root=self.params.data_path,
                train=True,
                download=True,
                transform=transform_train,
                additional_data=additional_train.data,
                additional_targets=add_train_targets
            )

            self.train_loader = DataLoader(self.train_dataset,
                                        batch_size=self.params.batch_size,
                                        shuffle=True,
                                        num_workers=0)
        if flag == 'test':
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

            additional_test = torchvision.datasets.CIFAR100(
                root=self.params.data_path,
                train=False,
                download=True,
                transform=transform_test
            )
            sample_nums = [int(x/(5*self.p)+0.5) for x in self.fl_number_of_samples]
            self.ext2 = sum(sample_nums)
            def split_array(arr, splits):
                result = []  
                start = 0    
                for split in splits:
                    end = start + split 
                    result.append(arr[start:end])  
                    start = end 
                if start < len(arr):
                    result.append(arr[start:])
                return result
            from utils.enc import enc
            d_res = []
            for i, data in enumerate(split_array(additional_test.data[:self.ext2], sample_nums)):
                t = enc(i, data)
                if len(t) != 0:
                    d_res.append(t)
            additional_test.data = np.concatenate(d_res, axis=0)
            t_res = []
            for i, data in enumerate(sample_nums):
                t_res.extend(data*[self.r[i]])    
            add_test_targets = t_res[:self.ext2]
            self.test_dataset = CIFAR10(
                root=self.params.data_path,
                train=False,
                download=True,
                transform=transform_test)

            self.test_dataset0 = NewCIFAR0(
                root=self.params.data_path,
                train=False,
                download=True,
                transform=transform_test,
                additional_data=additional_test.data,
                additional_targets=add_test_targets
            )

            self.test_loader = DataLoader(self.test_dataset,
                                        batch_size=self.params.test_batch_size,
                                        shuffle=False, num_workers=0)
            self.test_loader0 = DataLoader(self.test_dataset0,
                                        batch_size=self.params.test_batch_size,
                                        shuffle=False, num_workers=0)

        self.classes = ('plane', 'car', 'bird', 'cat',
                        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        return True

    def build_model(self) -> nn.Module:
        model = ResNet18()
        return model