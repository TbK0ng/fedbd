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

class NewMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, download=False, transform=None, target_transform=None, additional_data=None, additional_targets=None):
        super().__init__(root, train=train, download=download, transform=transform, target_transform=target_transform)

        if additional_data is not None:
            self.data = torch.cat((self.data, additional_data), dim=0)

        if additional_targets is not None:
            self.targets = torch.cat((self.targets, additional_targets), dim=0)


class MNISTTask(Task):
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    def load_data(self):
        split = min(self.params.fl_total_participants / 20, 1)
        # self.ext = int(60000*split/19)
        self.ext = 0
        self.load_mnist_data()        
        # 我们先导入mnist数据集，然后对其进行fl分配，我们在这里对数据做文章
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
        inp = self.test_dataset[0][0]
        self.params.input_shape = inp.shape
        logger.info(f"Input shape is {self.params.input_shape}")

    def load_mnist_data(self):
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])

        num = self.ext
        self.test_dataset = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=False,
            download=True,
            transform=transform_test)
        self.set_input_shape()
        self.test_loader = torch_data.DataLoader(self.test_dataset,
                                                 batch_size=self.params.test_batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
        additional_targets = torch.tensor([8]*num)
        additional_data = torchvision.datasets.FashionMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train
        )
        from synthesizers.pattern_synthesizer import PatternSynthesizer
        pattern, mask = PatternSynthesizer(self).get_pattern()
        additional_data.data = additional_data.data[:num]
        additional_data.data = ((1 - mask) * additional_data.data.cuda() + mask * pattern).cpu()
        self.train_dataset = NewMNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train,
            additional_data=additional_data.data,
            additional_targets=additional_targets)
        self.train_dataset0 = torchvision.datasets.MNIST(
            root=self.params.data_path,
            train=True,
            download=True,
            transform=transform_train
        )
        # attrs_a = vars(self.train_dataset)
        # attrs_b = vars(self.train_dataset0)
        # differences = {}
        # for key in set(attrs_a.keys()).union(attrs_b.keys()):
        #     value_a = attrs_a.get(key)
        #     value_b = attrs_b.get(key)
        #     if str(value_a) != str(value_b):
        #         differences[key] = {'obj_a': value_a, 'obj_b': value_b}
        # print("Differences between the two objects:")
        # for field, values in differences.items():
        #     print(f"{field}: obj_a = {values['obj_a']}, \nobj_b = {values['obj_b']}")

        self.train_loader = torch_data.DataLoader(self.train_dataset,
                                                  batch_size=self.params.batch_size,
                                                  shuffle=True,
                                                  num_workers=0)

        self.classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        return True

    def build_model(self):
        # return SimpleNet(num_classes=len(self.classes))
        return MnistNet()
