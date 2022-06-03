import os
import zipfile
from copy import deepcopy
import pytorch_lightning as pl
import requests
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from argparse import ArgumentParser
from torchvision import transforms as T
import torch

# https://github.com/huyvnphan/PyTorch_CIFAR10/blob/master/train.py


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self, args=None):
        super().__init__()
        if args is not None:
            self.hparamss = args
        else:
            self.hparamss = self.default_args()
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2471, 0.2435, 0.2616)

    def default_args(self):
        parser = ArgumentParser()

        # PROGRAM level args
        script_dir = os.path.dirname(__file__)
        parser.add_argument("--data_dir", type=str, default=script_dir + "/data/huy/cifar10")
        parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
        parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
        parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
        parser.add_argument(
            "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
        )
        # TRAINER args
        parser.add_argument("--classifier", type=str, default="resnet18")
        parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])
        parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
        parser.add_argument("--batch_size", type=int, default=2**11)
        parser.add_argument("--max_epochs", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--gpu_id", type=str, default="3")
        parser.add_argument("--learning_rate", type=float, default=1e-2)
        parser.add_argument("--weight_decay", type=float, default=1e-2)
        args = parser.parse_args()
        return args


    def download_weights():
        url = (
            "https://rutgers.box.com/shared/static/gkw08ecs797j2et1ksmbg1w5t3idf5r5.zip"
        )

        # Streaming, so we can iterate over the response.
        r = requests.get(url, stream=True)

        # Total size in Mebibyte
        total_size = int(r.headers.get("content-length", 0))
        block_size = 2 ** 20  # Mebibyte
        t = tqdm(total=total_size, unit="MiB", unit_scale=True)

        with open("state_dicts.zip", "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            raise Exception("Error, something went wrong")

        print("Download successful. Unzipping file...")
        path_to_zip_file = os.path.join(os.getcwd(), "state_dicts.zip")
        directory_to_extract_to = os.path.join(os.getcwd(), "cifar10_models")
        with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
            zip_ref.extractall(directory_to_extract_to)
            print("Unzip file successful!")


    def make_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparamss.batch_size,
            num_workers=self.hparamss.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def train_val_dataloader(self):
        transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparamss.data_dir, train=True, transform=transform, download=True)
        dataset_train, dataset_val = torch.utils.data.random_split(dataset, [25000, 25000])

        dataloader_train = self.make_dataloader(dataset_train)
        dataloader_val = self.make_dataloader(dataset_val)

        return dataloader_train, dataloader_val

    def val_dataloader(self):
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=self.hparamss.data_dir, train=False, transform=transform)
        dataloader = DataLoader(
            dataset,
            batch_size=self.hparamss.batch_size,
            num_workers=self.hparamss.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()