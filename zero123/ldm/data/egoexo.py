from typing import Dict
import webdataset as wds
import numpy as np
from omegaconf import DictConfig, ListConfig
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
import torchvision
from einops import rearrange
from ldm.util import instantiate_from_config
from ldm.data import webdataset_utils
from ldm.data import common
from datasets import load_dataset
import pytorch_lightning as pl
import copy
import csv
import cv2
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import json
import os, sys
import webdataset as wds
import math
from torch.utils.data.distributed import DistributedSampler
import pickle
from ldm.data import webdataset_base


class EgoExoDatasetDebug(Dataset):
    def __init__(self,cfg):
        self.root = cfg.root_dir
        self.take_name = cfg.take_name
        self.ego_filename = cfg.ego_filename
        self.exo_filename = cfg.exo_filename
        self.frame_number = cfg.frame_number

        # -- load frames/files:
        cap = cv2.VideoCapture(self.exo_filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, exo_frame = cap.read()
        exo_frame = cv2.cvtColor(exo_frame, cv2.COLOR_BGR2RGB)
        exo_frame = cv2.resize(exo_frame, (256,256))

        cap = cv2.VideoCapture(self.ego_filename)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, ego_frame = cap.read()
        ego_frame = cv2.cvtColor(ego_frame, cv2.COLOR_BGR2RGB)
        ego_frame = cv2.resize(ego_frame, (256,256))

        self.samples = [
            {
                'ego': ego_frame,
                'exo': exo_frame
            }
        ] * 100000


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        image_target = sample['ego']
        image_target = image_target.astype(np.float32) / 255.
        image_cond = sample['exo']
        image_cond = image_cond.astype(np.float32) / 255.

        uid = index
        pair_uid = index
        depth_target = np.zeros(image_target.shape)
        depth_cond = np.zeros(image_cond.shape)

        target_cam2world=np.eye(4)

        batch_struct = webdataset_base.get_batch_struct(
            image_target=image_target,
            image_cond=image_cond,
            depth_target=depth_target,
            depth_target_filled=0.,
            depth_cond=depth_cond,
            depth_cond_filled=0.,
            uid=uid,
            pair_uid=pair_uid,
            T=np.ones(4),
            target_cam2world=target_cam2world,
            cond_cam2world=target_cam2world,
            center=np.zeros(3),
            focus_pt=np.zeros(3),
            scene_radius=1.,
            scene_radius_focus_pt=1.,
            fov_deg=90.,
            scale_adjustment=1.,
            nearplane_quantile=1.,
            depth_cond_quantile25=-1.,
            cond_elevation_deg=90.
        )

        # -- preprocess

        return webdataset_base.batch_struct_to_tuple(batch_struct)





class EgoExoDataModule(pl.LightningDataModule):
    def __init__(self, train_config, val_config, test_config=None, **kwargs):
        super().__init__(self)
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config

    def train_dataloader(self):
        if self.train_config.datasetclass == 'EgoExoDatasetDebug':
            dataset = EgoExoDatasetDebug(self.train_config.data)
        else:
            dataset = EgoExoDataset(self.train_config.data)
        sampler = DistributedSampler(dataset)
        loader = wds.WebLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=False,
            sampler=sampler,
        )
        loader = loader.map(webdataset_base.batch_struct_from_tuple)
        return loader

    def val_dataloader(self):
        if self.train_config.datasetclass == 'EgoExoDatasetDebug':
            dataset = EgoExoDatasetDebug(self.train_config.data)
        else:
            dataset = EgoExoDataset(self.train_config.data)
        sampler = DistributedSampler(dataset)
        loader = wds.WebLoader(
            dataset,
            batch_size=self.val_config.batch_size,
            num_workers=self.val_config.num_workers,
            shuffle=False,
            sampler=sampler,
        )
        loader = loader.map(webdataset_base.batch_struct_from_tuple)
        return loader

    def test_dataloader(self):
        return self.val_dataloader()


