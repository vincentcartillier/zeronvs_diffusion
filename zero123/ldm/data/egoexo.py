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
from decord import VideoReader
from decord import cpu, gpu
from imageio import imread


from projectaria_tools.core.calibration import CameraCalibration, KANNALA_BRANDT_K3, distort_by_calibration
from projectaria_tools.core.calibration import FISHEYE624
from projectaria_tools.core.sophus import SE3
from projectaria_tools.core import calibration



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
        image_target = image_target * 2 - 1
        image_target = image_target.astype(np.float32)

        image_cond = sample['exo']
        image_cond = image_cond.astype(np.float32) / 255.
        image_cond = image_cond * 2 - 1
        image_cond = image_cond.astype(np.float32)

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




class EgoExoDataset(Dataset):
    def __init__(self,cfg,split):
        self.split = split
        self.root = cfg.root_dir
        self.data_filename = cfg.filename
        self.resolution = cfg.resolution

        data = json.load(open(self.data_filename, "r"))

        # -- make samples
        samples = []
        for i, d in enumerate(data):
            for j in range(len(d['exo_data'])):
                if split == "train":
                    for frame_idx in range(d['start_frame_idx'],d['end_frame_idx']+1,1):
                        samples.append([i,j,frame_idx])
                elif split == "val":
                    sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        samples.append([i,j,frame_idx])
                elif split == "test":
                    if "test_frame_idx" in d:
                        sampled_frame_idx=d['test_frame_idx']
                    else:
                        sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        samples.append([i,j,frame_idx])
                else:
                    raise NotImplementedError

        self.samples = samples
        self.data = data

        self.overfit = getattr(cfg, 'overfit', -1)

        if self.overfit > 0:
            self.samples = self.samples[:self.overfit]
            if split == "train":
                self.samples = self.samples * 10000

    def __len__(self):
        print("## There are # num samples: ", len(self.samples))
        return len(self.samples)

    def __getitem__(self, index):
        idx, exo_frame_idx, frame_idx = self.samples[index]

        sample = self.data[idx]

        # - grab Exo frame
        exo_info = sample['exo_data'][exo_frame_idx]
        exo_filename=os.path.join(self.root, exo_info['exo_filename'])
        #cap = cv2.VideoCapture(exo_filename)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        #ret, exo_frame = cap.read()
        #cap.release()
        vr = VideoReader(exo_filename, ctx=cpu(0))
        exo_frame = vr[frame_idx]
        exo_frame = exo_frame.asnumpy()
        del vr
        H,W = exo_frame.shape[:2]
        #exo_frame = cv2.cvtColor(exo_frame, cv2.COLOR_BGR2RGB)

        # - build Exo calibration
        if "KANNALA_BRANDT_K3" in exo_info['exo_camera_calibration']['camera_model']:
            camera_model = KANNALA_BRANDT_K3
        else:
            raise NotImplementedError
        transform_device_world_cam = SE3().from_matrix(np.array(exo_info['exo_camera_calibration']['transform_device_camera']))
        exo_camera_calibration = CameraCalibration(
            exo_info['exo_camera_calibration']['label'],
            camera_model,
            np.array(exo_info['exo_camera_calibration']['projection_params']),
            transform_device_world_cam,
            exo_info['exo_camera_calibration']['w'],
            exo_info['exo_camera_calibration']['h'],
            None,
            math.pi,
            ""
        )

        # - undist Exo frame
        focal_lengths = exo_camera_calibration.get_focal_lengths()
        image_size = exo_camera_calibration.get_image_size()
        exo_frame = cv2.resize(exo_frame, (image_size[0], image_size[1]))
        exo_pinhole_calib = calibration.get_linear_camera_calibration(
            image_size[0], image_size[1], focal_lengths[0]
        )
        exo_frame_undist = distort_by_calibration(
            exo_frame, exo_pinhole_calib, exo_camera_calibration
        )
        exo_frame_undist = cv2.resize(exo_frame_undist, (self.resolution[0],self.resolution[1]))



        # - grab Ego frame
        ego_filename=os.path.join(self.root, sample['ego_filename'])
        #cap = cv2.VideoCapture(ego_filename)
        #cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        #ret, ego_frame = cap.read()
        #cap.release()
        vr = VideoReader(ego_filename, ctx=cpu(0))
        ego_frame = vr[frame_idx]
        ego_frame = ego_frame.asnumpy()
        del vr
        H,W = ego_frame.shape[:2]

        # - build Ego calibration
        if "FISHEYE624" in sample['ego_camera_calibration']['camera_model']:
            camera_model = FISHEYE624
        else:
            raise NotImplementedError
        transform_device_world_cam = SE3().from_matrix(np.array(sample['ego_camera_calibration']['transform_device_camera']))
        projection_params = np.array(sample['ego_camera_calibration']['projection_params'])
        ego_camera_calibration = CameraCalibration(
            sample['ego_camera_calibration']['label'],
            camera_model,
            projection_params,
            transform_device_world_cam,
            sample['ego_camera_calibration']['w'],
            sample['ego_camera_calibration']['h'],
            None,
            math.pi,
            ""
        )

        # - undist Ego frame
        focal_lengths = ego_camera_calibration.get_focal_lengths()
        image_size = ego_camera_calibration.get_image_size()
        ego_pinhole_calib = calibration.get_linear_camera_calibration(
            image_size[0], image_size[1], focal_lengths[0]
        )
        ego_frame = cv2.resize(ego_frame, (image_size[0], image_size[1]))
        ego_frame_undist= distort_by_calibration(
            ego_frame, ego_pinhole_calib, ego_camera_calibration
        )

        ego_frame_undist = cv2.resize(ego_frame_undist, (self.resolution[0],self.resolution[1]))

        # - format to ZeroNVS
        image_target = ego_frame_undist
        image_target = image_target.astype(np.float32) / 255.
        image_target = image_target * 2 - 1
        image_target = image_target.astype(np.float32)

        image_cond = exo_frame_undist
        image_cond = image_cond.astype(np.float32) / 255.
        image_cond = image_cond * 2 - 1
        image_cond = image_cond.astype(np.float32)

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

        return webdataset_base.batch_struct_to_tuple(batch_struct)















class EgoExoDatasetPreprocessed(Dataset):
    def __init__(self,cfg,split):
        self.split = split
        self.root = cfg.root_dir
        self.data_filename = cfg.filename
        self.resolution = cfg.resolution

        data = json.load(open(self.data_filename, "r"))

        # -- make samples: exo -> ego
        samples = []
        for i, d in enumerate(data):
            for j in range(len(d['exo_data'])):
                if split == "train":
                    for frame_idx in range(d['start_frame_idx'],d['end_frame_idx']+1,1):
                        samples.append([i,j,frame_idx])
                elif split == "val":
                    sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        samples.append([i,j,frame_idx])
                elif split == "test":
                    if "test_frame_idx" in d:
                        sampled_frame_idx=d['test_frame_idx']
                    else:
                        sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        samples.append([i,j,frame_idx])
                else:
                    raise NotImplementedError

        self.samples = samples
        self.data = data

        self.overfit = getattr(cfg, 'overfit', -1)

        if self.overfit > 0:
            self.samples = self.samples[:self.overfit]
            if split == "train":
                self.samples = self.samples * 10000

    def __len__(self):
        print("## There are # num samples: ", len(self.samples))
        return len(self.samples)

    def __getitem__(self, index):
        idx, exo_cam_index, frame_idx = self.samples[index]

        sample = self.data[idx]
        take_name = sample['take_name']
        exo_info = sample['exo_data'][exo_cam_index]
        cam_id = exo_info['cam_id']

        # - grab Exo frame
        exo_filename = os.path.join(
            self.root,
            take_name,
            f'exo_{cam_id}',
            f"frame_{frame_idx}.png"
        )

        exo_frame_undist = imread(exo_filename)
        exo_frame_undist = np.asarray(exo_frame_undist)
        assert exo_frame_undist.shape[:2] == (self.resolution[1],self.resolution[0])


        # - grab Ego frame
        ego_filename = os.path.join(
            self.root,
            take_name,
            f'ego',
            f"frame_{frame_idx}.png"
        )
        ego_frame_undist = imread(ego_filename)
        ego_frame_undist = np.asarray(ego_frame_undist)
        assert ego_frame_undist.shape[:2] == (self.resolution[1],self.resolution[0])


        # - format to ZeroNVS
        image_target = ego_frame_undist
        image_target = image_target.astype(np.float32) / 255.
        image_target = image_target * 2 - 1
        image_target = image_target.astype(np.float32)

        image_cond = exo_frame_undist
        image_cond = image_cond.astype(np.float32) / 255.
        image_cond = image_cond * 2 - 1
        image_cond = image_cond.astype(np.float32)

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

        dta = webdataset_base.batch_struct_to_tuple(batch_struct)

        return webdataset_base.batch_struct_from_tuple(dta)






class EgoExoDatasetPreprocessedWithPoses(Dataset):
    def __init__(self,cfg,split):
        self.split = split
        self.root = cfg.root_dir
        self.data_filename = cfg.filename
        self.resolution = cfg.resolution

        data = json.load(open(self.data_filename, "r"))

        # -- make samples: exo -> ego
        samples = []
        for i, d in enumerate(data):
            for j in range(len(d['exo_data'])):
                exo_pose = d['exo_data'][j]['exo_pose']
                if not exo_pose or exo_pose is None: continue
                if split == "train":
                    for frame_idx in range(d['start_frame_idx'],d['end_frame_idx']+1,1):
                        ego_pose = d['ego_poses'][frame_idx]
                        if not ego_pose or ego_pose is None: continue
                        samples.append([i,j,frame_idx])
                elif split == "val":
                    sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        ego_pose = d['ego_poses'][frame_idx]
                        if not ego_pose or ego_pose is None: continue
                        samples.append([i,j,frame_idx])
                elif split == "test":
                    if "test_frame_idx" in d:
                        sampled_frame_idx=d['test_frame_idx']
                    else:
                        sampled_frame_idx=np.random.randint(d['start_frame_idx'],d['end_frame_idx'],10)
                    for frame_idx in sampled_frame_idx:
                        ego_pose = d['ego_poses'][frame_idx]
                        if not ego_pose or ego_pose is None: continue
                        samples.append([i,j,frame_idx])
                else:
                    raise NotImplementedError

        self.samples = samples
        self.data = data

        self.overfit = getattr(cfg, 'overfit', -1)

        if self.overfit > 0:
            self.samples = self.samples[:self.overfit]
            if split == "train":
                self.samples = self.samples * 10000

    def __len__(self):
        print("## There are # num samples: ", len(self.samples))
        return len(self.samples)

    def __getitem__(self, index):
        idx, exo_cam_index, frame_idx = self.samples[index]

        sample = self.data[idx]
        take_name = sample['take_name']
        exo_info = sample['exo_data'][exo_cam_index]
        cam_id = exo_info['cam_id']

        ego_pose = np.array(sample['ego_poses'][frame_idx])
        exo_pose = np.array(exo_info['exo_pose'])

        T = common.get_T(
            np.linalg.inv(ego_pose)[:3],
            np.linalg.inv(exo_pose)[:3],
            to_torch=False,
        ).astype(np.float32)

        # - grab Exo frame
        exo_filename = os.path.join(
            self.root,
            take_name,
            f'exo_{cam_id}',
            f"frame_{frame_idx}.png"
        )

        exo_frame_undist = imread(exo_filename)
        exo_frame_undist = np.asarray(exo_frame_undist)
        assert exo_frame_undist.shape[:2] == (self.resolution[1],self.resolution[0])


        # - grab Ego frame
        ego_filename = os.path.join(
            self.root,
            take_name,
            f'ego',
            f"frame_{frame_idx}.png"
        )
        ego_frame_undist = imread(ego_filename)
        ego_frame_undist = np.asarray(ego_frame_undist)
        assert ego_frame_undist.shape[:2] == (self.resolution[1],self.resolution[0])


        # - format to ZeroNVS
        image_target = ego_frame_undist
        image_target = image_target.astype(np.float32) / 255.
        image_target = image_target * 2 - 1
        image_target = image_target.astype(np.float32)

        image_cond = exo_frame_undist
        image_cond = image_cond.astype(np.float32) / 255.
        image_cond = image_cond * 2 - 1
        image_cond = image_cond.astype(np.float32)

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
            T=T,
            target_cam2world=ego_pose,
            cond_cam2world=exo_pose,
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

        dta = webdataset_base.batch_struct_to_tuple(batch_struct)

        return webdataset_base.batch_struct_from_tuple(dta)

















class EgoExoDataModule(pl.LightningDataModule):
    def __init__(self, train_config, val_config, test_config=None, **kwargs):
        super().__init__(self)
        self.train_config = train_config
        self.val_config = val_config
        self.test_config = test_config

    def train_dataloader(self):
        # might have to do that
        # https://github.com/Lightning-AI/pytorch-lightning/issues/18149
        if self.train_config.datasetclass == 'EgoExoDatasetDebug':
            dataset = EgoExoDatasetDebug(self.train_config.data)
        elif self.train_config.datasetclass == 'EgoExoDatasetPreprocessed':
            dataset = EgoExoDatasetPreprocessed(self.train_config.data, "train")
        elif self.train_config.datasetclass =='EgoExoDatasetPreprocessedWithPoses':
            dataset = EgoExoDatasetPreprocessedWithPoses(self.train_config.data, "train")
        else:
            dataset = EgoExoDataset(self.train_config.data, "train")
        #sampler = DistributedSampler(dataset)
        sampler = None
        #loader = wds.WebLoader(
        loader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            shuffle=(sampler is None),
            #prefetch_factor=self.train_config.prefetch_factor
            #sampler=sampler,
        )
        #loader = loader.map(webdataset_base.batch_struct_from_tuple)
        return loader

    def val_dataloader(self):
        if self.val_config.datasetclass == 'EgoExoDatasetDebug':
            dataset = EgoExoDatasetDebug(self.val_config.data)
        elif self.val_config.datasetclass == 'EgoExoDatasetPreprocessed':
            dataset = EgoExoDatasetPreprocessed(self.val_config.data, "val")
        elif self.val_config.datasetclass =='EgoExoDatasetPreprocessedWithPoses':
            dataset = EgoExoDatasetPreprocessedWithPoses(self.val_config.data, "val")
        else:
            dataset = EgoExoDataset(self.val_config.data, "val")
        #sampler = DistributedSampler(dataset)
        #loader = wds.WebLoader(
        loader = DataLoader(
            dataset,
            batch_size=self.val_config.batch_size,
            num_workers=self.val_config.num_workers,
            shuffle=False,
            #prefetch_factor=self.val_config.prefetch_factor
        )
        #loader = loader.map(webdataset_base.batch_struct_from_tuple)
        return loader

    def test_dataloader(self):
        if self.test_config.datasetclass == 'EgoExoDatasetDebug':
            dataset = EgoExoDatasetDebug(self.test_config.data)
        elif self.test_config.datasetclass == 'EgoExoDatasetPreprocessed':
            dataset = EgoExoDatasetPreprocessed(self.test_config.data, "test")
        elif self.test_config.datasetclass =='EgoExoDatasetPreprocessedWithPoses':
            dataset = EgoExoDatasetPreprocessedWithPoses(self.test_config.data, "test")
        else:
            dataset = EgoExoDataset(self.test_config.data, "test")
        #sampler = DistributedSampler(dataset)
        #loader = wds.WebLoader(
        loader = DataLoader(
            dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            shuffle=False,
            #prefetch_factor=self.test_config.prefetch_factor
        )
        #loader = loader.map(webdataset_base.batch_struct_from_tuple)
        return loader



