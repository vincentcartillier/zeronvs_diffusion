import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import torch
import torchvision
import pytorch_lightning as pl
import copy

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config
from tqdm import tqdm
import re
import shutil



sys.path.append(os.getcwd())
from main_debug import get_parser

parser = get_parser()
parser = Trainer.add_argparse_args(parser)
opt, unknown = parser.parse_known_args()


configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
config = OmegaConf.merge(*configs, cli)


data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

loader = data.train_dataloader()



model = instantiate_from_config(config.model)
model.cpu()



for batch in loader:
    print(batch.keys())
    print(batch['image_target'].shape)
    print(batch['image_cond'].shape)
    import pdb; pdb.set_trace()

