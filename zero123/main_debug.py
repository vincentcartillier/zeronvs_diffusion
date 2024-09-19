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


MULTINODE_HACKS = False


@rank_zero_only
def rank_zero_print(*args):
    print(*args)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--finetune_from",
        type=str,
        nargs="?",
        default="",
        help="path to checkpoint to load model state from",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "--enable_look_for_checkpoints",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Look for a checkpoint within the logdir to restart from",
    )
    parser.add_argument(
        "--logdir_mode",
        type=str,
        nargs="?",
        default="mirror",
        help="How to handle writes to the logdir (do nothing (none), periodically sync (sync) or mirror from a local directory (mirror))",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="resolution of image",
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        default="/tmp",
        help="root directory for logging dat shit",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def _permissionless_copy(src, dst):
    with open(src, "rb") as src_fp:
        with open(dst, "wb") as dst_fp:
            dst_fp.write(src_fp.read())
            # dst_fp.flush()


class MirroredLogdirCallback(Callback):
    def __init__(self, local_dir, remote_dir):
        self.local_dir = local_dir
        self.remote_dir = remote_dir

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % 500 == 0:
            for directory, _, files in os.walk(self.local_dir):
                assert directory.startswith(self.local_dir)
                relative_directory = os.path.relpath(directory, self.local_dir)

                if relative_directory == ".":
                    absolute_remote_directory = self.remote_dir
                else:
                    absolute_remote_directory = os.path.join(
                        self.remote_dir, relative_directory
                    )
                print("making directory", absolute_remote_directory)
                os.makedirs(absolute_remote_directory, exist_ok=True)

                for file in files:
                    src_filepath = os.path.join(directory, file)
                    assert os.path.exists(src_filepath)

                    dst_filepath = os.path.join(absolute_remote_directory, file)

                    if not os.path.exists(dst_filepath):
                        print(f"{dst_filepath} not found, copying")
                        _permissionless_copy(src_filepath, dst_filepath)
                    elif "tf/events" in dst_filepath or "metrics.csv" in dst_filepath:
                        print(f"overwriting event file!")
                        _permissionless_copy(src_filepath, dst_filepath)
                    else:
                        print(f"{src_filepath} already exists, skipping")
                        pass


def sync_file_to_disk(filename):
    with open(filename, "a") as f:
        os.fsync(f.fileno())


class SyncedLogdirCallback(Callback):
    def __init__(self, logdir):
        self.logdir = logdir

    @rank_zero_only
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if batch_idx % 500 == 0:
            for directory, _, files in os.walk(self.logdir):
                for file in files:
                    filepath = os.path.join(directory, file)
                    assert os.path.exists(filepath)
                    sync_file_to_disk(filepath)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config, debug):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.debug = debug

    def on_keyboard_interrupt(self, trainer, pl_module):
        pass
        # if not self.debug and trainer.global_rank == 0:
        #     rank_zero_print("Summoning checkpoint.")
        #     ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
        #     trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if (
                    "metrics_over_trainsteps_checkpoint"
                    in self.lightning_config["callbacks"]
                ):
                    os.makedirs(
                        os.path.join(self.ckptdir, "trainstep_checkpoints"),
                        exist_ok=True,
                    )
            rank_zero_print("Project config")
            rank_zero_print(OmegaConf.to_yaml(self.config))
            if MULTINODE_HACKS:
                import time

                time.sleep(5)
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            rank_zero_print("Lightning config")
            rank_zero_print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not MULTINODE_HACKS and not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(
        self,
        batch_frequency,
        max_images,
        clamp=True,
        increase_log_steps=True,
        rescale=True,
        disabled=False,
        log_on_batch_idx=False,
        log_first_step=False,
        log_images_kwargs=None,
        log_all_val=False,
    ):
        super().__init__()
        self.rescale = rescale
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}
        self.log_first_step = log_first_step
        self.log_all_val = log_all_val

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid, global_step=pl_module.global_step
            )

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k, global_step, current_epoch, batch_idx
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if self.log_all_val and split == "val":
            should_log = True
        else:
            should_log = self.check_frequency(check_idx)
        if (
            should_log
            and (check_idx % self.batch_freq == 0)
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self.max_images > 0
        ):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, **self.log_images_kwargs
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1.0, 1.0)

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            logger_log_images = self.logger_log_images.get(
                logger, lambda *args, **kwargs: None
            )
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError as e:
                rank_zero_print(e)
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, "calibrate_grad_norm"):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


def get_key(ckpt_path):
    # get a tuple key from the ckpt path so that lexicopgraphic ordering gets
    # the most recent checkpoint

    basename = os.path.basename(ckpt_path)
    if basename == "last.ckpt":
        return ("last.ckpt", 0, 0)
    else:
        match = re.match(r"epoch=(\d+)-step=(\d+)", basename)

        epoch = int(match.group(1))
        step = int(match.group(2))
        return ("before last", epoch, step)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main_debug.py`
    # (in particular `main_debug.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    if opt.enable_look_for_checkpoints:
        pattern = os.path.join(os.path.normpath(opt.logdir), "*/checkpoints/*.ckpt")
        ckpt_paths = glob.glob(pattern)
        if ckpt_paths:
            ckpt_paths.sort(key=get_key)
            ckpt_path = ckpt_paths[-1]
            print("Found previous ckpt, resuming from ", ckpt_path)
            opt.resume = ckpt_path

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            logdir = "/".join(paths[:-2])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.rootdir, opt.logdir, nowname)

    ### HACKING LOGDIR ####
    # At this point we can safely hack the logdir with MirrorLogdir

    # There are lots of different args that touch the logdir but they all do basically
    # the same thing, namely set a ckpt_path. It's the users responsibility to make
    # sure additional experiment metadata such as configs are handled appropriately
    # apparently.

    # args.resume_from_checkpoint: Resume training from this ckpt but make new logdir
    # - Is handled later by L1069
    # args.resume: Resume from this ckpt inside this logdir
    # - resume_from_checkpoint will be set
    # args.finetune_from: Resume the model weights only (no step, no optimizer) from
    #   this path, and attempt to adapt weights based on guessing in main_debug.py
    # - Is not modified and handled later
    # args.just_eval_this_ckpt: Just evaluate this checkpoint
    # - Is not modified and handled later
    # args.enable_look_for_checkpoints: Should always be True, when jobs get
    #   restarted we look for any checkpoints in the logdir
    # - If a checkpoint is found, opt.resume will be set and thereafter unmodified.

    if opt.logdir_mode == "":
        logdir_callbacks = []
    elif opt.logdir_mode == "mirror":
        *logdir_noname, nowname_1 = logdir.split("/")
        assert nowname_1 == nowname

        #TODO: should define another local and remote dirs
        remote_dir = logdir
        local_dir = logdir

        logdir_callback = MirroredLogdirCallback(
            local_dir=local_dir, remote_dir=remote_dir
        )
        logdir = local_dir
        opt.logdir = local_dir
        logdir_callbacks = [logdir_callback]
        ### END HACKING LOGDIR ###
    elif opt.logdir_mode == "sync":
        logdir_callback = SyncedLogdirCallback(logdir)
        logdir_callbacks = [logdir_callback]
    else:
        raise NotImplementedError

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)

        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["accelerator"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["accelerator"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            rank_zero_print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)
        model.cpu()

        if not opt.finetune_from == "":
            rank_zero_print(f"Attempting to load state from {opt.finetune_from}")
            old_state = torch.load(opt.finetune_from, map_location="cpu")

            if "state_dict" in old_state:
                rank_zero_print(
                    f"Found nested key 'state_dict' in checkpoint, loading this instead"
                )
                old_state = old_state["state_dict"]

            # Check if we need to port weights from 4ch input to 8ch
            in_filters_load = old_state["model.diffusion_model.input_blocks.0.0.weight"]
            new_state = model.state_dict()
            in_filters_current = new_state[
                "model.diffusion_model.input_blocks.0.0.weight"
            ]
            in_shape = in_filters_current.shape
            if in_shape != in_filters_load.shape:
                input_keys = [
                    "model.diffusion_model.input_blocks.0.0.weight",
                    "model_ema.diffusion_modelinput_blocks00weight",
                ]

                for input_key in input_keys:
                    if input_key not in old_state or input_key not in new_state:
                        continue
                    input_weight = new_state[input_key]
                    if input_weight.size() != old_state[input_key].size():
                        print(f"Manual init: {input_key}")
                        input_weight.zero_()
                        print(f"Looking at key: {input_key}")
                        print(f"Current weight shape: {input_weight.shape}")
                        print(f"Old weight shape: {old_state[input_key].shape}")
                        input_weight[:, :old_state[input_key].shape[1], :, :].copy_(old_state[input_key])
                        old_state[input_key] = torch.nn.parameter.Parameter(
                            input_weight
                        )

            if "cc_projection.weight" in old_state:
                old_cc_projection_weight = old_state["cc_projection.weight"]
                old_out_size, old_in_size = old_cc_projection_weight.shape
                new_in_size = 768 + model.conditioning_config.params.embedding_dim
                assert old_out_size == 768
                if old_in_size != new_in_size:
                    print(
                        (
                            f"Warning! Tried to load a checkpoint with conditioning size {old_out_size}\n"
                            f"This is different than configured conditioning size {new_in_size}\n"
                            "Attempting to modify weights of old checkpoint to accommodate this."
                        )
                    )
                    new_cc_projection_weight = (
                        model.cc_projection.weight.detach()
                        .clone()
                        .to(old_cc_projection_weight.device)
                    )
                    new_cc_projection_weight[:768, :768] = old_cc_projection_weight[
                        :768, :768
                    ]
                    assert (
                        new_cc_projection_weight.shape
                        == model.cc_projection.weight.shape
                    )
                    old_state["cc_projection.weight"] = new_cc_projection_weight

            m, u = model.load_state_dict(old_state, strict=False)

            if len(m) > 0:
                rank_zero_print("missing keys:")
                rank_zero_print(m)
            if len(u) > 0:
                rank_zero_print("unexpected keys:")
                rank_zero_print(u)

        # trainer and callbacks
        trainer_kwargs = dict()

        # default logger configs
        default_logger_cfgs = {
            "wandb": {
                "target": "pytorch_lightning.loggers.WandbLogger",
                "params": {
                    "name": nowname,
                    "save_dir": logdir,
                    "offline": opt.debug,
                    "id": nowname,
                },
            },
            "testtube": {
                "target": "pytorch_lightning.loggers.TestTubeLogger",
                "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                },
            },
        }
        default_logger_cfg = default_logger_cfgs["testtube"]
        if "logger" in lightning_config:
            logger_cfg = lightning_config.logger
        else:
            logger_cfg = OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)

        my_logger = instantiate_from_config(logger_cfg)

        trainer_kwargs["logger"] = my_logger

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}-{step:09}",
                "verbose": True,
                "save_top_k": -1,
                "save_last": not lightning_config.just_eval_this_ckpt,
            },
        }
        if hasattr(model, "monitor") and not lightning_config.no_monitor:
            rank_zero_print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if "modelcheckpoint" in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        rank_zero_print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
        if version.parse(pl.__version__) < version.parse("1.4.0"):
            trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_debug.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "debug": opt.debug,
                },
            },
            "image_logger": {
                "target": "main_debug.ImageLogger",
                "params": {"batch_frequency": 750, "max_images": 4, "clamp": True},
            },
            "learning_rate_logger": {
                "target": "main_debug.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                },
            },
            "cuda_callback": {"target": "main_debug.CUDACallback"},
        }
        if version.parse(pl.__version__) >= version.parse("1.4.0"):
            default_callbacks_cfg.update({"checkpoint_callback": modelckpt_cfg})

        if "callbacks" in lightning_config:
            callbacks_cfg = lightning_config.callbacks
        else:
            callbacks_cfg = OmegaConf.create()

        if "metrics_over_trainsteps_checkpoint" in callbacks_cfg:
            rank_zero_print(
                "Caution: Saving checkpoints every n train steps without deleting. This might require some free space."
            )
            default_metrics_over_trainsteps_ckpt_dict = {
                "metrics_over_trainsteps_checkpoint": {
                    "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                    "params": {
                        "dirpath": os.path.join(ckptdir, "trainstep_checkpoints"),
                        "filename": "{epoch:06}-{step:09}",
                        "verbose": True,
                        "save_top_k": -1,
                        "every_n_train_steps": 10000,
                        "save_weights_only": True,
                    },
                }
            }
            default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        if "ignore_keys_callback" in callbacks_cfg and hasattr(trainer_opt, "resume_from_checkpoint"):
            callbacks_cfg.ignore_keys_callback.params["ckpt_path"] = trainer_opt.resume_from_checkpoint
        elif "ignore_keys_callback" in callbacks_cfg:
            del callbacks_cfg["ignore_keys_callback"]

        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        trainer_kwargs["callbacks"].extend(logdir_callbacks)

        if not "plugins" in trainer_kwargs:
            trainer_kwargs["plugins"] = list()
        if not lightning_config.get("find_unused_parameters", True):
            from pytorch_lightning.plugins import DDPPlugin
            trainer_kwargs["plugins"].append(DDPPlugin(find_unused_parameters=False))

        if MULTINODE_HACKS:
            # disable resume from hpc ckpts
            # NOTE below only works in later versions
            # from pytorch_lightning.plugins.environments import SLURMEnvironment
            # trainer_kwargs["plugins"].append(SLURMEnvironment(auto_requeue=False))
            # hence we monkey patch things
            from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector
            setattr(CheckpointConnector, "hpc_resume_path", None)

        if lightning_config.just_eval_this_ckpt:
            # hack to set everything up for trainer.test
            trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs, max_steps=1)
        else:
            trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        trainer.logdir = logdir  ###

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()
        rank_zero_print("#### Data ####")
        try:
            for k in data.datasets:
                rank_zero_print(
                    f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}"
                )
        except:
            rank_zero_print("datasets not yet initialized.")

        # configure learning rate
        bs, base_lr = (
            config.data.params.train_config.batch_size,
            config.model.base_learning_rate,
        )
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(","))
        else:
            ngpu = 1
        if "accumulate_grad_batches" in lightning_config.trainer:
            accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
        else:
            accumulate_grad_batches = 1
        rank_zero_print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        if opt.scale_lr:
            model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
            rank_zero_print(
                "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                    model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr
                )
            )
        else:
            model.learning_rate = base_lr
            rank_zero_print("++++ NOT USING LR SCALING ++++")
            rank_zero_print(f"Setting learning rate to {model.learning_rate:.2e}")

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                rank_zero_print("Saving checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb
                pudb.set_trace()

        # import signal

        # signal.signal(signal.SIGUSR1, melk)
        # signal.signal(signal.SIGUSR2, divein)
        # signal.signal(signal.SIGTERM, melk) # -- handles scancel from SLURM
        # signal.signal(signal.SIGCONT, melk) # -- handles scancel from SLURM
        # signal.signal(signal.SIGINT, melk) # -- handles scancel from SLURM

        print(
            "just_eval_this_ckpt",
            (
                lightning_config.just_eval_this_ckpt,
                type(lightning_config.just_eval_this_ckpt),
            ),
        )


        # -- train model
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                if not opt.debug:
                    melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            if lightning_config.just_eval_this_ckpt:
                # trainer.accelerator.setup(trainer, model)
                # run a single step of training
                trainer.fit(model, data)

                trainer.test(None, data, ckpt_path=lightning_config.just_eval_this_ckpt)
            else:
                trainer.test(model, data)

    except RuntimeError as err:
        if MULTINODE_HACKS:
            import requests
            import datetime
            import os
            import socket

            device = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
            hostname = socket.gethostname()
            ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            resp = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
            rank_zero_print(
                f"ERROR at {ts} on {hostname}/{resp.text} (CUDA_VISIBLE_DEVICES={device}): {type(err).__name__}: {err}",
                flush=True,
            )
        raise err
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            rank_zero_print(trainer.profiler.summary())
