if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import dill
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.train_robomimic_image_workspace import TrainRobomimicImageWorkspace
# set cudann off to calculate backward for RNN
torch.backends.cudnn.enabled = False

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainUniAdvPatchRobomimicImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.checkpoint = cfg.checkpoint

        # configure model
        payload = torch.load(open(self.checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']
        self.action_space = cfg_loaded.shape_meta.action.shape

        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.model = workspace.model
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        # dataset = hydra.utils.instantiate(cfg.task.dataset)
        dataset = RobomimicReplayImageDataset(shape_meta = cfg.task.dataset.shape_meta,
            dataset_path = cfg.task.dataset.dataset_path,
            horizon = cfg.task.dataset.horizon,
            pad_before = cfg.task.dataset.pad_before,
            pad_after=cfg.task.dataset.pad_after,
            n_obs_steps=cfg.task.dataset.n_obs_steps,
            abs_action=cfg.task.dataset.abs_action,
            # rotation_rep='rotation_6d', # ignored when abs_action=False
            rotation_rep=cfg.task.dataset.rotation_rep,
            use_legacy_normalizer=cfg.task.dataset.use_legacy_normalizer,
            use_cache=cfg.task.dataset.use_cache,
            seed=cfg.task.dataset.seed,
            val_ratio=cfg.task.dataset.val_ratio,
            )
        dataset = hydra.utils.call(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        # set the dataloader num of workers to min of specified and available
        cfg.dataloader.num_workers = min(cfg.dataloader.num_workers, os.cpu_count())
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        self.adv_patch = torch.zeros((1, 3, 84, 84)).to(self.model.device)

        # configure env
        # env_runner: BaseImageRunner
        # env_runner = hydra.utils.instantiate(
        #     cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # assert isinstance(env_runner, BaseImageRunner)

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.model.eval()

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        self.adv_patch, loss = self.model.train_adv_patch(batch, self.adv_patch, cfg)
                        train_losses.append(loss)
                        tepoch.set_postfix(loss=loss)
                    print(f"Epoch {self.epoch} train loss: {np.mean(train_losses)}")
                    self.epoch += 1


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainUniAdvPatchRobomimicImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()