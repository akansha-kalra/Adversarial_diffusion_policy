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
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.bet_image_policy import BETImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer, 
    SingleFieldLinearNormalizer
)
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainBETImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.policy: BETImagePolicy
        self.policy = hydra.utils.instantiate(cfg.policy)

        # configure training state
        self.optimizer = self.policy.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        print(cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        print("Loading dataset")
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = None
        if cfg.training.enable_normalizer:
            normalizer = dataset.get_normalizer()
        else:
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()


        self.policy.set_normalizer(normalizer)

        # fit action_ae (K-means)
        self.policy.fit_action_ae(
            normalizer['action'].normalize(dataset.get_all_actions())
        )
        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )
        print("Configuring environment runner")
        # configure env
        # cfg.task.env_runner.n_envs = 2
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.policy.to(device)
        optimizer_to(self.optimizer, device)

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
        print(f"Training for {cfg.training.num_epochs} epochs")

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.policy.obs_encoder.eval()
                    self.policy.obs_encoder.requires_grad_(False)
                    print("Freezing encoder")
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                        # compute loss
                        raw_loss, loss_components = self.policy.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.policy
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy, cfg=cfg)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, _ = self.policy.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        n_samples = cfg.training.sample_max_batch
                        batch = dict_apply(train_sampling_batch, 
                            lambda x: x.to(device, non_blocking=True))
                        obs_dict = dict_apply(batch['obs'], lambda x: x[:n_samples])
                        gt_action = batch['action']
                        
                        result = self.policy.predict_action(obs_dict)
                        pred_action = result['action']
                        start = cfg.n_obs_steps - 1
                        end = start + cfg.n_action_steps
                        gt_action = gt_action[:,start:end]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        print(pred_action.shape, gt_action.shape)
                        print("MSE: ", mse)
                        # log
                        step_log['train_action_mse_error'] = mse.item()
                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
import dill
import pickle 
import torch.nn.functional as F
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from baukit import Trace

def differentiable_normalize(tensor, eps=1e-8):
    B, C, H, W = tensor.shape
    reshaped = tensor.reshape(B*C, -1)
    min_vals = reshaped.min(dim=1, keepdim=True)[0]
    max_vals = reshaped.max(dim=1, keepdim=True)[0]
    normalized = (reshaped - min_vals) / (max_vals - min_vals + eps)
    return normalized.reshape(B, C, H, W)

class TrainBETUniPertImageWorkspaceDP(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir)
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        checkpoint = cfg.checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']

        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        try:
            self.model = workspace.model
        except:
            self.model = workspace.policy
        self.model.to(torch.device(cfg.training.device))

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        view = cfg.view
        device = cfg.training.device

        print("Loading dataset")
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        normalizer = None
        if cfg.training.enable_normalizer:
            normalizer = dataset.get_normalizer()
        else:
            normalizer = LinearNormalizer()
            normalizer['action'] = SingleFieldLinearNormalizer.create_identity()
            normalizer['obs'] = SingleFieldLinearNormalizer.create_identity()


        self.model.set_normalizer(normalizer)

        # fit action_ae (K-means)
        self.model.fit_action_ae(
            normalizer['action'].normalize(dataset.get_all_actions())
        )
        print("Configuring environment runner")
        # configure env
        # cfg.task.env_runner.n_envs = 2
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        if cfg.log:
            wandb.init(
                project="regularized_universal_perturbation_test",
                name=f"BET_{cfg.epsilon}_targeted_{cfg.targeted}_view_{view}_orig_obs"
            )
            wandb.log({"epsilon": cfg.epsilon, "epsilon_step": cfg.epsilon_step, "targeted": cfg.targeted, "view": view})        # configure checkpoint
        self.model.eval()

        # device transfer
        self.model.to(device)
        # obs_encoder = self.model.obs_encoder
        # print the class of observation encoder
        # image_encoder = self.model.obs_encoder.obs_nets[cfg.view]
        # net = image_encoder.backbone.nets
        # print(f'Net is {net}')

        if cfg.log:
            wandb.log({'eta': cfg.eta, 'lambda_feat': cfg.lambda_feat, 'kernel_size': cfg.kernel_size})
        # training loop for the universal perturbation
        self.univ_pert = {}
        if cfg.targeted:
            self.univ_pert['perturbations'] = torch.tensor(cfg.perturbations, device='cpu')
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        else:
            views = [view]
        for view in views:
            self.univ_pert[view] = torch.zeros((3, 84, 84)).to(device)

        # add some initial noise to the self.univ_pert for untargeted attacks
        if cfg.targeted == False:
            for view in views:
                self.univ_pert[view] = torch.rand((3, 84, 84)).to(device) * 2 * cfg.epsilon - cfg.epsilon
        # define the perturbation to be random noise within epsilon
        # self.univ_pert = torch.rand((1, 3, 84, 84)).to(device) * 2 * cfg.epsilon - cfg.epsilon
        if cfg.retrain:
            self.univ_pert = pickle.load(open(cfg.patch_path, 'rb'))
            print(f"Loaded perturbation with shape {self.univ_pert.shape}")
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
        print(f"Training for {cfg.training.num_epochs} epochs")

        # Define a 3x3 box filter kernel
        channels = 3
        kernel_size = cfg.kernel_size
        ws = torch.ones((channels, 1, kernel_size, kernel_size)) /(kernel_size**2)
        ws = ws.to(device)
        eta = cfg.eta
        self.layers = ['0.nets.6.1.conv1']
        # self.adversarial_image = torch.zeros((512, 3, 84, 84)).to(device)

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        To = self.model.n_obs_steps
        switch = cfg.switch
        gradients = {}
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                loss_per_epoch = 0
                if cfg.view == 'both':
                    total_grad = {}
                    for view in views:
                        total_grad[view] = torch.zeros((1, 2, 3, 84, 84)).to(device)
                else:
                    total_grad = torch.zeros((1, 2, 3, 84, 84)).to(device)
                raw_loss = 0
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        distance = 0
                        torch.cuda.empty_cache()
                        self.model.zero_grad()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        original_obs = batch['obs'].copy()
                        # with torch.no_grad():
                        #     nobs_clean = self.model.normalizer.normalize(batch['obs'])
                        #     this_nobs_clean = dict_apply(nobs_clean, 
                        #         lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                        #     nobs_features_clean = self.model.obs_encoder(this_nobs_clean)
                        #     # raw_loss_orig, loss_components_orig, nobs_features_clean = self.model.compute_loss(batch, return_nobs_feat=True)
                        #     # representation_dict_orig = self.model.obs_encoder.representation_dict
                        obs = batch['obs'].copy()
                        # if batch['obs']['agentview_image'].shape[0] != cfg.dataloader.batch_size:
                        #     continue
                        # clamp the observation to be between clip_min and clip_max
                        for view in views:
                            obs[view] = obs[view] + self.univ_pert[view]
                            # clamp the observation to be between 0 and 1
                            obs[view] = torch.clamp(obs[view], 0, 1)
                            obs[view].requires_grad = True
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        batch_cp = batch.copy()
                        if cfg.targeted:
                            with torch.no_grad():
                                predicted_action = self.model.predict_action(original_obs)['action']
                            batch_cp['obs'] = obs
                            # batch_cp['action'] = predicted_action.to(device) + torch.tensor(cfg.perturbations, device =device)
                            batch_cp['action'] = batch['action'] + torch.tensor(cfg.perturbations, device =device)
                            # target_action = predicted_action + torch.tensor(cfg.perturbations, device =device)
                            # predicted_action = self.model.predict_action(obs)['action']
                            raw_loss, loss_components = self.model.compute_loss(batch_cp)
                            # if self.epoch < switch:
                            #     raw_loss = -loss_components['class']
                            # else:
                            #     raw_loss = -loss_components['offset']
                            raw_loss = -raw_loss
                            # raw_loss = -torch.nn.MSELoss()(predicted_action, target_action)
                        else:
                            batch_cp['obs'] = obs
                            # nobs_pert = self.model.normalizer.normalize(batch['obs'])
                            # this_nobs_pert = dict_apply(nobs_pert,
                            #     lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                            # nobs_features_pert = self.model.obs_encoder(this_nobs_pert)
                            # print(nobs_features_clean.shape, nobs_features_pert.shape)
                            # loss = torch.nn.MSELoss()(nobs_features_clean, nobs_features_pert)
                            # print(batch['obs'][view].shape)
                            raw_loss, loss_components = self.model.compute_loss(batch_cp)
                            # raw_loss, loss_components, nobs_features_pert = self.model.compute_loss(batch, return_nobs_feat=True)
                            # class_loss = loss_components['class']
                            # distance = torch.nn.MSELoss()(nobs_features_clean, nobs_features_pert)
                            # print(raw_loss, loss_components)
                            # representation_dict_pert = self.model.obs_encoder.representation_dict
                            # for layer in self.layers:
                            #     representation_orig = representation_dict_orig[layer]
                            #     representation_pert = representation_dict_pert[layer]
                                # calculate the cosine similarity between the feature maps
                                # distance += F.cosine_similarity(representation_orig, representation_pert, dim=1).mean()
                                # normalize the feature maps to [0, 1] based on min-max normalization
                                # print(torch.min(representation_orig), torch.max(representation_orig))
                                # print(torch.min(representation_pert), torch.max(representation_pert))
                                # representation_orig = differentiable_normalize(representation_orig)
                                # representation_pert = differentiable_normalize(representation_pert)
                                # print(torch.min(representation_orig), torch.max(representation_orig))
                                # print(torch.min(representation_pert), torch.max(representation_pert))
                                # orig_feat = torch.sign(representation_orig) * (torch.abs(representation_orig)**cfg.alpha)
                                # pert_feat = torch.sign(representation_pert) * (torch.abs(representation_pert)**cfg.alpha)
                            #   print(f"Shape of original and perturbed feature maps: {orig_feat.shape}, {pert_feat.shape}")
                                # distance += torch.norm(orig_feat - pert_feat, p = 2)
                                # print(f"Distance between original and perturbed feature maps: {distance}")
                            # print(f"Distance between original and perturbed feature maps: {distance}")
                            # if cfg.log:
                            #     wandb.log({"distance": distance.item()})
                            # response_map = F.conv2d((obs[view][:, 1] - original_obs[view][:, 1]), ws, groups=3)
                            # reg_term = torch.sum(torch.abs(response_map))
                            # print(reg_term)
                            # loss += raw_loss + eta * reg_term + cfg.lambda_feat * distance
                            # class_loss += eta * reg_term
                            # raw_loss += cfg.lambda_feat * distance
                            # raw_loss += eta * reg_term
                            # print(f"Loss after regularization: {raw_loss}")
                            # print(f"Mean and std of univ perturbation: {torch.mean(self.univ_pert)}, {torch.std(self.univ_pert)}")
                        # compute loss
                        # loss = distance
                        loss = raw_loss
                        if self.epoch == 0:
                            loss_per_epoch += loss.item()
                            continue
                        loss.backward()
                        loss_per_epoch += loss.item()
                        # update the perturbation
                        if cfg.view == 'both':
                            for view in views:
                                total_grad[view] += obs[view].grad.sum(dim=0, keepdim=True)
                        else:
                            total_grad += obs[view].grad.sum(dim=0, keepdim=True)
                        # log the magnitude of the gradient
                        if cfg.log:
                            wandb.log({"gradient_magnitude": torch.norm(obs[view].grad).item()})
                            # log the loss components
                            for key, value in loss_components.items():
                                wandb.log({key: value.item()})
                # gradients[self.epoch] = total_grad
                for view in views:
                    self.univ_pert[view] = self.univ_pert[view] + cfg.epsilon_step * torch.sign(total_grad[view])
                    self.univ_pert[view] = torch.clamp(self.univ_pert[view], -cfg.epsilon, cfg.epsilon)
                print(f"Loss per epoch: {loss_per_epoch}")
                for view in views:
                    if cfg.log:
                        wandb.log({f"mean_pert_{view}": torch.mean(self.univ_pert[view]).item(), f"std_pert_{view}": torch.std(self.univ_pert[view]).item()})
                    else:
                        print(f"Mean and std of perturbation for {view}: {torch.mean(self.univ_pert[view]).item()}, {torch.std(self.univ_pert[view]).item()}")
                # f_perturbation = fft2(self.univ_pert[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
                # # visualize the magnitude of the perturbation spectrum
                # plt.imshow(np.log(np.abs(f_perturbation)+1))
                # plt.title('Fourier Transform of the perturbation')
                # pert_save_path = os.path.join(os.path.dirname(cfg.checkpoint), f'fourier_pert_{cfg.epsilon}_epoch_{self.epoch}_{view}.png')
                # plt.savefig(pert_save_path)
                # smoothed_perturbation = F.conv2d(self.univ_pert, ws, groups=3)
                # print("shape of smoothed perturbation is ", smoothed_perturbation.shape)
                # # Visualize the smoothed magnitude spectrum
                # plt.imshow(smoothed_perturbation[0].squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
                # plt.title('Smoothed Perturbation')
                # smooth_save_path = os.path.join(os.path.dirname(cfg.checkpoint), f'smoothed_fourier_pert_{cfg.epsilon}_epoch_{self.epoch}_{view}.png')
                # plt.savefig(smooth_save_path)
                

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model, self.univ_pert, cfg)
                    # log all
                    step_log.update(runner_log)
                    test_mean_score= runner_log['test/mean_score']
                    print(f"Test mean score: {test_mean_score}")
                    save_name = f'pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{cfg.view}.pkl'
                    if cfg.log:
                        wandb.log({"test_mean_score": test_mean_score, "epoch": self.epoch})
                    if cfg.retrain:
                        patch_path = cfg.patch_path[:-4] + '_retrain.pkl'
                    elif cfg.targeted:
                        # patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'tar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}_ker_{cfg.kernel_size}_with_reg{cfg.eta}.pkl')
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'tar_{save_name}')
                        perturbations = cfg.perturbations.copy()
                        perturbations = torch.tensor(perturbations).cpu()
                        self.univ_pert['perturbations'] = perturbations
                    else:
                        # patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'untar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}_ker_{cfg.kernel_size}_with_reg{cfg.eta}.pkl')
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'untar_{save_name}')
                    pickle.dump(self.univ_pert, open(patch_path, 'wb'))
                self.epoch += 1
                json_logger.log(step_log)
                self.global_step += 1
        # if cfg.targeted:
        #     gradients_path = os.path.join(os.path.dirname(cfg.checkpoint), f'gradients_tar_{save_name}')
        # else:
        #     gradients_path = os.path.join(os.path.dirname(cfg.checkpoint), f'gradients_untar_{save_name}')
        # pickle.dump(gradients, open(gradients_path, 'wb'))
        if cfg.log:
            wandb.finish()



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainIbcDfoHybridWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

