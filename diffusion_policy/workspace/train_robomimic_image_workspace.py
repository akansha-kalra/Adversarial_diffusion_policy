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
import torch.utils
from torch.utils.data import DataLoader
import copy
import random
import torch.utils.data
import wandb
import tqdm
import numpy as np
import shutil
import pickle
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.robomimic_image_policy import RobomimicImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to


OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainRobomimicImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: RobomimicImagePolicy = hydra.utils.instantiate(cfg.policy)

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
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
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
        self.model.to(device)

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
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        info = self.model.train_on_batch(batch, epoch=self.epoch)

                        # logging 
                        loss_cpu = info['losses']['action_loss'].item()
                        tepoch.set_postfix(loss=loss_cpu, refresh=False)
                        train_losses.append(loss_cpu)
                        step_log = {
                            'train_loss': loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch
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
                self.model.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(self.model, cfg=cfg)
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
                                info = self.model.train_on_batch(batch, epoch=self.epoch, validate=True)
                                loss = info['losses']['action_loss']
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
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        T = gt_action.shape[1]

                        pred_actions = list()
                        self.model.reset()
                        for i in range(T):
                            result = self.model.predict_action(
                                dict_apply(obs_dict, lambda x: x[:,[i]])
                            )
                            pred_actions.append(result['action'])
                        pred_actions = torch.cat(pred_actions, dim=1)
                        mse = torch.nn.functional.mse_loss(pred_actions, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_actions
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
                self.model.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

import dill
class TrainRobomimicUniPertImageWorkspace(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        checkpoint = cfg.checkpoint
        print(f" Check you are seeding correctly {seed} for {checkpoint} with num_test envs {cfg.task.env_runner.n_test} and n_envs {cfg.task.env_runner.n_envs}")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg_loaded = payload['cfg']

        cls = hydra.utils.get_class(cfg_loaded._target_)
        workspace = cls(cfg_loaded, output_dir=output_dir)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        try:
            self.model = workspace.model
        except AttributeError:
            self.model = workspace.policy
        self.model.to(torch.device(cfg.training.device))
        # configure training state
        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        view = cfg.view
        device = cfg.training.device
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # self.layers = ['0.nets.4.0.conv1']
        # self.layers = ['0.nets.1']
        # self.layers = ['0.nets.2.1.conv1']
        self.layers = []


        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            cfg.log = False
            

        if cfg.log:
            wandb.init(
                project="Adv_diffusion_policy",
                name=f"{cfg.exp_name}-vanilla_bc_{cfg.epsilon}_targeted_{cfg.targeted}_view_{view}_feature_std"
            )
            wandb.log({"epsilon": cfg.epsilon, "epsilon_step": cfg.epsilon_step, "targeted": cfg.targeted, "view": view,"seed": cfg.training.seed, "pretrained_checkpoint":str(cfg.checkpoint)})
        # set the model in eval mode
        self.model.eval()
        # training loop for the universal perturbation
        self.univ_pert = {}
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        else:
            views = [view]
        for view in views:
            self.univ_pert[view] = torch.zeros((3, 84, 84)).to(device)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        gradients = {}
        cfg_activation = {}
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                loss_per_epoch = 0
                if cfg.view == 'both':
                    total_grad = {}
                    for view in views:
                        total_grad[view] = torch.zeros((1, 1, 3, 84, 84)).to(device)
                else:
                    total_grad = torch.zeros((1, 1, 3, 84, 84)).to(device)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        self.model.zero_grad()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        orig_obs = batch['obs'].copy()
                        cfg_activation['modify_act'] = False
                        with torch.no_grad():
                            prediction2 = self.model.predict_action(orig_obs, cfg_activation=cfg_activation)
                            predicted_action2 = prediction2['action'].to(device)
                            predicted_features2 = prediction2['features'].to(device)
                            representation_dict_orig = self.model.nets['policy'].nets['encoder'].nets['obs'].representation_dict
                        obs = batch['obs'].copy()
                        # apply the patch the view
                        for view in views:
                            obs[view] = obs[view] + self.univ_pert[view]
                            # clamp the observation to be between 0 and 1
                            obs[view] = torch.clamp(obs[view], 0, 1)
                            obs[view].requires_grad = True
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        # set the requires_grad to true
                        cfg_activation['modify_act'] = False
                        cfg_activation['gamma'] = cfg.gamma
                        # cfg_activation['orig_act'] = representation_dict_orig[self.layers[0]]
                        prediction = self.model.predict_action(obs, cfg_activation=cfg_activation)
                        predicted_action = prediction['action'].to(device)
                        predicted_features = prediction['features'].to(device)
                        if cfg.targeted:
                            batch['action'] = batch['action'] + torch.tensor(cfg.perturbations).to(device)
                            if cfg.perturbations[-1] == -1:
                                batch['action'][:, :, -1] = -1
                            # target_action = predicted_action2 + torch.tensor(cfg.perturbations).to(device)
                            loss = -torch.nn.functional.mse_loss(predicted_action, batch['action'])
                        else:
                            loss = torch.nn.functional.mse_loss(predicted_action, batch['action'])
                            # loss = torch.nn.functional.mse_loss(predicted_action, predicted_action2)
                            # loss = torch.nn.functional.mse_loss(predicted_features, predicted_features2)
                            # loss = -1 * predicted_features.std()
                        # loss = torch.nn.functional.mse_loss(predicted_action, predicted_action2)
                        # take the gradient of the loss with respect to the perturbation
                        if self.epoch == 0:
                            loss_per_epoch += loss.item()
                            continue
                        loss.backward()
                        loss_per_epoch += loss.item()
                        if cfg.view == 'both':
                            for view in views:
                                total_grad[view] += obs[view].grad.sum(dim=0, keepdim=True)
                        else:
                            total_grad += obs[view].grad.sum(dim=0, keepdim=True)
                        # update the perturbation
                        # self.univ_pert = self.univ_pert + cfg.epsilon_step * torch.sum(obs[view].grad.sign(), dim=0)
                        # clip the perturbation
                        # self.univ_pert = torch.clamp(self.univ_pert, -cfg.epsilon, cfg.epsilon)
                # gradients[self.epoch] = total_grad
                for view in views:
                    self.univ_pert[view] = self.univ_pert[view] + cfg.epsilon_step * torch.sign(total_grad[view])
                    self.univ_pert[view] = torch.clamp(self.univ_pert[view], -cfg.epsilon, cfg.epsilon)
                print(f"Loss for {self.epoch}: {loss_per_epoch}")
                if cfg.log:
                    wandb.log({"loss": loss_per_epoch, "epoch": self.epoch})
                # print(f"Linf norm of the perturbation: {torch.norm(self.univ_pert, p=float('inf'))}")
                for view in views:
                    print(f"L2 norm of the {view} perturbation: {torch.norm(self.univ_pert[view], p=2)}")
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch != 0:
                    runner_log = env_runner.run(self.model, self.univ_pert, cfg)
                    # log all
                    step_log.update(runner_log)
                    test_mean_score= runner_log['test/mean_score']
                    print(f"Test mean score: {test_mean_score}")
                    if cfg.log:
                        wandb.log({"test_mean_score": test_mean_score, "epoch": self.epoch})
                    # save the patch
                    if cfg.targeted:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_tar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{cfg.view}.pkl')
                    else:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_untar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{cfg.view}.pkl')
                    pickle.dump(self.univ_pert, open(patch_path, 'wb'))
                self.epoch += 1
        # gradients_path = os.path.join(os.path.dirname(cfg.checkpoint), f'gradients_untar_pert_{cfg.epsilon}_{cfg.gamma}_feature_dist.pkl')
        # pickle.dump(gradients, open(gradients_path, 'wb'))
        wandb.finish()

import dill
# turn off cudann for backprop
torch.backends.cudnn.enabled = False
class TrainRobomimicUniPertImageWorkspaceRNN(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

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
        except AttributeError:
            self.model = workspace.policy
        self.model.to(torch.device(cfg.training.device))
        # configure training state
        self.global_step = 0
        self.epoch = 0


    def run(self):
        cfg = copy.deepcopy(self.cfg)
        view = cfg.view
        device = cfg.training.device
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)


        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if cfg.log:
            wandb.init(
                project="Adv_diffusion_policy",
                name=f"{cfg.exp_name}-lstm_gmm_{cfg.epsilon}_targeted_{cfg.targeted}_view_{view}"
            )
            wandb.log({"epsilon": cfg.epsilon, "epsilon_step": cfg.epsilon_step, "targeted": cfg.targeted, "view": view})
        # set the model in eval mode
        self.model.eval()
        image_shape = cfg.task['image_shape']
        # training loop for the universal perturbation
        self.univ_pert = {}
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        else:
            views = [view]
        for view in views:
            # self.univ_pert[view] = torch.zeros((3, 84, 84)).to(device)
            self.univ_pert[view] = torch.zeros((image_shape[0], image_shape[1], image_shape[2])).to(device)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        gradients = {}
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                loss_per_epoch = 0
                total_grad = {}
                for view in views:
                    total_grad[view] = torch.zeros((1, cfg.n_obs_steps, image_shape[0], image_shape[1], image_shape[2])).to(device)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        self.model.zero_grad()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        obs = batch['obs']
                        if batch['obs'][view].shape[0] != cfg.dataloader.batch_size:
                            continue
                        for view in views:
                            print(f"obs shape: {obs[view].shape}, pert shape: {self.univ_pert[view].shape}")
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        with torch.no_grad():
                            action_dist_clean = self.model.action_dist(obs)
                            action_means_clean = action_dist_clean.component_distribution.base_dist.loc
                        for view in views:
                            # print(f"obs shape: {obs[view].shape}, pert shape: {self.univ_pert[view].shape}")
                            # print(f"obs shape: {obs[view].shape}, pert shape: {self.univ_pert[view].shape}")
                            # obs[view] = obs[view] + self.univ_pert[view]
                            obs[view][:, :cfg.n_obs_steps, ...] = obs[view][:, :cfg.n_obs_steps, ...] + self.univ_pert[view]
                            # clamp the observation to be between 0 and 1
                            obs[view] = torch.clamp(obs[view], 0, 1)
                            obs[view].requires_grad = True

                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        action_dist = self.model.action_dist(obs)
                        action_means = action_dist.component_distribution.base_dist.loc
                        if cfg.targeted:
                            action_means_clean = action_means_clean + torch.tensor(cfg.perturbations).to(device)
                            loss = -torch.nn.functional.mse_loss(action_means, action_means_clean)
                            # loss = -torch.nn.functional.mse_loss(predicted_action, batch['action'])
                        else:
                            loss = torch.nn.functional.mse_loss(action_means, action_means_clean)
                        # loss = torch.nn.functional.mse_loss(predicted_action, predicted_action2)
                        # take the gradient of the loss with respect to the perturbation
                        loss.backward()
                        loss_per_epoch += loss.item()
                        for view in views:
                            total_grad[view] += obs[view].grad.sum(dim=0, keepdim=True)[:, :cfg.n_obs_steps, ...]
                for view in views:
                    # print(f"Total grad shape: {total_grad[view].shape}")
                    self.univ_pert[view] = self.univ_pert[view] + cfg.epsilon_step * torch.sign(total_grad[view])
                    self.univ_pert[view] = torch.clamp(self.univ_pert[view], -cfg.epsilon, cfg.epsilon)
                # gradients[self.epoch] = total_grad
                for view in views:
                    # print(f"Total grad shape: {total_grad[view].shape}")
                    self.univ_pert[view] = self.univ_pert[view] + cfg.epsilon_step * torch.sign(total_grad[view])
                    self.univ_pert[view] = torch.clamp(self.univ_pert[view], -cfg.epsilon, cfg.epsilon)

                print(f"Loss for {self.epoch}: {loss_per_epoch}")
                if cfg.log:
                    wandb.log({"loss": loss_per_epoch, "epoch": self.epoch})
                # print(f"Linf norm of the perturbation: {torch.norm(self.univ_pert, p=float('inf'))}")
                for view in views:
                    print(f"L2 norm of the {view} perturbation: {torch.norm(self.univ_pert[view], p=2)}")
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch != 0:
                    runner_log = env_runner.run(self.model, self.univ_pert, cfg)
                    # log all
                    step_log.update(runner_log)
                    test_mean_score= runner_log['test/mean_score']
                    print(f"Test mean score: {test_mean_score}")
                    if cfg.log:
                        wandb.log({"test_mean_score": test_mean_score, "epoch": self.epoch})
                    # save the patch
                    if cfg.targeted:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_tar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}.pkl')
                    else:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_untar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}.pkl')
                    pickle.dump(self.univ_pert, open(patch_path, 'wb'))
                self.epoch += 1
        # gradients_path = os.path.join(os.path.dirname(cfg.checkpoint), f'gradients_untar_pert_{cfg.epsilon}.pkl')
        # pickle.dump(gradients, open(gradients_path, 'wb'))
        wandb.finish()

class TrainRobomimicUniPertImageWorkspaceIBC(BaseWorkspace):

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

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
        except AttributeError:
            self.model = workspace.policy
        self.model.to(torch.device(cfg.training.device))
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        view = cfg.view
        device = cfg.training.device
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)


        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        if cfg.log:
            wandb.init(
                project="Adv_diffusion_policy",
                name=f"{cfg.exp_name}-lstm_gmm_{cfg.epsilon}_targeted_{cfg.targeted}_view_{view}"
            )
            wandb.log({"epsilon": cfg.epsilon, "epsilon_step": cfg.epsilon_step, "targeted": cfg.targeted, "view": view})
        # set the model in eval mode
        self.model.eval()
        # training loop for the universal perturbation
        self.univ_pert = {}
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        else:
            views = [view]
        for view in views:
            self.univ_pert[view] = torch.zeros((3, 84, 84)).to(device)
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
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
                        total_grad[view] = torch.zeros((1, 1, 3, 84, 84)).to(device)
                else:
                    total_grad = torch.zeros((1, 1, 3, 84, 84)).to(device)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        self.model.zero_grad()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        obs = batch['obs'].copy()
                        for view in views:
                            obs[view] = obs[view] + self.univ_pert[view]
                            # clamp the observation to be between 0 and 1
                            obs[view] = torch.clamp(obs[view], 0, 1)
                            obs[view].requires_grad = True
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        with torch.no_grad():
                            action_dist_clean = self.model.action_dist(obs)
                            action_means_clean = action_dist_clean.component_distribution.base_dist.loc
                        # apply the patch the view
                        obs[view] = obs[view] + self.univ_pert
                        # clamp the observation to be between 0 and 1
                        obs[view] = torch.clamp(obs[view], 0, 1)
                        obs = dict_apply(obs, lambda x: x.to(device, non_blocking=True))
                        # set the requires_grad to true
                        obs[view].requires_grad = True
                        if cfg.targeted:
                            pass
                            # loss = -torch.nn.functional.mse_loss(predicted_action, batch['action'])
                        else:
                            batch['obs'] = obs
                            loss = self.model.compute_loss(batch)
                        # loss = torch.nn.functional.mse_loss(predicted_action, predicted_action2)
                        # take the gradient of the loss with respect to the perturbation
                        loss.backward()
                        loss_per_epoch += loss.item()
                        if cfg.view == 'both':
                            for view in views:
                                total_grad[view] += obs[view].grad.sum(dim=0, keepdim=True)
                        else:
                            total_grad += obs[view].grad.sum(dim=0, keepdim=True)
                gradients[self.epoch] = total_grad
                for view in views:
                    self.univ_pert[view] = self.univ_pert[view] + cfg.epsilon_step * torch.sign(total_grad[view])
                    self.univ_pert[view] = torch.clamp(self.univ_pert[view], -cfg.epsilon, cfg.epsilon)
                print(f"Loss for {self.epoch}: {loss_per_epoch}")
                if cfg.log:
                    wandb.log({"loss": loss_per_epoch, "epoch": self.epoch})
                # print(f"Linf norm of the perturbation: {torch.norm(self.univ_pert, p=float('inf'))}")
                for view in views:
                    print(f"L2 norm of the {view} perturbation: {torch.norm(self.univ_pert[view], p=2)}")
                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.epoch != 0:
                    runner_log = env_runner.run(self.model, self.univ_pert, cfg)
                    # log all
                    step_log.update(runner_log)
                    test_mean_score= runner_log['test/mean_score']
                    print(f"Test mean score: {test_mean_score}")
                    if cfg.log:
                        wandb.log({"test_mean_score": test_mean_score, "epoch": self.epoch})
                    # save the patch
                    if cfg.targeted:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_tar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}_feature_std.pkl')
                    else:
                        patch_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_untar_pert_{cfg.epsilon}_epoch_{self.epoch}_mean_score_{test_mean_score}_{view}_feature_std.pkl')
                    pickle.dump(self.univ_pert, open(patch_path, 'wb'))
                self.epoch += 1
        gradients_path = os.path.join(os.path.dirname(cfg.checkpoint), f'{cfg.exp_name}_gradients_untar_pert_{cfg.epsilon}.pkl')
        pickle.dump(gradients, open(gradients_path, 'wb'))
        wandb.finish()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainRobomimicImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()