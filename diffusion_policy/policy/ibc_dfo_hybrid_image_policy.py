from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import sys
import numpy as np
import pickle


class IbcDfoHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1,
            train_n_neg=128,
            pred_n_iter=5,
            pred_n_samples=16384,
            kevin_inference=False,
            andy_train=False,
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            crop_shape=(76, 76),
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.obs_encoder = obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        in_action_channels = action_dim * n_action_steps
        in_obs_channels = obs_feature_dim * n_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_energy=False, adversarial_action=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Ta = self.n_action_steps
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps
        self.pred_n_iter = 5

        # build input
        device = self.device
        dtype = self.dtype

        # encode obs
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        # print(this_nobs.keys())
        nobs_features = self.obs_encoder(this_nobs)
        # print(nobs_features.shape)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)

        # only take necessary obs
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=dtype)
        if adversarial_action is not None:
            # sneak in the adversarial action in the samples
            samples[:, 0, :] = adversarial_action
        # (B, N, Ta, Da)
        # print("Number of iterations", self.pred_n_iter)
        # print(nobs_features.shape)
        # save the obs_dict['agentview_image'] to a npy file
        # np.save('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/obs_dict.npy', obs_dict['agentview_image'])
        # samplesxlogits = {}

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(nobs_features, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(nobs_features, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(nobs_features, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)
                # samplesxlogits[i] = (samples, logits)
                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)
            # Return one sample per x in batch.
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)
        
        # save the samplesxlogits to a pickle file
        # with open('/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/samplesxlogits.pkl', 'wb') as f:
        #     pickle.dump(samplesxlogits, f)
        # exit
        # sys.exit(0)
        action = self.normalizer['action'].unnormalize(acts_n)
        result = {
            'action': action
        }
        if return_energy:
            result['energy'] = logits
            result['samples'] = samples
        return result


    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])

        # shapes
        Do = self.obs_feature_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        # encode obs
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)

        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        action_samples = torch.cat([
            this_action.unsqueeze(1), samples], dim=1)
         # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(nobs_features, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(nobs_features, action_samples)
            loss = F.cross_entropy(logits, labels)
        return loss

    def compute_loss_with_grad(self, obs_dict_copy, actions, action_samples = None):
        """
        Similar to the compute_loss function, but does not create 
        a new observation dictionary for the forward pass. Instead,
        the observation dictionary is passed directly to the forward
        pass, for backpropagation the gradient of the observation.
        """
        # normalize input
        # assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(obs_dict_copy)
        naction = self.normalizer['action'].normalize(actions)

        # shapes
        Do = self.obs_feature_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        # encode obs
        # reshape B, T, ... to B*T
        nobs = dict_apply(nobs,
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(nobs)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)
        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        if action_samples is None:
            action_samples = torch.cat([
                this_action.unsqueeze(1), samples], dim=1)
        # print("Action samples in compute_loss: ", action_samples[0, 10, :])
        # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(nobs_features, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(nobs_features, action_samples)
            loss = F.cross_entropy(logits, labels)
        # print the actions given and the most probable action
        # print("Actions given: ", actions)
        # most_probable_action = torch.argmax(logits, dim=1)
        # most_probable_action = action_samples[torch.arange(action_samples.size(0)), most_probable_action, :]
        # print("Most probable action: ", most_probable_action)
        # print('Energy of the first action: ', logits[0, 0])
        # print('Energy of the second action: ', logits[0, 1])
        # print('Energy of the most probable action: ', torch.max(logits[0, 1:]))
        return loss, obs_dict_copy, nobs_features
 

    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats


    def train_adv_patch(self, batch, adv_patch, mask, cfg):
        """
        Train an adversarial patch for the batch of images.
        """
        B = batch['obs'][cfg.view].shape[0]
        T_neg = self.train_n_neg
        T_a = self.n_action_steps
        if len(adv_patch.shape) == 3:
            adv_patch = adv_patch.unsqueeze(0).to(self.device)
        else:
            adv_patch = adv_patch.to(self.device)
        obs_dict = batch['obs']
        obs_dict = dict_apply(obs_dict, lambda x: x.to(self.device))
        # check if the obs_dict[cfg.view] is in the correct range
        assert torch.min(obs_dict[cfg.view].flatten()) >= 0
        assert torch.max(obs_dict[cfg.view].flatten()) <= 1
        perturbed_view = obs_dict[cfg.view] * (1 - mask) + adv_patch * mask
        perturbed_view = perturbed_view.clamp(0, 1)
        perturbed_obs_dict = obs_dict.copy()

        perturbed_obs_dict[cfg.view] = perturbed_view
        perturbed_obs_dict = dict_apply(perturbed_obs_dict, lambda x: x.clone().detach().requires_grad_(True))

        predicted_action = self.predict_action(obs_dict)['action']
        
        actions = batch['action']
        actions = actions[:, -self.n_action_steps:, :]
        # target_actions = actions + torch.tensor(cfg.perturbations).unsqueeze(0).to(self.device)
        target_actions = predicted_action + torch.tensor(cfg.perturbations).unsqueeze(0).to(self.device)
        # clean_actions = target_actions
        clean_actions = predicted_action
        clean_actions = self.normalizer['action'].normalize(clean_actions)
        naction_stats = self.get_naction_stats()
        action_samples = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        ).sample((B, T_neg, T_a)).to(device=self.device)
        # print(min(action_samples.flatten()), max(action_samples.flatten()))
        action_samples = torch.cat([target_actions.unsqueeze(1), action_samples], dim=1)
        action_samples[:, 1, ...] = clean_actions
        # print(min(action_samples.flatten()), max(action_samples.flatten()))
        prev_obs_dict = obs_dict.copy()
        for j in range(cfg.n_iter):
            loss, perturbed_obs_dict, nobs_features = self.compute_loss_with_grad(perturbed_obs_dict, target_actions, action_samples)
            loss = -loss
            loss.backward()
            print(f'Loss at iteration {j}: {loss.item()}')
            assert perturbed_obs_dict[cfg.view].grad is not None
            grad = torch.sign(perturbed_obs_dict[cfg.view].grad)
            grad = grad * mask
            grad = torch.sum(grad, dim=0)
            adv_patch = adv_patch + cfg.eps_iter * grad
            adv_patch = torch.clamp(adv_patch, -cfg.eps, cfg.eps)
            perturbed_obs_dict[cfg.view] = torch.clamp(obs_dict[cfg.view] * (1 - mask) + adv_patch * mask, \
                cfg.clip_min, cfg.clip_max)
            # perturbed_obs_dict[cfg.view] = torch.clamp(perturbed_obs_dict[cfg.view] + cfg.eps_iter * grad, 0, 1)
            perturbed_obs_dict = dict_apply(perturbed_obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        adv_patch = torch.clamp(adv_patch, -cfg.eps, cfg.eps)
        return adv_patch, loss.item()

    def action_dist(self, obs_dict: Dict[str, torch.Tensor]):
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Ta = self.n_action_steps    # 1 for lift
        Da = self.action_dim    # 7 for lift
        Do = self.obs_feature_dim #137 for lift
        To = self.n_obs_steps
        # build input
        device = self.device
        dtype = self.dtype

        # encode obs
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        # print(this_nobs.keys())
        nobs_features = self.obs_encoder(this_nobs)
        # print(nobs_features.shape)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)

        # only take necessary obs
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=dtype)

        # (B, N, Ta, Da).  (B, 1024, 1, 7) for lift
        # print("Number of iterations", self.pred_n_iter)
        # print(nobs_features.shape)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(nobs_features, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(nobs_features, samples)
            probs = F.softmax(logits, dim=-1)
            return probs
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(nobs_features, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)
            return prob
