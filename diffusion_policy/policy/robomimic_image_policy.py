from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class RobomimicImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            algo_name='bc_rnn',
            obs_type='image',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76)
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
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)

        
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
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result

    def action_dist(self, obs_dict: Dict[str, torch.Tensor]):
        nobs_dict = self.normalizer(obs_dict)
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        action_dist = self.model.get_action_dist(robomimic_obs_dict)
        return action_dist


    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch, validate=False):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        robomimic_batch = {
            'obs': nobs,
            'actions': nactions
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info
    
    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers['policy']

    def train_adv_patch(self, batch, adv_patch, mask, cfg):
        """
        Train the adversarial patch on the batch of data
        Based on paper: https://arxiv.org/pdf/1610.08401
        For each image in the batch, try to perturb the image according to 
        the perturbation specified in the config
        """
        # turn the adversarial patch into a parameter
        adv_patch = adv_patch.unsqueeze(0).to(self.model.device)

        obs_dict = batch['obs']
        perturbed_view = obs_dict[cfg.view]* (1 - mask) + adv_patch * mask
        # clamp the perturbed_obs to be within the range of the original image
        perturbed_view = torch.clamp(perturbed_view, 0, 1)
        obs_dict[cfg.view] = perturbed_view
        perturbed_obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))

        # nactions = self.normalizer['action'].normalize(batch['action'])
        actions = batch['action']
        self.model.reset()
        # get the predicted action for the perturbed observation
        with torch.no_grad():
            action_dist = self.action_dist(perturbed_obs_dict)
            action_means = action_dist.component_distribution.base_dist.loc
        predicted_action_dist = self.action_dist(perturbed_obs_dict)
        predicted_means = predicted_action_dist.component_distribution.base_dist.loc
        # calculate the loss
        loss = nn.MSELoss()(predicted_means, action_means)
        # since we are doing a targeted attack, we want to minimize the loss
        loss.backward()
        # perturb the observation with the gradient according to FGSM
        grad = torch.sign(perturbed_obs_dict[cfg.view].grad)
        grad = grad * mask
        grad = torch.sum(grad, dim=0)
        adv_patch = adv_patch + cfg.eps_iter * grad
        # clip the adversarial patch to be within cfg.eps
        adv_patch = torch.clamp(adv_patch, -cfg.eps, cfg.eps).squeeze(0)
        return adv_patch, loss.item()


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg.shape_meta

    policy = RobomimicImagePolicy(shape_meta=shape_meta)

