import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.env_runner.robomimic_image_runner import AdversarialRobomimicImageRunner
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path, instantiate
from hydra.core.global_hydra import GlobalHydra


torch.backends.cudnn.enabled = False
@hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_single_pgd')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_single_adversarial')
def main(cfg):
    checkpoint = cfg.checkpoint
    task = cfg.task
    attack = cfg.attack
    algo = cfg.algo
    device = cfg.device
    dataset_path = cfg.dataset_path
    view = cfg.view

    if cfg.log:
        wandb.init(project="diffusion_experimentation")
        wandb.log({"task": task, "attack": attack, "epsilon": cfg.epsilon, \
            "eps_iter": cfg.eps_iter})
        table = wandb.Table(columns=["Perturbations"])
        print("Perturbations: ", cfg.perturbations)
        for pertubation in cfg.perturbations:
            table.add_data(pertubation)


    # the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
    output_dir = os.path.join(os.getcwd(), f"diffusion_policy/data/experiments/image/{task}/{algo}/eval_single")
    if os.path.exists(output_dir):
        raise ValueError(f"Output path {output_dir} already exists!")

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg_loaded = payload['cfg']
    cfg.action_space = cfg_loaded.shape_meta.action.shape

    cls = hydra.utils.get_class(cfg_loaded._target_)
    workspace = cls(cfg_loaded, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model


    cfg_loaded.task.env_runner['dataset_path'] = str(dataset_path)

    try:
        if cfg_loaded.training.use_ema:
            policy = workspace.ema_model
    except:
        pass

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    cfg_loaded.task.env_runner['_target_'] = cfg._target_
    if cfg.max_steps is not None:
        cfg_loaded.task.env_runner['max_steps'] = cfg.max_steps
    env_runner = hydra.utils.instantiate(
        cfg_loaded.task.env_runner,
        output_dir=output_dir)
    # if attack:
    #     runner_log = env_runner.run(policy, cfg.epsilon, cfg)
    # else:
    #     runner_log = env_runner.run(policy)
    # env_runner.probability_of_action(policy, cfg)
    # env_runner.create_videos(policy, cfg, perturbation = 0.6)
    # env_runner.apply_patch_attack_single_image(policy, cfg)
    # env_runner.attack_single_image(policy, cfg)
    # env_runner.run_lstm_gmm_pgd(policy, cfg)
    # env_runner.create_trajectory_evolution(policy, cfg)
    print("Running DP with attack")
    env_runner.run_dp_with_attack(policy, cfg)
    # env_runner.apply_patch_attack_lstm(policy, cfg)

    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = value
    # print("Test/mean_score: ", json_log["test/mean_score"])

if __name__ == '__main__':
    main()