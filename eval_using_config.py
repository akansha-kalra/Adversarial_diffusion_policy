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
import pickle

torch.backends.cudnn.enabled = True


def get_run_name(checkpoint, cfg, attack, view):
    if attack:
        return f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-' \
               f'{checkpoint.split("/")[-3]}-{cfg.attack_type}_adversarial_on_{view}_tar_{cfg.targeted}'
    else:
        return f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-{checkpoint.split("/")[-3]}'
    # return f'bet_pgd_perturbation_gradview_check_{cfg.perturbations}'


def init_wandb(checkpoint, cfg, attack, view):
    run_name = get_run_name(checkpoint, cfg, attack, view)

    # Try to find an existing run with a similar name
    api = wandb.Api()
    # project = "grad_check_adv"
    project = "Adv_diffusion_policy"
    # project = "transferability_adv"
    # project = "vanilla_bc_image_policy"
    runs = api.runs(f"akanshakalracmu/{project}")

    existing_run = None
    for run in runs:
        if run.name == run_name:
            existing_run = run
            break

    # if existing_run:
    #     # If a run with the same name exists, resume it
    #     return wandb.init(project=project, id=existing_run.id, resume='must')
    # else:
    #     # If no matching run exists, create a new one
    return wandb.init(project=project, name=run_name)


# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_pgd_adversarial')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_pgd_adversarial')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_adversarial')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_ph_pick_adversarial_patch')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='ibc_image_ph_pick_adversarial_patch.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='vanilla_bc_ph_pick_adversarial_patch.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='bet_image_ph_pick_pgd_adversarial_patch.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='ibc_image_ph_pick_adversarial.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='vanilla_bc_image_ph_pick_pgd_adversarial.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='ibc_image_ph_pick_adversarial.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='bet_image_ph_pick_adversarial.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='ibc_image_ph_pick_pgd_adversarial.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_pusht.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='lstm_gmm_image_pusht.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_pgd_adversarial_Square_PH.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_fgsm_adversarial_Square_PH.yaml')
# @hydra.main(config_path='diffusion_policy/eval_configs', config_name='diffusion_policy_image_ph_pick_fgsm_adversarial_Square_PH.yaml')
@hydra.main(config_path='diffusion_policy/eval_configs', config_name='Transferability.yaml')
def main(cfg):
    checkpoint = cfg.checkpoint
    task = cfg.task
    algo = cfg.algo
    n_envs = cfg.n_envs
    device = cfg.device
    attack = cfg.attack
    epsilon = cfg.epsilon
    dataset_path = cfg.dataset_path
    print(f" Dataset path: {dataset_path}")
    view = cfg.view
    print(f"Running attack {attack} on {view} view")

    # the output directory should depend on the current directory and the checkpoint path and the attack type and epsilon
    output_dir = os.path.join(os.getcwd(),
                              f"data/experiments/image/{task}/{algo}/eval_{checkpoint.split('/')[-3]}_{epsilon}_{view}_attack_{attack}_{cfg.attack_type}_targeted_{cfg.targeted}")
    if cfg.attack_type=='patch':
        output_dir = os.path.join(os.getcwd(),
                                  f"data/experiments/image/{task}/{algo}/eval_{checkpoint.split('/')[-3]}_{epsilon}_{view}_attack_{attack}_{cfg.attack_type}_targeted_{cfg.targeted}_patch_path{cfg.patch_path}")
    else:
        output_dir = os.path.join(os.getcwd(),
                                  f"data/experiments/image/{task}/{algo}/eval_{checkpoint.split('/')[-3]}_{epsilon}_{view}_attack_{attack}_{cfg.attack_type}_targeted_{cfg.targeted}")

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
    try:
        policy = workspace.model
    except AttributeError:
        policy = workspace.policy

    if attack:
        print("Running adversarial Attack")
        cfg_loaded.task.env_runner['_target_'] = cfg._target_
        cfg_loaded.task.env_runner['n_envs'] = n_envs
        if cfg.max_steps is not None:
            cfg_loaded.task.env_runner['max_steps'] = cfg.max_steps
    if cfg.n_test > 0:
        cfg_loaded.task.env_runner['n_test'] = cfg.n_test
    if cfg.n_train > 0:
        cfg_loaded.task.env_runner['n_train'] = cfg.n_train
    try:
        cfg_loaded.task.env_runner['dataset_path'] = str(dataset_path)
    except:
        print("No dataset path provided")
        pass

    try:
        if cfg_loaded.training.use_ema:
            policy = workspace.ema_model
    except:
        pass

    if cfg.log:
        # wandb.init(project='BC_Evaluation', name=f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-\
        #     {checkpoint.split("/")[-3]}-{cfg.attack_type}_adversarial_on_{view}_randtar_{cfg.rand_target}' if attack else
        #   f'{checkpoint.split("/")[-6]}-{checkpoint.split("/")[-5]}-{checkpoint.split("/")[-4]}-{checkpoint.split("/")[-3]}')
        # wandb.init(project='BC_Evaluation', id='kfn7tfal', resume='must')
        # wandb.init(project='ibc_pgd_experimentation', name=f'epsilon-{cfg.epsilons[0]}-rand_target-{cfg.rand_target}-rand_init-{cfg.rand_int}')
        # wandb.init(project='ibc_pgd_experimentation', name=f'epsilon-{cfg.epsilons[0]}-target_perturbations-{cfg.target_perturbations}-pertubation-{cfg.perturbations[1]}')
        # wandb.init(project="BC_Evaluation", id='n3nvemg8', resume='must')
        # wandb.init(project='diffusion_experimentation', name=f'diffusion_policy_norm_monitoring')
        # wandb.init(project='adv_patch_test', name=f'lstm_gmm_{checkpoint.split("/")[-3]}_{cfg.patch_type}_patch')
        # wandb.init(project='adv_patch_test', name=f'vanilla_bc_{checkpoint.split("/")[-3]}_{cfg.patch_type}_patch')

        wandb_run = init_wandb(checkpoint, cfg, attack, view)
        print("Wandb run initialized", wandb_run)
        config_path = 'diffusion_policy/eval_configs'
        config_name="Transferability.yaml"
        # config_name = 'diffusion_policy_image_ph_pick_fgsm_adversarial_Square_PH'
        # config_name = 'diffusion_policy_image_ph_pick_pgd_adversarial_Square_PH'
        # config_name = 'vanilla_bc_ph_pick_adversarial_patch'
        # config_name = 'ibc_image_ph_pick_adversarial'
        # config_name = 'lstm_gmm_image_ph_pick_adversarial'
        # config_name = 'bet_image_ph_pick_adversarial'
        # config_name = 'ibc_image_ph_pick_pgd_adversarial'
        # config_name = 'vanilla_bc_image_ph_pick_pgd_adversarial'
        # wandb.log({"xloc": cfg.x_loc, "yloc": cfg.y_loc, "patch_size": cfg.patch_size})
        # config_name = 'lstm_gmm_image_ph_pick_pgd_adversarial'
        config_file_path = to_absolute_path(f"{config_path}/{config_name}.yaml")
        # save the config file to wandb from the hydras config
        wandb.save(config_file_path)

    device = torch.device(device)
    policy.to(device)
    policy.eval()
    env_runner = hydra.utils.instantiate(
        cfg_loaded.task.env_runner,
        output_dir=output_dir)
    if attack and cfg.attack_type == 'patch':
        patch = pickle.load(open(cfg.patch_path, 'rb'))
        # print("Shape of the patch: ", patch.shape)
        # patch[0, :] = torch.ones_like(patch[0, :])
        # patch[0, 0] = torch.ones_like(patch[0, 0])
        # patch[0, 1] = torch.ones_like(patch[0, 1])
        # print(patch[0])
        runner_log = env_runner.run(policy, adversarial_patch=patch, cfg=cfg)
    elif attack:
        runner_log = env_runner.run(policy, epsilon=cfg.epsilon, cfg=cfg)
    else:
        runner_log = env_runner.run(policy, cfg=cfg)
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    if cfg.log:
        if cfg.attack_type=="patch":
            wandb.log({"test/mean_score": json_log["test/mean_score"], "train/mean_score": json_log["train/mean_score" ], \
                   "Epsilon": float(cfg.epsilon),
        "patch_path": str(cfg.patch_path)})
        else :
            wandb.log({"test/mean_score": json_log["test/mean_score"], "train/mean_score": json_log["train/mean_score"], \
                       "Epsilon": float(cfg.epsilon)})
    print("Test/mean_score: ", json_log["test/mean_score"])
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    wandb.finish()


if __name__ == '__main__':
    main()