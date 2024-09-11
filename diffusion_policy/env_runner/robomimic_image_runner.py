import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.utils.attack_utils import optimize_linear, clip_perturb

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ArtistAnimation
from pathlib import Path
import random
# for deterministic results
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # or ':16:8'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        try:
            env_meta = FileUtils.get_env_metadata_from_dataset(
                dataset_path)
        except FileNotFoundError:
            # append the relative path to the dataset path
            directory_path = Path(__file__).parent.parent.parent
            dataset_path = os.path.join(directory_path, dataset_path)
            print(f"New dataset path: {dataset_path}")
            env_meta = FileUtils.get_env_metadata_from_dataset(
                dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy, adversarial_patch=None, cfg=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        torch.use_deterministic_algorithms(True)
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        observations = []
        # randomly select envs to visualize from the n_envs
        vis_envs = np.random.choice(n_envs, cfg.n_vis, replace=False)
        try:
            obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        except:
            obs_encoder = policy.obs_encoder
        image_encoder = obs_encoder.obs_nets['robot0_eye_in_hand_image']
        policy.eval()
        # set_seed(1024)

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            if cfg.view == 'both':
                views = ['agentview_image', 'robot0_eye_in_hand_image']
            elif cfg.view == 'None':
                views = []
            else:
                views = [cfg.view]
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # print(f"Min and Max of obs_dict: {torch.min(obs_dict['robot0_eye_in_hand_image'])}, {torch.max(obs_dict['robot0_eye_in_hand_image'])}")
                clean_obs_dict = obs_dict.copy()
                random_obs_dict = obs_dict.copy()
                if adversarial_patch is not None:
                    for view in views:
                        obs_dict[view] = obs_dict[view] + adversarial_patch[view].to(device)
                        obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
                    # maximum_perturbation = torch.max(torch.abs(adversarial_patch))
                    # random_perturbation = torch.rand_like(obs_dict[cfg.view]) * 2 * maximum_perturbation - maximum_perturbation
                    # random_obs_dict[cfg.view] = random_obs_dict[cfg.view] + random_perturbation
                obs_dict = dict_apply(obs_dict, lambda x: x.to(device=device))
                # run policy
                with torch.no_grad():
                    # calculate the l2 distance clean and obs_dict
                    clean_action_dict = policy.predict_action(clean_obs_dict)
                    action_dict = policy.predict_action(obs_dict)
                    # clean_output = policy.predict_action(clean_obs_dict, return_latent=True)['output']
                    # output = policy.predict_action(obs_dict, return_latent=True)['output']
                    # latents = policy.predict_action(obs_dict, return_latent=True)
                    # clean_latents = policy.predict_action(clean_obs_dict, return_latent=True)
                    # # print("L2 distance between clean and adversarial output: ", torch.norm(clean_output - output, p=2) / output.shape[0])
                    # print(latents['latent'])
                    # print(clean_latents['latent'])
                    # print("L2 distance between clean and adversarial latents: ", torch.norm(clean_latents['latent'][0].float() - latents['latent'][0].float(), p=2) / latents['latent'][0].shape[0])
                    # random_action_dict = policy.predict_action(random_obs_dict)
                observations.append(obs_dict['robot0_eye_in_hand_image'][vis_envs].detach().cpu().numpy())
                # print(f"observations appending has shape {observations[0].shape}")
                # device_transfer
                try:
                    np_action_dict = dict_apply(action_dict,
                        lambda x: x.detach().to('cpu').numpy())
                    # np_clean_action_dict = dict_apply(clean_action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    # np_random_action_dict = dict_apply(random_action_dict,
                    #     lambda x: x.detach().to('cpu').numpy())
                    action = np_action_dict['action']
                    # clean_actions = np_clean_action_dict['action']
                    # random_actions = np_random_action_dict['action']
                except AttributeError:
                    action = action_dict['action'].detach().to('cpu').numpy()
                # l1 distance between actions and clean actions before perturbation
                # for i in range(action.shape[-1]):
                #     print(f"l1_distance_{i}: {torch.norm(action_dict['action'][:,:,i] - clean_action_dict['action'][:,:,i], p=1) / action.shape[0]}")
                if cfg.targeted and cfg.log:
                    try:
                    # log the l1 distance between the predicted action and the target action per environment
                        target_action = clean_action_dict['action'].to('cpu') + adversarial_patch['perturbations'].to('cpu')
                        for i in range(target_action.shape[-1]):
                            wandb.log({f"l1_distance_{i}": torch.norm(action_dict['action'][:,:,i].to('cpu') - target_action[:,:,i], p=1) / target_action.shape[0]})
                    except KeyError:
                        pass
                # features = np_action_dict['features']
                # clean_features = np_clean_action_dict['features']
                # random_features = np_random_action_dict['features']
                # features = np.mean(features, axis=0)
                # actions = np.mean(action, axis=0)
                # clean_actions = np.mean(clean_actions, axis=0)
                # clean_features = np.mean(clean_features, axis=0)
                # random_features = np.mean(random_features, axis=0)
                # random_actions = np.mean(random_actions, axis=0)
                # l2 distance between features
                # diff = np.linalg.norm(features - clean_features)
                # diff_actions = np.linalg.norm(actions - clean_actions)
                # diff_random = np.linalg.norm(random_features - clean_features)
                # diff_random_actions = np.linalg.norm(random_actions - clean_actions)
                # print(f"Diff: {diff}, Random Diff: {diff_random}")
                # print(f"Diff Actions: {diff_actions}, Random Diff Actions: {diff_random_actions}")
                # if cfg.log:
                #     wandb.log({"Diff": diff, "Random Diff": diff_random, 
                #                 "Diff Actions": diff_actions, "Random Diff Actions": diff_random_actions})
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                 # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # visualize the video and save to wandb
        if cfg.save_video and cfg.log:
            for i in range(cfg.n_vis):
                ims = []
                fig, ax = plt.subplots()
                for j in range(len(observations)):
                    try:
                        print(observations[j][i][1].shape)
                        im = observations[j][i][1].transpose(1, 2, 0)
                    except IndexError:
                        im = observations[j][i][0].transpose(1, 2, 0)
                    ims.append([ax.imshow(im)])
                save_path = Path(cfg.checkpoint).parent

                save_path = save_path.joinpath(f'{cfg.exp_name}_vis_{i}.gif')
                print(f"Saving video to {save_path}")

                if os.path.exists(save_path):
                    os.remove(save_path)

                ani = ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
                ani.save(str(save_path), writer='pillow')
                plt.close(fig)
        #         
                wandb.log({f'vis_{i}': wandb.Image(str(save_path))})
            
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

        #     # visualize sim
        #     video_path = all_video_paths[i]
        #     if video_path is not None:
        #         sim_video = wandb.Video(video_path)
        #         log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

class AdversarialRobomimicImageRunner(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy,\
                    cfg, action):
        view = cfg.view
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max

        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")

        # check if the input tensors are within the clip range
        for view in views:
            assert torch.all(obs_dict[view] >= clip_min) and torch.all(obs_dict[view] <= clip_max), \
                "Input tensor is not within the clip range"
        
        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        predicted_action = policy.predict_action(obs_dict)['action']
        # print("ACtion dist: ", action_dist)
        # predicted_scales = predicted_action_dist.component_distribution.base_dist.scale
        # print(predicted_means, predicted_scales, means)
        # print(predicted_action.grad_fn, action.grad_fn)
        mse_loss = torch.nn.MSELoss()
        if cfg.targeted:
            action = action + torch.tensor(cfg.perturbations).to(policy.device)
            loss = mse_loss(predicted_action, action)
            loss = -loss
        else:
            # loss = mse_loss(predicted_action, action)
            # apply loss for the first two dimensions of the action
            loss = mse_loss(predicted_action[:,:, :2], action[:,:, :2])
        loss.backward()
        if cfg.log:
            wandb.log({"Loss": loss.item()})
            # log per dimension mse loss for predicted_action and action
            for i in range(7):
                wandb.log({f"Loss{i}d": mse_loss(predicted_action[:,:, i], action[:,:, i])})
        for view in views:
            # assert obs_dict_copy[view].grad_fn is not None, "Input tensor does not have a grad_fn"
            grad = torch.sign(obs_dict[view].grad)
            obs_dict[view] = obs_dict[view] + optimize_linear(grad, cfg.eps_iter, cfg.norm)
            obs_dict[view] = torch.clamp(obs_dict[view], clip_min, clip_max)
        # obs_dict['agentview_image'].requires_grad = False
        return obs_dict

    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, cfg, action):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        view = cfg.view
        n_iter = cfg.n_iter
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        eps_iter = cfg.eps_iter
        norm = cfg.norm
        rand_int = cfg.rand_int
        # check if the input tensors are within the clip range
        if view == 'both':
            # make a list of views in the rgb space
            views = ['agentview_image', 'robot0_eye_in_hand_image']
            assert torch.all(obs_dict[views[0]] >= clip_min) and torch.all(obs_dict[views[0]] <= clip_max), \
                "Input tensor is not within the clip range"
            assert torch.all(obs_dict[views[1]] >= clip_min) and torch.all(obs_dict[views[1]] <= clip_max), \
                "Input tensor is not within the clip range"
        else:
            views = list(str(view))
            assert torch.all(obs_dict[view] >= clip_min) and torch.all(obs_dict[view] <= clip_max), \
                "Input tensor is not within the clip range"
        if rand_int:
            # randomly initialize the perturbation from a uniform distribution within the epsilon bound
            for view in views:
                perturbation = torch.FloatTensor(obs_dict[view].shape).uniform_(-self.epsilon, self.epsilon).to(policy.device)
                perturbation = torch.clamp(perturbation, -eps_iter, eps_iter)
                perturbation = torch.clamp(obs_dict[view] + perturbation, clip_min, clip_max) - obs_dict[view]
                adv_obs_dict[view] = obs_dict[view] + perturbation
        adv_obs_dict = obs_dict.copy()
        for i in range(n_iter):
            policy.zero_grad()
            adv_obs_dict = self.apply_fgsm_attack(adv_obs_dict, policy, cfg, action)
            for view in views:
                perturbation = adv_obs_dict[view] - obs_dict[view]
                if norm == 'l2':
                    perturbation = perturbation * self.epsilon / torch.norm(perturbation, p=2)
                elif norm == 'linf':
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_obs_dict[view] = obs_dict[view] + perturbation
                adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
        return adv_obs_dict

    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        attack_type = cfg.attack_type
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # device transfer and enable gradients
                # obs_dict = dict_apply(np_obs_dict,
                #    lambda x: torch.from_numpy(x).to(device=device).requires_grad_(True))
                prev_obs_dict = obs_dict
                # apply attack
                with torch.no_grad():
                    action = policy.predict_action(obs_dict)['action']
                if attack_type == 'fgsm':
                    obs_dict = self.apply_fgsm_attack(obs_dict, policy, \
                                            cfg, action)
                    for view in views:
                        perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                        perturbation = perturbation.view(perturbation.shape[0], -1)
                        # log the maximum perturbation across the batch
                        perturbation_max = torch.sum(perturbation, dim=1)
                elif attack_type == 'pgd':
                    obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg, action)
                elif attack_type == 'noise':
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                elif attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                # if cfg.log == True:
                    # log the perturbation for each environment separately
                    # for i in range(len(perturbation_norm)):
                    #     wandb.log({f'perturbation_{view}_{i}': perturbation_norm[i]})
                    # for i in range(len(perturbation_max)):
                    #     wandb.log({f'max_perturbation_{view}_{i}': perturbation_max[i]})
                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

class AdversarialRobomimicImageRunnerLSTM(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy,\
                    cfg, target_means=None):
        view = cfg.view
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max

        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")

        # check if the input tensors are within the clip range
        for view in views:
            assert torch.all(obs_dict[view] >= clip_min) and torch.all(obs_dict[view] <= clip_max), \
                "Input tensor is not within the clip range"
        
        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        # create a model prediction as ground truth
        with torch.no_grad():
            # action_dict = policy.predict_action(obs_dict)
            # action = action_dict['action']
            action_dist = policy.action_dist(obs_dict)
            means = action_dist.component_distribution.base_dist.loc

        # predicted_action = policy.predict_action(obs_dict)['action']
        predicted_action_dist= policy.action_dist(obs_dict)
        # print("ACtion dist: ", action_dist)
        predicted_means = predicted_action_dist.component_distribution.base_dist.loc
        # predicted_scales = predicted_action_dist.component_distribution.base_dist.scale
        # print(predicted_means, predicted_scales, means)
        # print(predicted_action.grad_fn, action.grad_fn)
        mse_loss = torch.nn.MSELoss()
        if not cfg.targeted:
            loss = mse_loss(predicted_means, target_means)
            loss.backward()
        else:
            loss = mse_loss(predicted_means, target_means)
            loss = -loss
            loss.backward()
        if cfg.log:
            wandb.log({"Loss": loss.item()})
        for view in views:
            # assert obs_dict_copy[view].grad_fn is not None, "Input tensor does not have a grad_fn"
            grad = torch.sign(obs_dict[view].grad)
            if target_means is None:
                obs_dict[view] = obs_dict[view] + optimize_linear(grad, self.epsilon, cfg.norm)
            else:
                obs_dict[view] = obs_dict[view] + optimize_linear(grad, cfg.eps_iter, cfg.norm)
            obs_dict[view] = torch.clamp(obs_dict[view], clip_min, clip_max)
        # obs_dict['agentview_image'].requires_grad = False
        return obs_dict

    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        view = cfg.view
        n_iter = cfg.n_iter
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        eps_iter = cfg.eps_iter
        norm = cfg.norm
        rand_int = cfg.rand_int
        # check if the input tensors are within the clip range
        if view == 'both':
            # make a list of views in the rgb space
            views = ['agentview_image', 'robot0_eye_in_hand_image']
            assert torch.all(obs_dict[views[0]] >= clip_min) and torch.all(obs_dict[views[0]] <= clip_max), \
                "Input tensor is not within the clip range"
            assert torch.all(obs_dict[views[1]] >= clip_min) and torch.all(obs_dict[views[1]] <= clip_max), \
                "Input tensor is not within the clip range"
        else:
            views = list(str(view))
            assert torch.all(obs_dict[view] >= clip_min) and torch.all(obs_dict[view] <= clip_max), \
                "Input tensor is not within the clip range"
        if rand_int:
            # randomly initialize the perturbation from a uniform distribution within the epsilon bound
            for view in views:
                perturbation = torch.FloatTensor(obs_dict[view].shape).uniform_(-self.epsilon, self.epsilon).to(policy.device)
                perturbation = torch.clamp(perturbation, -eps_iter, eps_iter)
                perturbation = torch.clamp(obs_dict[view] + perturbation, clip_min, clip_max) - obs_dict[view]
                adv_obs_dict[view] = obs_dict[view] + perturbation
        with torch.no_grad():
            clean_action_dist = policy.action_dist(obs_dict)
            clean_action_means = clean_action_dist.component_distribution.base_dist.loc
            if cfg.targeted:
                target_means = clean_action_means.clone().detach() + torch.tensor(cfg.perturbations).unsqueeze(0).to(policy.device)
            else:
                target_means = clean_action_means
        adv_obs_dict = obs_dict.copy()
        for i in range(n_iter):
            policy.zero_grad()
            adv_obs_dict = self.apply_fgsm_attack(adv_obs_dict, policy, cfg, target_means)
            for view in views:
                perturbation = adv_obs_dict[view] - obs_dict[view]
                if norm == 'l2':
                    perturbation = perturbation * self.epsilon / torch.norm(perturbation, p=2)
                elif norm == 'linf':
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_obs_dict[view] = obs_dict[view] + perturbation
                adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
        return adv_obs_dict

    def apply_noise_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Applies noise attack to the input image.
        A basic type of attack to serve as a baseline.
        """
        view = cfg.view
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        noise_bound = cfg.noise_bound
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        for view in views:
            noise = torch.FloatTensor(obs_dict[view].shape).uniform_(-noise_bound, noise_bound).to(obs_dict[view].device)
            obs_dict[view] = obs_dict[view] + noise
            obs_dict[view] = torch.clamp(obs_dict[view], clip_min, clip_max)
        return obs_dict


    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        attack_type = cfg.attack_type
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # device transfer and enable gradients
                # obs_dict = dict_apply(np_obs_dict,
                #    lambda x: torch.from_numpy(x).to(device=device).requires_grad_(True))
                prev_obs_dict = obs_dict
                with torch.no_grad():
                    clean_action_dist = policy.action_dist(obs_dict)
                    clean_action_means = clean_action_dist.component_distribution.base_dist.loc
                    target_means = clean_action_means

                # apply attack
                if attack_type == 'fgsm':
                    obs_dict = self.apply_fgsm_attack(obs_dict, policy, \
                                            cfg, target_means)
                    for view in views:
                        perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                        perturbation = perturbation.view(perturbation.shape[0], -1)
                        # log the maximum perturbation across the batch
                        perturbation_max = torch.sum(perturbation, dim=1)
                elif attack_type == 'pgd':
                    obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg)
                elif attack_type == 'noise':
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                elif attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")
                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                # if cfg.log == True:
                    # log the perturbation for each environment separately
                    # for i in range(len(perturbation_norm)):
                    #     wandb.log({f'perturbation_{view}_{i}': perturbation_norm[i]})
                    # for i in range(len(perturbation_max)):
                    #     wandb.log({f'max_perturbation_{view}_{i}': perturbation_max[i]})
                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


class AdversarialRobomimicImageRunnerIBC(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lipschitz_consts = []
        self.episodes_lipschitz_consts = []
        self.loss_per_iteration = []
        self.loss_per_episode = []
        self.perturbation_per_episode = []

    def apply_fgsm_attack_loss_of_ibc(self, obs_dict_cp, policy:BaseImagePolicy, cfg, actions = None, \
                                        action_samples=None, clean_actions=None):
        # use the same loss as the policy
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")

        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict_copy = dict_apply(obs_dict_cp, lambda x: x.clone().detach().requires_grad_(True))
         # retain the grad on the non-leaf tensor, i.e. the input tensor
        # for view in views:
        #     obs_dict_copy[view].retain_grad()
        policy.zero_grad()

        assert actions is not None, "Actions must be provided for IBC loss"
        # create a model prediction as ground truth
        # batch = {}
        # batch['obs'] = obs_dict_copy
        # batch['action'] = actions
        if cfg.targeted:
            if action_samples is not None:
                loss, obs_dict_copy, obs_features = policy.compute_loss_with_grad(obs_dict_copy, actions, action_samples)
            else:
                loss, obs_dict_copy, obs_features = policy.compute_loss_with_grad(obs_dict_copy, actions)
        else:
            actions = policy.predict_action(obs_dict_copy, return_energy=True)
            action = actions['action']
            energy = actions['energy']
            loss = -torch.max(energy, dim=1)[0]
            loss = loss.sum()

        # print(loss.item())
        self.loss_per_iteration.append(loss.item())
        # for view in views:
        #     print("Obs dict copy is leaf? ", obs_dict_copy[view].is_leaf)
        # print("Loss: ", loss)
        if cfg.rand_target or cfg.targeted:
             loss = -loss
        if cfg.log:
            wandb.log({"FGSM_Loss": loss.item()})
        loss.backward()
        # print("Observation features and its grad ", obs_features, obs_features.grad)
        prev_obs_dict = obs_dict_cp
        for view in views:
            # check if the grad is not None
            assert obs_dict_copy[view].grad is not None, "Input tensor does not have a grad"
            # calculate the average value of the grad and store in wandb
            grad = torch.sign(obs_dict_copy[view].grad)
            # if obs_dict_copy[view].grad is not None:
            #     obs_dict_copy[view].grad.zero_()  # Clear gradients for obs_dict to save memory
            # log the lipschitz constant for the input tensor
            lip_grad = obs_dict_copy[view].grad.view(obs_dict_copy[view].shape[0], -1)
            # calculate seperate lipschitz constant for each env in the batch
            lip_const = torch.norm(lip_grad, p=2, dim=1)
            if cfg.log:
                wandb.log({"Lipschitz": lip_const})
            #  for i in range(len(lip_grad)):
            #     if cfg.log:
            #         wandb.log({f'Lipschitz_{view}_{i}': lip_const[i]})
            # self.lipschitz_consts.append((view, lip_const))
            # print("Grad: ", grad)
            # if cfg.attack_type != 'pgd':
            #     if cfg.rand_target:
            #         obs_dict_copy[view] = obs_dict_copy[view] - self.epsilon * grad
            #     else:
            #         obs_dict_copy[view] = obs_dict_copy[view] + self.epsilon * grad
            # elif cfg.attack_type == 'pgd':
            #     if cfg.rand_target:
            #         obs_dict_copy[view] = obs_dict_copy[view] - cfg.eps_iter * grad
            #     else:
            #         obs_dict_copy[view] = obs_dict_copy[view] + cfg.eps_iter * grad
            # else:
            #     raise ValueError("Only PGD and FGSM attacks are supported")

            if cfg.attack_type == 'pgd':
                obs_dict_copy[view] = obs_dict_copy[view] + optimize_linear(grad, cfg.eps_iter, cfg.norm)
            elif cfg.attack_type == 'fgsm' :
                obs_dict_copy[view] = obs_dict_copy[view] + optimize_linear(grad, self.epsilon, cfg.norm)
            else:
                raise ValueError("Only PGD and FGSM attacks are supported")
            old_obs_dict = obs_dict_copy[view]
            # print("Grad: ", grad)
            if (cfg.clip_min is not None) or (cfg.clip_max is not None):
                if cfg.clip_min is None or cfg.clip_max is None:
                    raise ValueError(
                        "One of clip_min and clip_max is None but we don't currently support one-sided clipping"
                    )
                obs_dict_copy[view] = torch.clamp(obs_dict_copy[view], cfg.clip_min, cfg.clip_max)
                # if torch.any(old_obs_dict != obs_dict_copy[view]):
                    # print("Clipping applied")
            # diff = abs(obs_dict_copy[view] - prev_obs_dict[view])
            # diff = diff.view(diff.shape[0], -1)
            # grad = grad.view(grad.shape[0], -1)
            # log the maximum perturbation across the batch
            # diff_max = torch.max(diff, dim=1)
            # grad_max = torch.max(grad, dim=1)
            # diff_sum = torch.sum(diff, dim=1)
            # if cfg.log:
            #    wandb.log({"Sum_Perturbation": diff_sum})
        return obs_dict_copy


    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        # create a model prediction as ground truth
        with torch.no_grad():
            # action_dict = policy.predict_action(obs_dict)
            # action = action_dict['action']
            action_dist = policy.action_dist(obs_dict)
            # print("action_dist: ", action_dist.grad_fn)
            # print("The shape of action dist: ", action_dist.shape)
            # means = action_dist.component_distribution.base_dist.loc
        
        # predicted_action = policy.predict_action(obs_dict)['action']
        predicted_action_dist= policy.action_dist(obs_dict)
        # print("predicted_action_dist: ", predicted_action_dist)
        # print("ACtion dist: ", action_dist)
        # predicted_means = predicted_action_dist.component_distribution.base_dist.loc
        # predicted_scales = predicted_action_dist.component_distribution.base_dist.scale
        # print(predicted_means, predicted_scales, means)
        # print(predicted_action.grad_fn, action.grad_fn)
        cross_entropy_loss = torch.nn.CrossEntropyLoss()
        loss = cross_entropy_loss(predicted_action_dist, action_dist)
        if loss == 0:
            return obs_dict
        loss.backward()
        for view in views:
            print("Grad: ", obs_dict[view].grad)
            # check if all the values of the grad are zeros
            assert torch.all(obs_dict[view].grad == 0), "Input tensor grad are all zeros"
            grad = torch.sign(obs_dict[view].grad)
            obs_dict[view] = obs_dict[view] + self.epsilon * grad
            obs_dict[view] = torch.clamp(obs_dict[view], 0, 1)
            if obs_dict[view].grad is not None:
                obs_dict[view].grad.zero_()  # Clear gradients for obs_dict
        torch.cuda.empty_cache()
        return obs_dict

    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, 
                            cfg, clean_actions = None):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        # self.lipschitz_consts = []
        self.loss_per_iteration = []
        self.pertubation_per_iteration = []
        view = cfg.view
        num_iter = cfg.num_iter
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        norm = cfg.norm
        rand_int = cfg.rand_int
        adv_obs_dict = obs_dict
        B = obs_dict['agentview_image'].shape[0]
        T_neg = policy.train_n_neg
        T_a = policy.n_action_steps
        naction_stats = policy.get_naction_stats()
        action_samples = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        ).sample((B, T_neg, T_a)).to(device=obs_dict['agentview_image'].device)
        # check if the input tensors are within the clip range
        if view == 'both':
            # make a list of views in the rgb space
            views = ['agentview_image', 'robot0_eye_in_hand_image']
            assert torch.all(obs_dict[views[0]] >= clip_min) and torch.all(obs_dict[views[0]] <= clip_max), \
                "Input tensor is not within the clip range"
            assert torch.all(obs_dict[views[1]] >= clip_min) and torch.all(obs_dict[views[1]] <= clip_max), \
                "Input tensor is not within the clip range"
        else:
            views = list(str(view))
            assert torch.all(obs_dict[view] >= clip_min) and torch.all(obs_dict[view] <= clip_max), \
                "Input tensor is not within the clip range"
        # if rand_int:
        #     # randomly initialize the perturbation from a uniform distribution within the epsilon bound
        #     for view in views:
        #         perturbation = torch.FloatTensor(obs_dict[view].shape).uniform_(-self.epsilon, self.epsilon).to(obs_dict[view].device)
        #         perturbation = torch.clamp(perturbation, -cfg.eps_iter, cfg.eps_iter)
        #         perturbation = torch.clamp(obs_dict[view] + perturbation, clip_min, clip_max) - obs_dict[view]
        #         obs_dict[view] = obs_dict[view] + perturbation
        if cfg.rand_target:
            target_actions = torch.FloatTensor(obs_dict['agentview_image'].shape[0], \
                        self.n_action_steps, cfg.action_space[0]).uniform_(0, 1).to(obs_dict['agentview_image'].device)
        else:
            target_actions = policy.predict_action(obs_dict)['action']
            if clean_actions is None:
                clean_actions = target_actions
            if cfg.targeted:
                # print("Target Action before Perturbation: ", target_actions)
                target_actions = target_actions + torch.tensor(cfg.perturbations).to(obs_dict['agentview_image'].device)
            else:
                target_actions = target_actions
                # print("Target Action after Perturbation: ", target_actions)
        # add a dummy action at the beginning to get the shape right while
        # computing loss but it won't be used in the loss computation
        action_samples = torch.cat([target_actions.unsqueeze(1), action_samples], dim=1)
        # replace one of the action with the clean action
        if cfg.targeted:
            action_samples[:, 1] = clean_actions
        target_actions = torch.cat([torch.zeros_like(target_actions[:,0:1]), target_actions], dim=1)
        # print("First obs dict: ", adv_obs_dict['agentview_image'])
        for i in range(num_iter):
            policy.zero_grad()
            adv_obs_dict = self.apply_fgsm_attack_loss_of_ibc(adv_obs_dict, policy, cfg, target_actions, action_samples, clean_actions)
            for view in views:
                perturbation = adv_obs_dict[view] - obs_dict[view]
                if norm == 'l2':
                    perturbation = perturbation * self.epsilon / torch.norm(perturbation, p=2)
                elif norm == 'linf':
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_obs_dict[view] = obs_dict[view] + perturbation
                adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
                if cfg.log:
                    wandb.log({f"L2_norm_{view}": torch.norm(perturbation, p=2)})
                # if view == 'agentview_image':
                    # perturbation_wrt_original = torch.sum(abs(adv_obs_dict[view].view(adv_obs_dict[view].shape[0], -1).float() - obs_dict[view].view(obs_dict[view].shape[0], -1).float()), dim=1)
                    # print("Perturbation wrt original: ", perturbation_wrt_original)
                    # self.pertubation_per_iteration.append(perturbation_wrt_original)
            # print mean perturbation
            # print("Mean perturbation: ", torch.mean(perturbation))
        return adv_obs_dict

    def apply_noise_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Applies noise attack to the input image.
        A basic type of attack to serve as a baseline.
        """
        view = cfg.view
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        noise_bound = cfg.noise_bound
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        for view in views:
            noise = torch.FloatTensor(obs_dict[view].shape).uniform_(-noise_bound, noise_bound).to(obs_dict[view].device)
            obs_dict[view] = obs_dict[view] + noise
            obs_dict[view] = torch.clamp(obs_dict[view], clip_min, clip_max)
        return obs_dict


    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            average_perturbation = 0
            perturbed_obs_dicts = []
            timestep = 0
            # clean_obs_dicts = []

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                clean_obs_dict = obs_dict.copy()
                print(type(obs_dict), obs_dict['agentview_image'].shape)

                # clean_obs_dicts.append(obs_dict)
                if cfg.rand_target:
                    target_actions = torch.FloatTensor(obs_dict['agentview_image'].shape[0], \
                                self.n_action_steps, cfg.action_space[0]).uniform_(0, 1).to(obs_dict['agentview_image'].device)
                else:
                    target_actions = policy.predict_action(np_obs_dict)['action']
                # add a dummy action at the beginning to get the shape right while
                # computing loss but it won't be used in the loss computation
                target_actions = torch.cat([torch.zeros_like(target_actions[:,0:1]), target_actions], dim=1)
                # print("Target actions: ", target_actions)

                # device transfer and enable gradients
                # obs_dict = dict_apply(np_obs_dict,
                #    lambda x: torch.from_numpy(x).to(device=device).requires_grad_(True))
                # apply attack
                if cfg.attack_type == 'fgsm':
                    prev_obs_dict = obs_dict
                    obs_dict = self.apply_fgsm_attack(obs_dict, policy, cfg)
                    # log the l2 norm of the perturbation for fgsm attack
                    for view in views:
                        perturbation = obs_dict[view] - prev_obs_dict[view]
                        perturbation = perturbation.view(perturbation.shape[0], -1)
                        perturbation_norm = torch.norm(perturbation, p=2, dim=1)
                elif cfg.attack_type == 'fgsm_alt':
                    prev_obs_dict = obs_dict
                    obs_dict = self.apply_fgsm_attack_loss_of_ibc(obs_dict, policy, cfg, target_actions)
                    # log the l2 norm of the perturbation for fgsm attack
                    for view in views:
                        perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                        perturbation = perturbation.view(perturbation.shape[0], -1)
                        # log the maximum perturbation across the batch
                        perturbation_max = torch.max(perturbation, dim=1)
                        perturbation_sum = torch.sum(perturbation, dim=1)
                elif cfg.attack_type == 'pgd':
                    prev_obs_dict = obs_dict
                    clean_action = policy.predict_action(obs_dict, return_energy=True)
                    clean_actions = clean_action['action']
                    clean_energy = clean_action['energy']
                    for _ in range(cfg.num_epochs):
                        obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg, clean_actions=clean_actions)
                    # self.episodes_lipschitz_consts.append(self.lipschitz_consts)
                    # self.loss_per_episode.append(self.loss_per_iteration)
                    # self.perturbation_per_episode.append(self.pertubation_per_iteration)
                    # log the l2 norm of the perturbation for pgd attack
                    # for view in views:
                    #     perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                    #     perturbation = perturbation.view(perturbation.shape[0], -1)
                    #     perturbation_max = torch.max(perturbation, dim=1)
                    #     perturbation_sum = torch.sum(perturbation, dim=1)
                elif cfg.attack_type == 'noise':
                    prev_obs_dict = obs_dict
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                    # log the l2 norm of the perturbation for noise attack
                    for view in views:
                        perturbation = obs_dict[view] - prev_obs_dict[view]
                        perturbation = perturbation.view(perturbation.shape[0], -1)
                        perturbation_norm = torch.norm(perturbation, p=2, dim=1)
                        # take the mean of the perturbation norm
                elif cfg.attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")

                # average_perturbation += perturbation_max
                # if cfg.log == True:
                    # log the perturbation for each environment separately
                    # for i in range(len(perturbation_norm)):
                    #     wandb.log({f'perturbation_{view}_{i}': perturbation_norm[i]})
                    # get the energy of the clean and perturbed images
                    # energy_clean = policy.predict_action(prev_obs_dict, return_energy=True)
                    # energy_perturbed = policy.predict_action(obs_dict, return_energy=True)
                    # print("Energy clean: ", energy_clean['energy'])
                    # print("Shape, Max, min and median of clean energy: ", energy_clean['energy'].shape, torch.max(energy_clean['energy']), torch.min(energy_clean['energy']), torch.median(energy_clean['energy']))
                    # print("Energy perturbed: ", energy_perturbed['energy'])
                    # print("Shape, Max, min and median of perturbed energy: ", energy_clean['energy'].shape, torch.max(energy_perturbed['energy']), torch.min(energy_perturbed['energy']), torch.median(energy_perturbed['energy']))
                    # save the max, min, median, mean and std, and number of actions within max-5 
                    # for i in range(len(energy_clean['energy'])):
                    #     wandb.log({f'Energy_clean_max_{i}': torch.max(energy_clean['energy'][i]), \
                    #                 f'Energy_clean_min_{i}': torch.min(energy_clean['energy'][i]), \
                    #                 f'Energy_clean_median_{i}': torch.median(energy_clean['energy'][i]), \
                    #                 f'Energy_clean_mean_{i}': torch.mean(energy_clean['energy'][i]), \
                    #                 f'Energy_clean_std_{i}': torch.std(energy_clean['energy'][i]), \
                    #                 f'Energy_clean_max-5_{i}': torch.sum(energy_clean['energy'][i] > torch.max(energy_clean['energy'][i]) - 5), \
                    #                 f'Energy_perturbed_max_{i}': torch.max(energy_perturbed['energy'][i]), \
                    #                 f'Energy_perturbed_min_{i}': torch.min(energy_perturbed['energy'][i]), \
                    #                 f'Energy_perturbed_median_{i}': torch.median(energy_perturbed['energy'][i]), \
                    #                 f'Energy_perturbed_mean_{i}': torch.mean(energy_perturbed['energy'][i]), \
                    #                 f'Energy_perturbed_std_{i}': torch.std(energy_perturbed['energy'][i]), \
                    #                 f'Energy_perturbed_max-5_{i}': torch.sum(energy_perturbed['energy'][i] > torch.max(energy_perturbed['energy'][i]) - 5)})
                    # for i in range(len(perturbation_max)):
                    #     wandb.log({f'max_perturbation_{view}_{i}': perturbation_max[i], \
                    #                 f'sum_perturbation_{view}_{i}': perturbation_sum[i]})


                # run policy
                # random_noise_dict = clean_obs_dict.copy()
                # for view in views:
                #     random_noise = torch.FloatTensor(obs_dict[view].shape).uniform_(-self.epsilon, self.epsilon).to(obs_dict[view].device)
                #     random_noise_dict[view] = random_noise_dict[view] + random_noise
                #     random_noise_dict[view] = torch.clamp(random_noise_dict[view], 0, 1)
                energy_save_path = f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/"
                pickle.dump(obs_dict, open(f"{energy_save_path}ibc_perturbed_obs_dict_{self.epsilon}_timestep_{timestep}.pkl", "wb"))
                pickle.dump(clean_obs_dict, open(f"{energy_save_path}ibc_clean_obs_dict_{self.epsilon}_timestep_{timestep}.pkl", "wb"))
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict, return_energy=True)
                    clean_dict = policy.predict_action(clean_obs_dict, return_energy=True)
                    perturbed_energy = action_dict['energy'].detach().to('cpu').numpy()
                    clean_energy = clean_dict['energy'].detach().to('cpu').numpy()
                    # random_dict = policy.predict_action(random_noise_dict, return_energy=True)
                    # print(f"Clean energy {torch.max(clean_energy)}")
                    # print(f"Perturbed energy {torch.max(action_dict['energy'])}")
                    # print(f"Random energy {torch.max(random_dict['energy'])}")
                # perturbed_obs_dicts.append(obs_dict)
                pickle.dump(clean_energy, open(f"{energy_save_path}ibc_clean_energy_{self.epsilon}_timestep_{timestep}.pkl", "wb"))
                pickle.dump(perturbed_energy, open(f"{energy_save_path}ibc_perturbed_energy_{self.epsilon}_timestep_{timestep}.pkl", "wb"))

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())
                clean_action_dict = dict_apply(clean_dict,
                    lambda x: x.detach().to('cpu').numpy())
                # random_dict = dict_apply(random_dict,
                #     lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']
                clean_action = clean_action_dict['action']
                # print('action: ', action)
                # print('clean_action: ', clean_action)
                # print('difference: ', action - clean_action)
                # print('difference random: ', action - random_dict['action'])
                if cfg.log:
                    wandb.log({'L2_norm_actions': torch.norm(torch.tensor(action) - torch.tensor(clean_action), p=2)})
                
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action
                timestep += 1

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            #if cfg.log == True:
                # log the average perturbation for the whole step for 
                # each environment
                # average_perturbation = average_perturbation / self.max_steps
                # for i in range(len(average_perturbation)):
                #    wandb.log({f'average_perturbation_{view}_{i}': average_perturbation[i]})
            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        # save the obs_dicts as a pickle file
        # pickle.dump(clean_obs_dicts, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/clean_obs_dicts_{self.epsilon}_randtar_{cfg.rand_target}_{cfg.norm}.pkl", "wb"))
        # pickle.dump(perturbed_obs_dicts, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/target_perturbations/perturbed_obs_dicts_{self.epsilon}_tar_{cfg.target_perturbations}_perturb_{cfg.perturbations[0]}_{cfg.norm}.pkl", "wb"))

        # save the episode  lipschitz constants
        # pickle.dump(self.episodes_lipschitz_consts, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/lipschitz_consts_{self.epsilon}_randtar_{cfg.rand_target}_{cfg.norm}.pkl", "wb"))
        # pickle.dump(self.loss_per_episode, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/loss_per_episode_{self.epsilon}_tar_{cfg.target_perturbations}_{cfg.norm}_perturb_{cfg.perturbations[0]}_same_action_samples.pkl", "wb"))
        # pickle.dump(self.perturbation_per_episode, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/perturbation_per_episode_{self.epsilon}_randtar_{cfg.rand_target}_{cfg.norm}_lr_{cfg.eps_iter}.pkl", "wb"))
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data


class AdversarialRobomimicImageRunnerBET(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        The runner class for the adversarial attacks on the BET Policy
        """

    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy, cfg, target_action=None):
        """
        Applys the FGSM attack to the input image 
        """
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        # create a model prediction as ground truth
        with torch.no_grad():
            predicted_action = policy.predict_action(obs_dict)['action']
        # target latent
        # print(f"Shape of predicted action: {predicted_action.shape} and perturbations: {torch.tensor(cfg.perturbations).shape}")
        if target_action is None:
            target_action = predicted_action + torch.tensor(cfg.perturbations).unsqueeze(0).to(obs_dict['agentview_image'].device)
        # print(f"Prediction Action: {predicted_action} and Target Action: {target_action}")
        batch = {}
        batch['obs'] = obs_dict
        batch['action'] = target_action
        loss, _ = policy.compute_loss(batch)
        loss = -loss
        print("Loss: ", loss.item())
        loss.backward()
        for view in views:
            grad = torch.sign(obs_dict[view].grad)
            if cfg.attack_type == 'fgsm':
                obs_dict[view] = obs_dict[view] + self.epsilon * grad
            elif cfg.attack_type == 'pgd':
                obs_dict[view] = obs_dict[view] + cfg.eps_iter * grad
            else:
                raise ValueError("Only PGD and FGSM attacks are supported")
            obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
            if obs_dict[view].grad is not None:
                obs_dict[view].grad.zero_()
        
        return obs_dict


    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        view = cfg.view
        if view == 'both':
                views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        num_iter = cfg.num_iter
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        norm = cfg.norm

        adv_obs_dict = obs_dict.copy()
        with torch.no_grad():
            predicted_action = policy.predict_action(obs_dict)['action']
        target_action = predicted_action + torch.tensor(cfg.perturbations).to(obs_dict['agentview_image'].device)
        for i in range(num_iter):
            policy.zero_grad()
            adv_obs_dict = self.apply_fgsm_attack(adv_obs_dict, policy, cfg, target_action)
            for view in views:
                perturbation = adv_obs_dict[view] - obs_dict[view]
                if norm == 'l2':
                    perturbation = perturbation * self.epsilon / torch.norm(perturbation, p=2)
                elif norm == 'linf':
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_obs_dict[view] = obs_dict[view] + perturbation
                adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
        return adv_obs_dict



    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            self.step = 0
            while not done:
                print(f"Step: {self.step}")
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # print(type(obs_dict), obs_dict['agentview_image'].shape)

                # apply attack
                if cfg.attack_type == 'fgsm':
                    obs_dict = self.apply_fgsm_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'pgd':
                    obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'noise':
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")
                self.step += 1

class AdversarialRobomimicImageRunnerDP(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        The runner class for the adversarial attacks on the Diffusion Policy
        """

    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy, cfg, target_action=None):
        """
        Applys the FGSM attack to the input image 
        """
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']                
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        # create a copy of the original obs_dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()

        # create a model prediction as ground truth
        with torch.no_grad():
            predicted_action = policy.predict_action(obs_dict)['action']
        # print(f'Predicted Action shape: {predicted_action.shape}')

        if target_action is None:
            target_action = predicted_action + torch.tensor(cfg.perturbations).unsqueeze(0).to(obs_dict['agentview_image'].device)
        batch = {}
        batch['obs'] = obs_dict
        batch['action'] = target_action
        loss = policy.compute_loss(batch)
        loss = -loss
        print("Loss: ", loss.item())
        if cfg.log:
            wandb.log({"loss":loss.item()})
        loss.backward()
        for view in views:
            grad = torch.sign(obs_dict[view].grad)
            if cfg.attack_type == 'fgsm':
                obs_dict[view] = obs_dict[view] + self.epsilon * grad
            elif cfg.attack_type == 'pgd':
                obs_dict[view] = obs_dict[view] + cfg.eps_iter * grad
            else:
                raise ValueError("Only PGD and FGSM attacks are supported")
            obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
            if obs_dict[view].grad is not None:
                obs_dict[view].grad.zero_()
        return obs_dict

    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        return policy.pgd_perturbed_obs(obs_dict, cfg)

    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # self.observations = []
        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        perturbed_agentview_images = []
        perturbed_robot0_eye_in_hand_images = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            self.step = 0
            while not done:
                print(f"Step: {self.step}")
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # self.observations.append(obs_dict)
                # print(type(obs_dict), obs_dict['agentview_image'].shape)
                # apply attack
                prev_obs_dict = obs_dict.copy()
                if cfg.attack_type == 'fgsm':
                    obs_dict = self.apply_fgsm_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'pgd':
                    obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'noise':
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")
                self.step += 1
                # calculate the linf and l2 norm of the perturbation
                for view in views:
                    perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                    if cfg.log:
                        wandb.log({f'perturbation_{view}_l2': torch.norm(perturbation, p=2), \
                                    f'perturbation_{view}_linf': torch.norm(perturbation, p=float('inf'))})
                    # print(f"Norm of perturbation_{view}: ", torch.norm(perturbation, p=2), torch.norm(perturbation, p=float('inf')))
                perturbed_agentview_images.append(obs_dict['agentview_image'][1][1])
                perturbed_robot0_eye_in_hand_images.append(obs_dict['robot0_eye_in_hand_image'][1][1])
                with torch.no_grad():
                    action = policy.predict_action(obs_dict)['action']
                # device transfer
                np_action = action.detach().to('cpu').numpy()
                # step env
                env_action = np_action
                if self.abs_action:
                    env_action = self.undo_transform_action(np_action)
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = np_action

                # update pbar
                pbar.update(action.shape[1])
            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            pbar.close()
            # log the 


        def visualize(observations1, observations2, file_name, every_frame=2):
            # only get alternate frames to reduce the processing time
            print(f"Shape of observations1: {observations1[0].shape}")
            print(f"Shape of observations2: {observations2[0].shape}")
            observations1 = observations1[::every_frame]
            observations2 = observations2[::every_frame]
            im = []
            for i in range(len(observations1)):
                agentview_image1 = observations1[i].cpu().detach().numpy()
                agentview_image1 = np.transpose(agentview_image1, (1, 2, 0))
                
                agentview_image2 = observations2[i].cpu().detach().numpy()
                agentview_image2 = np.transpose(agentview_image2, (1, 2, 0))
                
                combined_image = np.hstack((agentview_image1, agentview_image2))
                im.append(combined_image)

            fig, ax = plt.subplots()
            ims = []
            for i in range(len(im)):
                im_obj = ax.imshow(im[i], animated=True)
                ims.append([im_obj])
                ax.axis('off')

            ani = ArtistAnimation(fig, ims, interval=100, blit=True)
            ani.save(file_name, writer='pillow', fps=10)
        
        # visualize the perturbed images
        filename = "/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/videos/diffusion_policy_perturbations/pgd_perturbations.gif"
        visualize(perturbed_agentview_images, perturbed_robot0_eye_in_hand_images, filename, every_frame=2)


        # clear out video buffer
        _ = env.reset()
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        print("All rewards: ", all_rewards)
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        # save the observation dicts as a pickle file
        # pickle.dump(self.observations, open(f"/teamspace/studios/this_studio/bc_attacks/diffusion_policy/plots/pkl_files/diffusion_policy_observed_dicts.pkl", "wb"))
        return log_data

            

class AdversarialRobomimicImageRunnerBET(RobomimicImageRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        The runner class for the adversarial attacks on the BET Policy
        """

    def apply_fgsm_attack(self, obs_dict, policy:BaseImagePolicy, cfg, predicted_action=None):
        """
        Applies the FGSM attack to the input image 
        """
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        # create a copy of the original dict that requires grad for backward pass
        obs_dict = dict_apply(obs_dict, lambda x: x.clone().detach().requires_grad_(True))
        policy.zero_grad()
        if predicted_action == None:
            with torch.no_grad():
                predicted_action = policy.predict_action(obs_dict)['action']
        batch = {}
        batch['obs'] = obs_dict
        batch['action'] = predicted_action
        loss, _ = policy.compute_loss(batch)
        if cfg.targeted:
            loss = -loss
        loss.backward()
        if cfg.log:
            wandb.log({"loss":loss.item()})
        for view in views:
            grad = torch.sign(obs_dict[view].grad)
            # log the gradient norm for each view
            if cfg.log:
                wandb.log({f'grad_{view}': torch.norm(obs_dict[view].grad, p=2)})
            if cfg.eps_iter != 'None':
                obs_dict[view] = obs_dict[view] + cfg.eps_iter * grad
            else:
                obs_dict[view] = obs_dict[view] + cfg.epsilon * grad
            obs_dict[view] = torch.clamp(obs_dict[view], cfg.clip_min, cfg.clip_max)
            if obs_dict[view].grad is not None:
                obs_dict[view].grad.zero_()
        return obs_dict


    def apply_pgd_attack(self, obs_dict, policy:BaseImagePolicy, cfg):
        """
        Apply projected gradient descent attack from Madry et al. (2017)
        """
        view = cfg.view
        if view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(view, list):
            views = view
        elif isinstance(view, str):
            views = [view]
        else:
            raise ValueError("view must be a string or a list of strings")
        num_iter = cfg.num_iter
        clip_min = cfg.clip_min
        clip_max = cfg.clip_max
        norm = cfg.norm

        adv_obs_dict = obs_dict.copy()
        with torch.no_grad():
            predicted_action = policy.predict_action(obs_dict)['action']
            if cfg.targeted:
                predicted_action = predicted_action + torch.tensor(cfg.perturbations).to(obs_dict['agentview_image'].device)
        for i in range(num_iter):
            policy.zero_grad()
            adv_obs_dict = self.apply_fgsm_attack(adv_obs_dict, policy, cfg, predicted_action)
            for view in views:
                perturbation = adv_obs_dict[view] - obs_dict[view]
                if norm == 'l2':
                    perturbation = perturbation * self.epsilon / torch.norm(perturbation, p=2)
                elif norm == 'linf':
                    perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                adv_obs_dict[view] = obs_dict[view] + perturbation
                adv_obs_dict[view] = torch.clamp(adv_obs_dict[view], clip_min, clip_max)
        return adv_obs_dict
    
    def run(self, policy: BaseImagePolicy, epsilon: float, cfg):
        self.epsilon = epsilon
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)
        if cfg.view == 'both':
            views = ['agentview_image', 'robot0_eye_in_hand_image']
        elif isinstance(cfg.view, list):
            views = cfg.view
        elif isinstance(cfg.view, str):
            views = [cfg.view]

        # self.observations = []
        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        perturbed_agentview_images = []
        perturbed_robot0_eye_in_hand_images = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            self.step = 0
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))
                # self.observations.append(obs_dict)
                # print(type(obs_dict), obs_dict['agentview_image'].shape)
                # apply attack
                prev_obs_dict = obs_dict.copy()
                if cfg.attack_type == 'fgsm':
                    if cfg.targeted:
                        with torch.no_grad():
                            predicted_action = policy.predict_action(obs_dict)['action']
                            predicted_action = predicted_action + torch.tensor(cfg.perturbations).to(obs_dict['agentview_image'].device)
                        obs_dict = self.apply_fgsm_attack(obs_dict, policy, cfg, predicted_action)
                    else:
                        obs_dict = self.apply_fgsm_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'pgd':
                    obs_dict = self.apply_pgd_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == 'noise':
                    obs_dict = self.apply_noise_attack(obs_dict, policy, cfg)
                elif cfg.attack_type == None:
                    pass
                else:
                    raise ValueError("Invalid attack type")
                self.step += 1
                # calculate the linf and l2 norm of the perturbation
                for view in views:
                    perturbation = abs(obs_dict[view] - prev_obs_dict[view])
                    if cfg.log:
                        wandb.log({f'perturbation_{view}_l2': torch.norm(perturbation, p=2), \
                                    f'perturbation_{view}_linf': torch.norm(perturbation, p=float('inf'))})
                    # print(f"Norm of perturbation_{view}: ", torch.norm(perturbation, p=2), torch.norm(perturbation, p=float('inf')))
                # perturbed_agentview_images.append(obs_dict['agentview_image'][1][0])
                # perturbed_robot0_eye_in_hand_images.append(obs_dict['robot0_eye_in_hand_image'][1][1])
                with torch.no_grad():
                    action = policy.predict_action(obs_dict)['action']
                # device transfer
                np_action = action.detach().to('cpu').numpy()
                # step env
                env_action = np_action
                if self.abs_action:
                    env_action = self.undo_transform_action(np_action)
                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = np_action

                # update pbar
                pbar.update(action.shape[1])
            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            pbar.close()

        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
                seed = self.env_seeds[i]
                prefix = self.env_prefixs[i]
                max_reward = np.max(all_rewards[i])
                max_rewards[prefix].append(max_reward)
                log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

                # visualize sim
                video_path = all_video_paths[i]
                if video_path is not None:
                    sim_video = wandb.Video(video_path)
                    log_data[prefix+f'sim_video_{seed}'] = sim_video
            
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

