__author__ = "akansha_kalra"
import numpy as np
import pickle as pkl
import torch

patch_path = '/home/ak/Documents/Adversarial_diffusion_policy/pre_trained_checkpoints/square/diffusion_policy_cnn/train_0/untar_pert_0.125_epoch_40_mean_score_0.0_robot0_eye_in_hand_image.pkl'
# load the patch
patch = pkl.load(open(patch_path, 'rb'))['agentview_image']
# random patch with the same shape as the patch with range of -0.0625 to 0.0625
random_patch = {}
random_patch['robot0_eye_in_hand_image'] = np.random.rand(*patch.shape) * 0.125 - 0.0625
random_patch['agentview_image'] = np.random.rand(*patch.shape) * 0.125 - 0.0625
random_patch['robot0_eye_in_hand_image'] = torch.tensor(random_patch['robot0_eye_in_hand_image'], dtype=torch.float32)
random_patch['agentview_image'] = torch.tensor(random_patch['agentview_image'], dtype=torch.float32)
pkl.dump(random_patch, open('random_patch_robomimic.pkl', 'wb'))