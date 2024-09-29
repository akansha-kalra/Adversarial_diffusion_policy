__author__ = "akansha_kalra"
import os
import hydra

import subprocess

import subprocess


def run_command(command):
    try:
        # Running the command using bash
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Command '{command}' output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Command '{command}' failed with error:\n{e.stderr}")


if __name__ == "__main__":
    # List of config names for the command
    # config_names = [
    #     'Run2-train_vq_bet_image_workspace_Square_PH-seed123.yaml',
    #     'Run3-train_vq_bet_image_workspace_Square_PH-seed1001.yaml',
    # ]
    config_names = [

        'Run3-train_vq_bet_image_workspace_Square_PH-seed1001.yaml',
    ]

    # Loop through each config name and run the corresponding command
    config_dir= '/home/ak/Documents/Adversarial_diffusion_policy/diffusion_policy/config/'

    for config in config_names:
        command = f'python train.py --config-name={config} --config-dir={config_dir} '
        print(f"Running command: {command}")
        run_command(command)

