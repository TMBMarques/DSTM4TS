# Necessary packages and functions call

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

from pathlib import Path

import importlib

from pre_trained_models.models import PreTrainedModel

def run_diffusion_model(pre_trained_model, stress_weight, df):

    folder = Path("Checkpoints_room_temperature_72")

    # Train the model if it was not already trained

    if not folder.exists() or not folder.is_dir() or not any(folder.iterdir()):

        # Build dataset and settings

        class Args_Example:
            def __init__(self) -> None:
                self.config_path = './diffusion_model/Config/room_temperature.yaml'
                self.save_dir = './diffusion_model/toy_exp'
                self.gpu = 0
                os.makedirs(self.save_dir, exist_ok=True)

        args =  Args_Example()
        configs = load_yaml_config(args.config_path)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        #dl_info = build_dataloader(configs, args)

        # - - - build_dataloader - - -
        batch_size = configs['dataloader']['batch_size']
        jud = configs['dataloader']['shuffle']
        configs['dataloader']['train_dataset']['params']['output_dir'] = args.save_dir

        #dataset = instantiate_from_config(config['dataloader']['train_dataset'])

        # - - - instantiate_from_config - - -
        config = configs['dataloader']['train_dataset']
        module, cls = config["target"].rsplit(".", 1)
        cls = getattr(importlib.import_module(module, package=None), cls)
        dataset = cls(df=df, generating_samples=False, **config.get("params", dict()))
        # - - - - - -

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=jud,
                                                 num_workers=0,
                                                 pin_memory=True,
                                                 sampler=None,
                                                 drop_last=jud)

        dataload_info = {
            'dataloader': dataloader,
            'dataset': dataset
        }

        dl_info = dataload_info
        # - - - - - -

        model = instantiate_from_config(configs['model']).to(device)
        trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

        # Training model

        trainer.train(pre_trained_model, stress_weight)

    # Loading the trained model

    class Args_Example:
        def __init__(self) -> None:
            self.gpu = 0
            self.config_path = './diffusion_model/Config/room_temperature.yaml'
            self.save_dir = './diffusion_model/toy_exp'
            self.mode = 'infill'
            self.missing_ratio = 0.5
            self.milestone = 10
            os.makedirs(self.save_dir, exist_ok=True)

    args =  Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    #dl_info = build_dataloader_cond(configs, args)

    # - - - build_dataloader_cond - - -
    batch_size = configs['dataloader']['sample_size']
    configs['dataloader']['test_dataset']['params']['output_dir'] = args.save_dir
    configs['dataloader']['test_dataset']['params']['missing_ratio'] = args.missing_ratio

    #test_dataset = instantiate_from_config(configs['dataloader']['test_dataset'])

    # - - - instantiate_from_config - - -
    config = configs['dataloader']['test_dataset']
    module, cls = config["target"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module, package=None), cls)
    test_dataset = cls(df=df, generating_samples=True, **config.get("params", dict()))
    # - - - - - -

    dataloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=False)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': test_dataset
    }

    dl_info = dataload_info
    # - - - - - -

    model = instantiate_from_config(configs['model']).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

    trainer.load(args.milestone)

    # Sampling

    dataloader, dataset = dl_info['dataloader'], dl_info['dataset']
    coef = configs['dataloader']['test_dataset']['coefficient']
    stepsize = configs['dataloader']['test_dataset']['step_size']
    sampling_steps = configs['dataloader']['test_dataset']['sampling_steps']
    seq_length, feature_dim = dataset.window, dataset.var_num
    samples, ori_data, masks = trainer.restore(dataloader, [seq_length, feature_dim], coef, stepsize, sampling_steps)

    """ if dataset.auto_norm:
        samples = unnormalize_to_zero_to_one(samples) """

    # Ploting the results

    """ plt.rcParams["font.size"] = 12
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 15))

    #ori_data = np.load(os.path.join(dataset.dir, f"sine_ground_truth_{seq_length}_test.npy"))
    ori_data = np.load(os.path.join(dataset.dir, f"room_temperature_norm_truth_{seq_length}_test.npy"))  # Uncomment the line if dataset other than Sine is used.
    #masks = np.load(os.path.join(dataset.dir, f"sine_masking_{seq_length}.npy"))
    masks = np.load(os.path.join(dataset.dir, f"room_temperature_masking_{seq_length}.npy"))
    sample_num, seq_len, feat_dim = ori_data.shape
    observed = ori_data * masks

    for feat_idx in range(feat_dim):
        df_x = pd.DataFrame({"x": np.arange(0, seq_len), "val": ori_data[0, :, feat_idx],
                            "y": masks[0, :, feat_idx]})
        df_x = df_x[df_x.y!=0]

        df_o = pd.DataFrame({"x": np.arange(0, seq_len), "val": ori_data[0, :, feat_idx],
                            "y": (1 - masks)[0, :, feat_idx]})
        df_o = df_o[df_o.y!=0]
        axes[feat_idx].plot(df_o.x, df_o.val, color='b', marker='o', linestyle='None')
        axes[feat_idx].plot(df_x.x, df_x.val, color='r', marker='x', linestyle='None')
        #axes[feat_idx].plot(range(0, seq_len), samples[0, :, feat_idx], color='g', linestyle='solid', label='Diffusion-TS')
        plt.setp(axes[feat_idx], ylabel='value')
        if feat_idx == feat_dim-1:
            plt.setp(axes[-1], xlabel='time')

    plt.show() """

    samples = unnormalize_to_zero_to_one(samples)
    samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
    ori_data = unnormalize_to_zero_to_one(ori_data)
    ori_data = dataset.scaler.inverse_transform(ori_data.reshape(-1, ori_data.shape[-1])).reshape(ori_data.shape)

    with open("samples.txt", "w") as file:
        file.write(np.array2string(samples, threshold=np.inf, separator=", "))

    with open("ori_data.txt", "w") as file:
        file.write(np.array2string(ori_data, threshold=np.inf, separator=", "))
    
    # Plot

    # Get sequence length and number of features
    seq_len, feat_dim = ori_data.shape[1], ori_data.shape[2]

    # Plot the original data (•) and the samples (x)
    plt.figure(figsize=(12, 6))

    for feat_idx in range(feat_dim):
        plt.scatter(range(seq_len), ori_data[0, :, feat_idx], color='b', marker='o', label='Original Data' if feat_idx == 0 else "")
        plt.scatter(range(seq_len), samples[0, :, feat_idx], color='r', marker='x', label='Generated Samples' if feat_idx == 0 else "")

    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title("Original vs Generated Data")
    plt.legend()
    plt.show()

    return samples


if __name__ == '__main__':
    run_diffusion_model()
