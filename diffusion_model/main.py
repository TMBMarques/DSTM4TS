# Necessary packages and functions call

import os
import torch

from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond
from Utils.io_utils import load_yaml_config, instantiate_from_config
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one

from config import TRAIN, DIFFUSION_MODEL_CONFIG_YAML

def run_diffusion_model():
    # Train model
    
    if TRAIN:

        # Build dataset and settings

        class Args_Example:
            def __init__(self) -> None:
                self.config_path = DIFFUSION_MODEL_CONFIG_YAML
                self.save_dir = './diffusion_model/toy_exp'
                self.gpu = 0
                os.makedirs(self.save_dir, exist_ok=True)

        args =  Args_Example()
        configs = load_yaml_config(args.config_path)
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

        dl_info = build_dataloader(configs, args)

        model = instantiate_from_config(configs['model']).to(device)
        trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

        # Training model

        trainer.train()

        return

    # Load and run the trained model

    # Build dataset and settings

    class Args_Example:
        def __init__(self) -> None:
            self.gpu = 0
            self.config_path = DIFFUSION_MODEL_CONFIG_YAML
            self.save_dir = './diffusion_model/toy_exp'
            self.mode = 'infill'
            self.missing_ratio = 0.5
            self.milestone = 10
            os.makedirs(self.save_dir, exist_ok=True)

    args =  Args_Example()
    configs = load_yaml_config(args.config_path)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dl_info = build_dataloader_cond(configs, args)

    model = instantiate_from_config(configs['model']).to(device)
    trainer = Trainer(config=configs, args=args, model=model, dataloader=dl_info)

    # Load model

    trainer.load(args.milestone)

    # Sampling

    dataloader, dataset = dl_info['dataloader'], dl_info['dataset']
    coef = configs['dataloader']['test_dataset']['coefficient']
    stepsize = configs['dataloader']['test_dataset']['step_size']
    sampling_steps = configs['dataloader']['test_dataset']['sampling_steps']
    seq_length, feature_dim = dataset.window, dataset.var_num
    samples, ori_data, masks = trainer.restore(dataloader, [seq_length, feature_dim], coef, stepsize, sampling_steps)

    samples = unnormalize_to_zero_to_one(samples)
    samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)

    return samples


if __name__ == '__main__':
    run_diffusion_model()
