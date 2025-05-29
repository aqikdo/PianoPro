# export OMP_NUM_THREADS=8
# CUDA_VISIBLE_DEVICES=1,2,4 torchrun --nproc_per_node=3 multi_task/train_high_level_ddp.py

import sys
directory = 'pianomime'
if directory not in sys.path:
    sys.path.append(directory)
from network import ConditionalUnet1D, EMAModel, ConvEncoder, VariationalConvMlpEncoder
import torch
import math
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from dataset import RoboPianistDataset, read_dataset
import sys
import time
import wandb
import sys
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def main():
    # Initialize DDP
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ['OMP_NUM_THREADS'] = '12'
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    print(f"Using device: {device}, local rank: {local_rank}")

    pred_horizon = 1
    action_horizon = 1
    obs_horizon = 1
    obs_dim = 212
    action_dim = 36
    num_songs = 50
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset_hl.zarr")

    # create dataloader with DistributedSampler
    dataloader, stats = read_dataset(pred_horizon=pred_horizon,
                              obs_horizon=obs_horizon,
                              action_horizon=action_horizon,
                              dataset_path=dataset_path,
                              normalization=True)
    sampler = DistributedSampler(dataloader.dataset)
    dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        sampler=sampler,
        num_workers=7,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    def create_midi_encoder(device='cuda'):
        midi_encoder = VariationalConvMlpEncoder(
            in_channels=16,
            mid_channels=32,
            out_channels=64,
            latent_dim=32,
            noise=0.08,
        ).to(device)
        return midi_encoder

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon,
        midi_dim=obs_dim,
        midi_encoder=create_midi_encoder,
        midi_cond_dim=36,
        freeze_encoder=False,
    ).to(device)

    # Wrap model with DDP
    noise_pred_net = DDP(noise_pred_net, 
                        device_ids=[local_rank], 
                        output_device=local_rank)

    num_epochs = 1200
    ema = EMAModel(
        model=noise_pred_net.module,
        power=0.75)
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(dataloader) * num_epochs
    )
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # Create save directory only on rank 0
    if dist.get_rank() == 0:
        exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp")
        os.makedirs(exp_dir, exist_ok=True)
        save_dir = os.path.join(exp_dir, f"high_level_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
    dist.barrier()
    run_name = "DF-HL-Original"

    # Initialize wandb only on rank 0
    if dist.get_rank() == 0:
        wandb.init(
            project="pianomime",
            name=run_name,
            config={
                "learning_rate": 1e-4,
                "epochs": num_epochs,
                "batch_size": dataloader.batch_size,
                "num_diffusion_iters": num_diffusion_iters,
                "action_dim": action_dim,
                "obs_dim": obs_dim
            }
        )

    # Training loop
    if dist.get_rank() == 0:
        epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    else:
        epoch_pbar = range(num_epochs)
        
    for epoch_idx in epoch_pbar:
        sampler.set_epoch(epoch_idx)
        epoch_loss = list()
        
        if dist.get_rank() == 0:
            batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx}", leave=False)
        else:
            batch_pbar = dataloader
            
        for nbatch in batch_pbar:
            nobs = nbatch['obs'].to(device)
            naction = nbatch['action'].to(device)
            naction = naction.reshape(naction.shape[0], 4, -1)
            naction = naction[:, :, :36]
            B = nobs.shape[0]
            obs_cond = nobs[:,:obs_horizon,:]
            obs_cond = obs_cond.flatten(start_dim=1)
            noise = torch.randn(naction.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            noisy_actions = noise_scheduler.add_noise(
                naction, noise, timesteps)
            noise_pred = noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise) + noise_pred_net.module.kl
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            ema.step(noise_pred_net.module)
            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            if dist.get_rank() == 0:
                batch_pbar.set_postfix({"loss": f"{loss_cpu:.4f}"})
                wandb.log({
                    "batch_loss": loss_cpu,
                    "epoch": epoch_idx,
                    "lr": lr_scheduler.get_last_lr()[0],
                    "kl_loss": noise_pred_net.module.kl
                })
        
        # Logging and checkpointing on rank 0
        if dist.get_rank() == 0:
            epoch_pbar.set_postfix({"avg_loss": f"{np.mean(epoch_loss):.4f}"})
            wandb.log({
                "epoch_loss": np.mean(epoch_loss),
                "epoch": epoch_idx
            })
            if epoch_idx % 400 == 0:
                ema_noise_pred_net = ema.averaged_model
                ema_model_state_dict = ema_noise_pred_net.state_dict()
                workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                save_dir = os.path.join(workspace_dir, f"exp/high_level_{time.strftime('%Y%m%d_%H%M%S')}")
                os.makedirs(save_dir, exist_ok=True)
                ema_model_weights_path = os.path.join(save_dir, f'ck_{epoch_idx}.ckpt')
                torch.save(ema_model_state_dict, ema_model_weights_path)
                print("Saved checkpoint at epoch {}".format(epoch_idx))
    
    # Final cleanup
    if dist.get_rank() == 0:
        wandb.finish()
        ema_noise_pred_net = ema.averaged_model
        ema_model_state_dict = ema_noise_pred_net.state_dict()
        workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_dir = os.path.join(workspace_dir, f"exp/high_level_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(save_dir, exist_ok=True)
        ema_model_weights_path = os.path.join(save_dir, f'ck_final.ckpt')
        torch.save(ema_model_state_dict, ema_model_weights_path)
        print("Done!")
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
