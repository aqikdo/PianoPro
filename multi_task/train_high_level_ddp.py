import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm import tqdm
import numpy as np
import datetime

from dataset import read_dataset
from network import ConditionalUnet1D, VariationalConvMlpEncoder

def create_midi_encoder(device='cuda'):
    return VariationalConvMlpEncoder(
        in_channels=16,
        mid_channels=32,
        out_channels=64,
        latent_dim=32,
        noise=0.08,
    ).to(device)

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12341'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main_worker(rank, world_size, dataset_path):
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M-%S')
        ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(ROOT, 'runs', f'{timestamp}')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    pred_horizon = 1
    action_horizon = 1
    obs_horizon = 1
    obs_dim = 212
    action_dim = 36
    num_epochs = 1200

    raw_loader, stats = read_dataset(
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        dataset_path=dataset_path,
        normalization=True
    )
    sampler = DistributedSampler(raw_loader.dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(raw_loader.dataset, batch_size=raw_loader.batch_size, sampler=sampler, num_workers=8, pin_memory=True)

    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        midi_dim=obs_dim,
        midi_encoder=create_midi_encoder,
        midi_cond_dim=36,
        freeze_encoder=False,
    ).to(device)
    noise_pred_net = nn.parallel.DistributedDataParallel(noise_pred_net, device_ids=[rank], find_unused_parameters=True)

    ema = EMAModel(noise_pred_net.module, power=0.75)
    optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(dataloader) * num_epochs)

    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    # run_name = f"DF-HL-{dataset_path.split('.')[0]}"
    run_name = f"train_high_level"
    epoch_bar = tqdm(range(num_epochs), desc='Epoch') if rank == 0 else range(num_epochs)
    for epoch_idx in epoch_bar:
        sampler.set_epoch(epoch_idx)
        epoch_loss = []

        batch_bar = tqdm(dataloader, desc=f'Batch {epoch_idx}', leave=True) if rank == 0 else dataloader
        for nbatch in batch_bar:
            nobs = nbatch['obs'].to(device)
            naction = nbatch['action'].to(device).reshape(nbatch['action'].shape[0], 4, -1)[:, :, :36]
            B = nobs.shape[0]
            obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1)

            noise = torch.randn_like(naction)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
            noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

            noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
            loss = nn.functional.mse_loss(noise_pred, noise) + noise_pred_net.module.kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            ema.step(noise_pred_net.module)

            loss_cpu = loss.item()
            epoch_loss.append(loss_cpu)
            if rank == 0:
                batch_bar.set_postfix(loss=loss_cpu)

        if rank == 0:
            avg_loss = np.mean(epoch_loss)
            epoch_bar.set_postfix(loss=avg_loss)
            writer.add_scalar("Loss/train", avg_loss, epoch_idx)
            writer.add_scalar("Learning Rate", lr_scheduler.get_last_lr()[0], epoch_idx)

            if epoch_idx % 400 == 0:
                ckpt_path = log_dir + f'/checkpoint_{run_name}_{epoch_idx}.ckpt'
                torch.save(ema.averaged_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint to {ckpt_path}")

    if rank == 0:
        # final_path = f'diffusion/ckpts/checkpoint_{run_name}_final_without_fingering.ckpt'
        final_path = os.path.join(log_dir, f'checkpoint_{run_name}_final_without_fingering.ckpt')
        torch.save(ema.averaged_model.state_dict(), final_path)
        writer.close()
        print("Done!")

    cleanup_ddp()


if __name__ == '__main__':
    dataset_path = sys.argv[1]
    available_gpus = list(range(0, 7))
    world_size = len(available_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))
    mp.spawn(main_worker, args=(world_size, dataset_path), nprocs=world_size, join=True)
