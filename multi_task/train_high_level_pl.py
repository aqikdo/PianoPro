import sys
import pytorch_lightning as pl  # 新增Lightning依赖
from pytorch_lightning.callbacks import ModelCheckpoint
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
import os
import datetime

# 将模型封装为LightningModule
class DiffusionModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # 初始化模型
        self.midi_encoder = VariationalConvMlpEncoder(
            in_channels=16,
            mid_channels=32,
            out_channels=64,
            latent_dim=32,
            noise=0.08,
        )
        
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config["action_dim"],
            global_cond_dim=config["obs_dim"]*config["obs_horizon"],
            midi_dim=config["obs_dim"],
            midi_encoder=self.midi_encoder,
            midi_cond_dim=36,
            freeze_encoder=False,
        )
        
        # EMA模型
        self.ema = EMAModel(
            model=self.noise_pred_net,
            power=0.75)
        
        # 噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config["num_diffusion_iters"],
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def training_step(self, batch, batch_idx):
        # 数据移动到当前设备
        nobs = batch['obs'].to(self.device)
        naction = batch['action'].to(self.device)
        
        naction = naction.reshape(naction.shape[0], 4, -1)
        naction = naction[:, :, :36]
        B = nobs.shape[0]

        # 准备条件
        obs_cond = nobs[:,:self.hparams["obs_horizon"],:]
        obs_cond = obs_cond.flatten(start_dim=1)

        # 扩散过程
        noise = torch.randn(naction.shape, device=self.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (B,), device=self.device
        ).long()
        noisy_actions = self.noise_scheduler.add_noise(naction, noise, timesteps)

        # 预测噪声
        noise_pred = self.noise_pred_net(
            noisy_actions, timesteps, global_cond=obs_cond)

        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise) + self.noise_pred_net.kl
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)  # 多卡同步日志
        
        # 更新EMA
        self.ema.step(self.noise_pred_net)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.noise_pred_net.parameters(),
            lr=1e-4, weight_decay=1e-6)
        
        lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=self.hparams["num_epochs"] * self.hparams["steps_per_epoch"]
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def on_train_epoch_end(self):
        # 每400个epoch保存EMA模型
        if self.current_epoch % 400 == 0:
            ema_model = self.ema.averaged_model
            ckpt_path = os.path.join(
                self.hparams["log_dir"], 
                f'checkpoint_{self.hparams["run_name"]}_{self.current_epoch}.ckpt'
            )
            torch.save(ema_model.state_dict(), ckpt_path)

if __name__ == '__main__':
    # 配置参数
    config = {
        "pred_horizon": 1,
        "obs_horizon": 1,
        "action_horizon": 1,
        "obs_dim": 212,
        "action_dim": 36,
        "num_epochs": 1200,
        "num_diffusion_iters": 100,
        "dataset_path": sys.argv[1],
        #"steps_per_epoch": len(dataloader),
        "run_name": "high_level"
    }
    
    # 设置日志目录
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M-%S')
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(ROOT, 'runs', f'{timestamp}')
    os.makedirs(log_dir, exist_ok=True)
    config["log_dir"] = log_dir

    # 创建数据加载器（需确保使用DistributedSampler）
    dataloader, stats = read_dataset(
        pred_horizon=config["pred_horizon"],
        obs_horizon=config["obs_horizon"],
        action_horizon=config["action_horizon"],
        dataset_path=config["dataset_path"],
        normalization=True
    )
    
    # 初始化Lightning模型
    model = DiffusionModel(config)
    
    # 配置多卡训练器
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,  # 使用4个GPU
        strategy="ddp",
        max_epochs=config["num_epochs"],
        default_root_dir=log_dir,
        callbacks=[
            ModelCheckpoint(
                dirpath=log_dir,
                filename='{epoch}-{train_loss:.2f}',
                save_top_k=-1,  # 保存所有checkpoint
                every_n_epochs=100
            )
        ]
    )
    
    # 开始训练
    trainer.fit(model, dataloader)
    
    # 训练完成后保存最终模型
    final_ckpt_path = os.path.join(log_dir, f'checkpoint_{config["run_name"]}_final.ckpt')
    torch.save(model.ema.averaged_model.state_dict(), final_ckpt_path)
    print("Training completed!")