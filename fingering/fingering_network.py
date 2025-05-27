import torch
import torch.nn as nn

class TransformerMLP(nn.Module):
    def __init__(self,
                 input_channels=3,    # 输入通道数（特征维度）
                 seq_len=4096,          # 序列长度
                 d_model=512,         # Transformer 隐藏维度
                 nhead=8,              # 注意力头数
                 num_layers=6,         # Transformer 层数
                 num_labels=10,       # 输出类别数
                 mlp_hidden=256):      # MLP 隐藏层维度
        super().__init__()
        
        # 输入维度说明
        # 输入形状: (batch_size, channels, seq_len)
        
        # 通道到隐藏维度的投影层
        self.channel_proj = nn.Linear(input_channels, d_model)  # (C → D)
        
        # 可学习位置编码（针对序列位置）
        self.pos_embedding = nn.Parameter(
            torch.randn(1, seq_len, d_model)  # (1, S, D)
        )
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )  # 输入输出: (S, B, D)
        
        # MLP 分类头
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),  # (D → H)
            nn.GELU(),
            nn.Linear(mlp_hidden, num_labels)  # (H → L)
        )

    def forward(self, x):
        """
        输入:
        x : torch.FloatTensor (batch_size, channels, seq_len)
        
        输出:
        logits : torch.FloatTensor (batch_size, num_labels)
        """
        # 初始维度验证
        B, C, S = x.shape  # (Batch, Channels, SeqLen)
        
        # 步骤 1: 通道投影
        x = x.permute(0, 2, 1)  # (B, S, C)
        x = self.channel_proj(x)  # (B, S, D)
        
        # 步骤 2: 添加位置编码
        x = x + self.pos_embedding[:, :S, :]  # (B, S, D)
        
        # 步骤 3: 调整维度给 Transformer
        x = x.permute(1, 0, 2)  # (S, B, D)
        
        # 步骤 4: Transformer 处理
        transformer_out = self.transformer(x)  # (S, B, D)
        
        # 步骤 5: 取序列平均作为全局特征
        pooled = transformer_out.mean(dim=0)  # (B, D)
        
        # 步骤 6: MLP 分类
        logits = self.mlp(pooled)  # (B, L)
        return logits

class ConvNet1D(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet1D, self).__init__()
        self.net = nn.Sequential(
            # 第一层卷积：2个输入通道，16个输出通道，保持长度不变
            nn.Conv1d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            # 第二层卷积：16 -> 32个通道
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, num_classes, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.net(x)

# 纯结构验证
if __name__ == "__main__":
    # 参数配置
    batch_size = 4
    channels = 64
    seq_len = 128
    num_labels = 10
    
    # 初始化模型
    model = TransformerMLP(
        input_channels=channels,
        seq_len=seq_len,
        num_labels=num_labels
    )
    
    # 生成随机输入 (模拟传感器数据/图像块序列/时间序列特征)
    dummy_input = torch.randn(batch_size, channels, seq_len)  # (4, 64, 128)
    
    # 前向传播
    output = model(dummy_input)
    
    print("输入维度:", dummy_input.shape)  # torch.Size([4, 64, 128])
    print("输出维度:", output.shape)      # torch.Size([4, 10])