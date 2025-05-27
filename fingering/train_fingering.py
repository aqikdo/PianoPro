import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
import pickle
import fingering_network
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time
from tqdm import tqdm


def read_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



class ProcessedNote:
    """处理后的音符信息类"""
    def __init__(self, start_time, midi_number, dif, tim_dif, hand, original_note):
        self.start_time = start_time  # 开始时间（秒）
        self.midi_number = midi_number  # MIDI编号
        self.hand = hand  # 左右手分配（'left'/'right'）
        self.fingering = original_note.fingering
        self.midi_difference = dif
        self.time_difference = tim_dif
        self.original_note = original_note  # 原始PianoNote对象

def process_notes(notes_data, dt=0.5):
    """
    处理音符数据，生成带时间、MIDI编号和左右手分配的结果
    :param notes_data: 二维列表，每个子列表表示一个时间步的音符
    :param dt: 每个时间步的持续时间（秒），默认0.5秒（对应120BPM的二分音符）
    :param split_point: 左右手分界MIDI编号（默认中央C=60）
    :return: 包含ProcessedNote对象的列表
    """
    processed_notes = []
    
    last_midi = None  # 用于记录上一个音符的MIDI编号
    last_time = None  # 用于记录上一个音符的开始时间
    
    for step_idx, step_notes in enumerate(notes_data):
        start_time = step_idx * dt
        
        for piano_note in step_notes:
            # 确定左右手分配（低音为左手，高音为右手)
            hand = 'left' if piano_note.fingering >= 5 else 'right'
            
            # 创建处理后的音符对象
            processed_note = ProcessedNote(
                start_time=start_time,
                midi_number=piano_note.number,
                dif = piano_note.number - last_midi if last_midi is not None else 0,
                tim_dif = start_time - last_time if last_time is not None else 0,
                hand=hand,
                original_note=piano_note
            )
            processed_notes.append(processed_note)

            last_midi = piano_note.number
            last_time = start_time
    
    return processed_notes

class CustomDataset(Dataset):
    def __init__(self, input_list, output_list):
        """
        input_list: 输入数据列表，每个元素是形状为 [2, sequence_length] 的数组
        output_list: 输出数据列表，每个元素是形状为 [1, sequence_length] 的数组
        """
        # 转换为PyTorch张量并确保数据类型正确
        self.inputs = torch.tensor(input_list, dtype=torch.float32)
        self.outputs = torch.tensor(output_list, dtype=torch.long)
        print(self.outputs.shape)
        # 检查数据维度是否匹配
        assert len(self.inputs) == len(self.outputs), "输入输出数据数量不匹配"
        assert self.inputs.shape[1:] == (2, 100), "输入数据形状应为 [n_samples, 2, seq_len]"

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# 示例用法
def find_pkl_files(folder_path):
    pkl_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pkl"):
                full_path = os.path.join(root, file)
                pkl_files.append(full_path)
    return pkl_files

# 使用示例
folder = "/data15/jinkun.liu.2502/shared/dataset/notes"
all_pkl = find_pkl_files(folder)
input_data = []
output_data = []
for file_path in all_pkl:
    print(f"Processing file: {file_path}")
    tmpdata = read_pkl_file(file_path)
    sample_notes = process_notes(tmpdata.notes, dt=tmpdata.dt)
    model_input_1 = []
    model_input_2 = []
    model_output = []
    cnt = 0 
    for note in sample_notes:
        model_input_1.append((note.midi_number-44)/44.0)
        model_input_2.append(note.time_difference)
        model_output.append(note.fingering)
        cnt += 1
        if cnt % 100 == 0:
            input_data.append([model_input_1, model_input_2])
            output_data.append(model_output)
            model_input_1 = []
            model_input_2 = []
            model_output = []
            cnt = 0
    # 读取数据


# 打印结果
"""
for note in processed_notes:
    print(
        f"Start: {note.start_time:.1f}s, "
        f"MIDI: {note.midi_number}, "
        f"Hand: {note.hand}, "
        f"Fingering: {note.fingering}, "
        f"Original: {note.original_note.name}"
    )
"""
"""
def compute_pitch_difference_representation(notes):
    
    输入参数:
    notes - 包含音符信息的列表，每个音符是包含以下键的字典:
        'hand' : 'left' 或 'right' (表示左右手)
        'start_time' : 音符开始时间 (float)
        'midi' : MIDI音高编号 (int)
    
    返回值:
    包含左右手编码结果的字典
    
    
    # 分离左右手数据
    left_notes = [n for n in notes if n.hand == 'left']
    right_notes = [n for n in notes if n.hand == 'right']

    def _process_hand(hand_notes):
        # 按开始时间分组
        time_groups = defaultdict(list)
        for note in hand_notes:
            time_groups[note.start_time].append(note)

        # 生成排序后的时间序列
        sorted_times = sorted(time_groups.keys())
        sequence = []
        
        # 构建带n值的序列
        for st in sorted_times:
            # 按音高排序
            sorted_notes = sorted(time_groups[st], key=lambda x: x.midi_number)
            num_notes = len(sorted_notes)
            
            # 确定n值 (单音为0)
            n = 0 if num_notes == 1 else num_notes
            sequence.append({
                'midi': sorted_notes[0].midi_number,  # 取最低音
                'n': n,
                'note': sorted_notes[0]  # 取最低音的原始音符对象
            })

        #print(sequence)

        # 计算d(t)序列
        d_sequence = []
        prev_midi = None
        
        for i, item in enumerate(sequence):
            if i == 0:  # 初始时间步
                d = 100 * item['n']
            else:
                diff = item['midi'] - prev_midi
                if abs(diff) < 12:
                    d = diff + 100 * item['n']
                else:
                    # 符号函数处理
                    d = 80 * np.sign(diff)
            
            d_sequence.append(int(d))
            prev_midi = item['midi']

        # 整数编码
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(d_sequence)
        
        return encoded.tolist()

    return {
        'left': _process_hand(left_notes),
        'right': _process_hand(right_notes)
    }
"""

EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# 简化训练函数
def simple_train(model, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        # 进度条包装数据加载器
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for inputs, targets in pbar:
            # 数据转移到设备
            inputs = inputs.to(device)   # [batch, 2, seq_len]
            targets = targets.to(device)  # [batch, seq_len]

            # 前向传播
            outputs = model(inputs)       # [batch, num_classes, seq_len]
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)  # 取预测类别
            correct += (predicted == targets).sum().item()
            total += targets.numel()  # 总预测数 = batch * seq_len

            # 更新进度条
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100*correct/total:.1f}%"
            })

        # 打印epoch结果
        print(f"Epoch {epoch+1} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Acc: {100*correct/total:.1f}%")

    return model

# 使用示例 --------------------------------------------------
if __name__ == "__main__":
    # 示例数据（假设序列长度100，3个类别）
    # 创建数据集和数据加载器
    dataset = CustomDataset(input_data, output_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型（3分类任务）
    model = fingering_network.ConvNet1D(num_classes=10)

    # 开始训练
    trained_model = simple_train(model, dataloader)