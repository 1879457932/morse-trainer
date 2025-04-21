import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
import os
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter
from torch.quantization import QuantStub, DeQuantStub, quantize_dynamic

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MorseAI')

class MorseDataset(Dataset):
    def __init__(self, signal_patterns, labels):
        self.signal_patterns = signal_patterns
        self.labels = labels
    
    def __len__(self):
        return len(self.signal_patterns)
    
    def __getitem__(self, idx):
        return self.signal_patterns[idx], self.labels[idx]

class MorseNet(nn.Module):
    def __init__(self, input_size=100, quantize=False):
        super(MorseNet, self).__init__()
        self.quantize = quantize
        
        # 如果使用量化，添加量化/反量化桩
        if quantize:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()
        
        # 使用1D卷积代替LSTM，更高效
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(2)
        
        # 使用全局平均池化代替全连接层，减少参数量
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 轻量级全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)  # 10个数字（0-9）
        
        # 激活函数和Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 添加通道维度并转置
        x = x.unsqueeze(1)  # [batch, 1, seq_len]
        
        # 如果使用量化
        if self.quantize:
            x = self.quant(x)
        
        # 卷积层
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.global_pool(x)  # [batch, 128, 1]
        x = x.view(batch_size, -1)  # [batch, 128]
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # 如果使用量化
        if self.quantize:
            x = self.dequant(x)
            
        return x
        
    def fuse_model(self):
        """为量化准备模型，融合卷积和激活层"""
        torch.quantization.fuse_modules(self, [['conv1', 'relu'], 
                                               ['conv2', 'relu'], 
                                               ['conv3', 'relu']], inplace=True)

class MorseAI:
    def __init__(self, history_file='morse_history.json', quantize=False):
        self.history_file = history_file
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.history_path = os.path.join(self.data_dir, self.history_file)
        
        # 训练参数
        self.batch_size = 32
        self.learning_rate = 0.001
        
        # 模型参数
        self.input_size = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.quantize = quantize
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型和TensorBoard记录器
        self.model = MorseNet(input_size=self.input_size, quantize=quantize).to(self.device)
        self.log_dir = os.path.join(self.data_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # 在实际需要时才初始化SummaryWriter，避免不必要的资源占用
        self.writer = None

    def prepare_data(self):
        """准备训练数据"""
        signal_patterns = []
        labels = []
        
        try:
            # 从历史文件加载数据
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    history_data = json.load(f)
                
                # 处理历史数据
                for entry in history_data:
                    morse_code = entry.get('morse_code')
                    decoded = entry.get('decoded')
                    
                    if morse_code and decoded and decoded.isdigit():
                        # 将摩尔斯码转换为信号模式（用于模拟）
                        signal_pattern = self._morse_to_signal(morse_code)
                        
                        # 将解码结果转换为标签（0-9）
                        label = int(decoded)
                        
                        signal_patterns.append(signal_pattern)
                        labels.append(label)
            
            # 确保有足够的数据进行训练
            if len(signal_patterns) < 10:
                logger.warning(f"训练数据不足: 只有{len(signal_patterns)}个样本")
                return None, None
                
            # 转换为PyTorch张量
            X = torch.tensor(signal_patterns, dtype=torch.float32).to(self.device)
            y = torch.tensor(labels, dtype=torch.long).to(self.device)
            
            return X, y
        except Exception as e:
            logger.error(f"准备数据时出错: {e}")
            return None, None

    def _morse_to_signal(self, morse_code, length=100):
        """将摩尔斯码转换为信号模式"""
        signal = np.zeros(length, dtype=np.float32)
        
        idx = 0
        for char in morse_code:
            if char == '.':
                # 短信号
                signal_length = min(5, length - idx)
                if signal_length > 0:
                    signal[idx:idx+signal_length] = 1.0
                    idx += signal_length + 2
            elif char == '-':
                # 长信号
                signal_length = min(15, length - idx)
                if signal_length > 0:
                    signal[idx:idx+signal_length] = 1.0
                    idx += signal_length + 2
            
            # 防止超出长度
            if idx >= length:
                break
        
        return signal

    def train(self, epochs=100, batch_size=32):
        """训练模型"""
        # 准备数据
        X, y = self.prepare_data()
        if X is None or y is None:
            logger.error("无法准备训练数据")
            return False
        
        # 初始化SummaryWriter，仅在训练时使用
        if self.writer is None:
            try:
                self.writer = SummaryWriter(log_dir=self.log_dir)
            except Exception as e:
                logger.error(f"创建SummaryWriter失败: {e}")
                self.writer = None  # 确保writer为None
                
        # 创建数据集和数据加载器
        dataset = MorseDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # 早停机制
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        logger.info(f"开始训练: {epochs}轮, 每批{batch_size}个样本")
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            train_loss = 0.0
            train_acc = 0.0
            
            for i, (inputs, targets) in enumerate(train_loader):
                optimizer.zero_grad()
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                # 计算训练损失
                train_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                train_acc += (predicted == targets).sum().item() / targets.size(0)
                
                # 记录训练进度信息
                if i % 10 == 0:  # 每10个批次记录一次
                    logger.info(f"轮次 {epoch+1}/{epochs}, 批次 {i+1}/{len(train_loader)}, 损失: {loss.item():.4f}")
                    
                    # 记录到TensorBoard（如果可用）
                    if self.writer:
                        try:
                            self.writer.add_scalar('training_loss', loss.item(), epoch * len(train_loader) + i)
                        except Exception as e:
                            logger.warning(f"记录TensorBoard数据失败: {e}")
            
            # 计算平均损失和准确率
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)
            
            # 验证模式
            self.model.eval()
            val_loss = 0.0
            val_acc = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
                    
                    # 计算准确率
                    _, predicted = torch.max(outputs, 1)
                    val_acc += (predicted == targets).sum().item() / targets.size(0)
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # 计算平均验证损失和准确率
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            
            # 计算其他评估指标
            precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
            f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            
            # 更新学习率
            scheduler.step(val_loss)
            
            # 记录到TensorBoard
            if self.writer:
                try:
                    self.writer.add_scalar('Loss/train', train_loss, epoch)
                    self.writer.add_scalar('Loss/validation', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/train', train_acc, epoch)
                    self.writer.add_scalar('Accuracy/validation', val_acc, epoch)
                    
                    # 额外计算并记录精确度、召回率和F1分数（对于多分类问题）
                    preds = torch.argmax(outputs, axis=1).cpu().numpy()
                    targets_np = targets.cpu().numpy()
                    precision = precision_score(targets_np, preds, average='weighted', zero_division=0)
                    recall = recall_score(targets_np, preds, average='weighted', zero_division=0)
                    f1 = f1_score(targets_np, preds, average='weighted', zero_division=0)
                    
                    self.writer.add_scalar('Precision', precision, epoch)
                    self.writer.add_scalar('Recall', recall, epoch)
                    self.writer.add_scalar('F1', f1, epoch)
                except Exception as e:
                    logger.warning(f"记录TensorBoard指标失败: {e}")
            
            logger.info(f"轮次 {epoch+1}/{epochs} - 训练损失: {train_loss:.4f}, 准确率: {train_acc:.4f}, 验证损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.data_dir, 'morse_model.pth'))
                logger.info(f"模型已保存，验证损失: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= patience:
                logger.info(f"早停触发, 轮次 {epoch+1}")
                break
        
        # 在训练完成后关闭TensorBoard writer
        self.close_writer()
        
        # 如果启用量化，创建量化版本
        if self.quantize:
            self.quantize_model()
            
        logger.info("训练完成")
        return True

    def close_writer(self):
        """安全关闭TensorBoard writer"""
        if hasattr(self, 'writer') and self.writer is not None:
            try:
                self.writer.close()
                self.writer = None
                logger.info("已关闭TensorBoard writer")
            except Exception as e:
                logger.error(f"关闭TensorBoard writer时发生错误: {e}")

    def quantize_model(self):
        """量化模型以减少大小和提高推理速度"""
        try:
            logger.info("开始模型量化...")
            
            # 加载最佳模型
            model_path = os.path.join(self.data_dir, 'morse_model.pth')
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # 准备量化
            self.model.eval()
            
            # 对于静态量化，需要校准
            if hasattr(self.model, 'fuse_model'):
                self.model.fuse_model()
            
            # 动态量化比较简单，适用于RNN等模型
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
            
            # 保存量化模型
            quantized_model_path = os.path.join(self.data_dir, 'morse_model_quantized.pth')
            torch.save(quantized_model.state_dict(), quantized_model_path)
            
            # 替换当前模型
            self.model = quantized_model
            
            logger.info(f"模型量化完成，保存到 {quantized_model_path}")
            return True
        except Exception as e:
            logger.error(f"模型量化失败: {e}")
            return False

    def predict(self, signal_pattern):
        """使用模型预测数字"""
        try:
            # 确保模型处于评估模式
            self.model.eval()
            
            # 格式化信号模式
            if isinstance(signal_pattern, np.ndarray):
                # 调整大小至预期输入长度
                if len(signal_pattern) != self.input_size:
                    # 如果太短，用0填充
                    if len(signal_pattern) < self.input_size:
                        padded = np.zeros(self.input_size, dtype=np.float32)
                        padded[:len(signal_pattern)] = signal_pattern
                        signal_pattern = padded
                    # 如果太长，截断
                    else:
                        signal_pattern = signal_pattern[:self.input_size]
                
                # 转换为张量
                tensor = torch.tensor(signal_pattern, dtype=torch.float32).unsqueeze(0).to(self.device)
            elif isinstance(signal_pattern, str):
                # 摩尔斯码字符串转换为信号
                signal = self._morse_to_signal(signal_pattern)
                tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                logger.error(f"不支持的信号类型: {type(signal_pattern)}")
                return None, 0.0
            
            # 进行预测
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # 获取预测结果和置信度
                confidence, prediction = torch.max(probabilities, 1)
                
            return prediction.item(), confidence.item()
        
        except Exception as e:
            logger.error(f"预测时发生错误: {e}")
            return None, 0.0

    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(self.data_dir, 'morse_model.pth')
            
        torch.save(self.model.state_dict(), path)
        logger.info(f"模型保存到 {path}")
        
        # 导出ONNX格式，便于在其他平台使用
        try:
            dummy_input = torch.randn(1, self.input_size, device=self.device)
            onnx_path = os.path.join(self.data_dir, 'morse_model.onnx')
            torch.onnx.export(self.model, dummy_input, onnx_path, 
                              input_names=['input'], output_names=['output'])
            logger.info(f"导出ONNX模型到 {onnx_path}")
        except Exception as e:
            logger.warning(f"导出ONNX模型失败: {e}")

    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            # 尝试加载量化模型
            quant_path = os.path.join(self.data_dir, 'morse_model_quantized.pth')
            regular_path = os.path.join(self.data_dir, 'morse_model.pth')
            
            if self.quantize and os.path.exists(quant_path):
                path = quant_path
                logger.info("加载量化模型")
            elif os.path.exists(regular_path):
                path = regular_path
                logger.info("加载常规模型")
            else:
                logger.warning("模型文件不存在")
                return False
                
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"模型从 {path} 加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_model_size(self):
        """获取模型大小信息"""
        try:
            # 计算参数数量
            params = sum(p.numel() for p in self.model.parameters())
            
            # 获取模型文件大小
            model_path = os.path.join(self.data_dir, 'morse_model.pth')
            quant_path = os.path.join(self.data_dir, 'morse_model_quantized.pth')
            
            model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0
            quant_size = os.path.getsize(quant_path) if os.path.exists(quant_path) else 0
            
            return {
                'params': params,
                'model_size_bytes': model_size,
                'quantized_size_bytes': quant_size,
                'size_reduction': (1 - quant_size/model_size) if (model_size and quant_size) else 0
            }
        except Exception as e:
            logger.error(f"获取模型大小信息失败: {e}")
            return None

    def __del__(self):
        """清理资源"""
        # 安全关闭TensorBoard writer
        self.close_writer() 