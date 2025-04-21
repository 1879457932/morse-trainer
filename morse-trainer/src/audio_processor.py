import numpy as np
import pyaudio
from scipy import signal
from threading import Thread, Event
import time
from scipy.fft import rfft, rfftfreq
from collections import deque

class AudioProcessor:
    def __init__(self, chunk_size=1024, rate=44100, channels=1):
        self.chunk_size = chunk_size
        self.rate = rate
        self.channels = channels
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_running = Event()
        self.processed_data = None
        self.noise_floor = 0.0
        self.adaptive_threshold = 0.1
        self.MAX_HISTORY = 50
        
        # 优化：预分配缓冲区，避免频繁内存分配
        self.buffer = np.zeros(chunk_size * 2, dtype=np.float32)
        self.history_buffer = deque(maxlen=self.MAX_HISTORY)
        
        # 频率调节参数
        self.target_frequency = 1000  # 默认目标频率
        self.frequency_bandwidth = 200  # 频率带宽
        self.frequency_sensitivity = 1.0  # 频率灵敏度
        
        # 频率分析参数
        self.frequency_history = deque(maxlen=100)  # 改用deque存储检测到的频率历史
        self.frequency_window = 10  # 频率分析窗口大小
        self.min_frequency = 20  # 最小频率
        self.max_frequency = 20000  # 最大频率
        
        # 滤波器参数
        self.nyquist = rate / 2
        self.low_cut = self.target_frequency - self.frequency_bandwidth
        self.high_cut = self.target_frequency + self.frequency_bandwidth
        self.filter_order = 4
        
        # 预计算窗口函数以加速FFT
        self.window = np.hanning(chunk_size)
        
        # 预计算rfft的频率轴，避免重复计算
        self.fft_freqs = rfftfreq(chunk_size, 1.0/rate)
        
        # 创建带通滤波器
        self._update_filter()
        
    def _update_filter(self):
        """更新滤波器参数"""
        self.low_cut = max(20, self.target_frequency - self.frequency_bandwidth)
        self.high_cut = min(self.nyquist, self.target_frequency + self.frequency_bandwidth)
        
        # 重新创建带通滤波器
        self.b, self.a = signal.butter(
            self.filter_order,
            [self.low_cut / self.nyquist, self.high_cut / self.nyquist],
            btype='band'
        )

    def set_target_frequency(self, frequency):
        """设置目标频率"""
        self.target_frequency = max(self.min_frequency, min(self.max_frequency, frequency))
        self._update_filter()
        
    def set_frequency_bandwidth(self, bandwidth):
        """设置频率带宽"""
        self.frequency_bandwidth = max(10, min(1000, bandwidth))
        self._update_filter()
        
    def set_frequency_sensitivity(self, sensitivity):
        """设置频率灵敏度"""
        self.frequency_sensitivity = max(0.1, min(2.0, sensitivity))
        
    def start(self):
        """启动音频处理线程"""
        if not self.is_running.is_set():
            self.is_running.set()
            
            # 打开音频流
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,  # 使用32位浮点格式提高精度
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
    def stop(self):
        """停止音频处理线程"""
        if self.is_running.is_set():
            self.is_running.clear()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
    def _analyze_frequency(self, audio_data):
        """使用优化的FFT分析频率"""
        # 应用窗口函数减少频谱泄漏
        windowed_data = audio_data * self.window
        
        # 使用rfft代替fft，仅计算实部，减少计算量一半
        fft_data = np.abs(rfft(windowed_data))
        
        # 使用预计算的频率轴
        freqs = self.fft_freqs
        
        # 找出带通滤波器范围内的最大频率
        mask = (freqs >= self.low_cut) & (freqs <= self.high_cut)
        if np.any(mask):
            # 向量化操作，避免循环
            filtered_spectrum = fft_data * mask
            max_idx = np.argmax(filtered_spectrum)
            max_freq = freqs[max_idx]
            max_amp = filtered_spectrum[max_idx]
            
            # 归一化振幅
            relative_amp = max_amp / np.sum(fft_data)
            
            # 只有当相对振幅超过阈值时才记录频率
            if relative_amp > 0.05 * self.frequency_sensitivity:
                self.frequency_history.append(max_freq)
                return max_freq, relative_amp
        
        return None, 0.0
                
    def auto_adjust_frequency(self):
        """自动调整频率"""
        if len(self.frequency_history) < self.frequency_window:
            return False
            
        # 使用向量化操作计算平均频率
        recent_freqs = np.array(list(self.frequency_history)[-self.frequency_window:])
        # 使用中位数而不是平均值，对异常值更鲁棒
        median_freq = np.median(recent_freqs)
        
        # 如果中位数频率与当前目标频率差距超过带宽的一半，则调整
        if abs(median_freq - self.target_frequency) > self.frequency_bandwidth / 2:
            self.target_frequency = median_freq
            self._update_filter()
            return True
            
        return False
            
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数"""
        if status:
            print(f"音频回调状态: {status}")
            
        if not self.is_running.is_set():
            return None, pyaudio.paComplete
            
        # 将字节数据转换为numpy数组 - 使用float32提高效率
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        
        # 更新自适应噪声阈值
        self._update_noise_floor(audio_data)
        
        # 应用带通滤波器 - 使用scipy的filtfilt替代自己实现的滤波
        filtered_data = signal.filtfilt(self.b, self.a, audio_data)
        
        # 自适应阈值处理
        thresholded_data = self._adaptive_thresholding(filtered_data)
        
        # 频率分析
        freq, amp = self._analyze_frequency(audio_data)
        
        # 保存处理后的数据
        self.processed_data = {
            'raw': audio_data,
            'filtered': filtered_data, 
            'thresholded': thresholded_data,
            'frequency': freq,
            'amplitude': amp
        }
        
        # 保存到历史缓冲区
        self.history_buffer.append(self.processed_data)
        
        return in_data, pyaudio.paContinue
        
    def _update_noise_floor(self, data):
        """更新噪声基准"""
        # 使用向量化操作计算RMS值
        rms = np.sqrt(np.mean(np.square(data)))
        # 使用指数移动平均更新噪声基准
        self.noise_floor = 0.9 * self.noise_floor + 0.1 * rms
        
    def _adaptive_thresholding(self, data):
        """自适应阈值处理"""
        # 计算自适应阈值
        threshold = self.noise_floor * 2.0 * self.frequency_sensitivity
        
        # 使用向量化操作应用阈值
        result = np.where(np.abs(data) > threshold, data, 0)
        
        # 缩放到[-1, 1]范围
        if np.max(np.abs(result)) > 0:
            result = result / np.max(np.abs(result))
            
        return result
        
    def get_processed_data(self):
        """获取处理后的数据"""
        return self.processed_data
        
    def get_noise_level(self):
        """获取噪声级别"""
        if self.noise_floor < 0.01:
            return "噪声水平: 极低"
        elif self.noise_floor < 0.05:
            return "噪声水平: 正常"
        elif self.noise_floor < 0.1:
            return "噪声水平: 较高"
        else:
            return "噪声水平: 很高"
            
    def get_frequency_info(self):
        """获取频率信息"""
        if self.processed_data and 'frequency' in self.processed_data and self.processed_data['frequency']:
            current_freq = self.processed_data['frequency']
            diff = current_freq - self.target_frequency
            
            freq_info = {
                'current': current_freq,
                'target': self.target_frequency,
                'diff': diff,
                'in_range': abs(diff) <= self.frequency_bandwidth
            }
            
            return freq_info
            
        return None
        
    def adjust_sensitivity(self, value):
        """调整灵敏度"""
        self.frequency_sensitivity = max(0.1, min(2.0, value))
        
    def __del__(self):
        """析构函数"""
        self.stop()
        if self.audio:
            self.audio.terminate()