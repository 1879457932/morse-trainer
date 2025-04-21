import numpy as np
from datetime import datetime
import json
import os
from threading import Thread, Event
from queue import Queue
from collections import deque
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MorseDecoder')

class MorseDecoder:
    def __init__(self):
        # 摩尔斯电码查找表
        self.MORSE_CODE = {
            '-----': '0', '.----': '1', '..---': '2',
            '...--': '3', '....-': '4', '.....': '5',
            '-....': '6', '--...': '7', '---..': '8',
            '----.': '9'
        }
        
        # 创建反向查找表，用于错误纠正
        self.REVERSE_MORSE = {v: k for k, v in self.MORSE_CODE.items()}
        
        # 缓存所有可能的摩尔斯电码模式，用于快速查找
        self.all_patterns = list(self.MORSE_CODE.keys())
        
        # 自适应时间参数
        self.DOT_DURATION = 0.1  # 初始点持续时间
        self.DASH_DURATION = 0.3  # 初始划持续时间
        self.SIGNAL_GAP = 0.2  # 初始信号间隔
        self.ADAPTATION_RATE = 0.1  # 自适应率
        
        # 信号处理参数
        self.current_signal = []
        self.current_timing = []
        self.signal_buffer = []
        self.last_signal_time = None
        self.timing_history = deque(maxlen=10)  # 存储最近的时间参数
        
        # 错误纠正参数
        self.error_threshold = 0.2  # 错误阈值
        self.confidence_scores = {}  # 置信度分数
        
        # 创建字典树，用于快速模式匹配
        self.trie = self._build_trie()
        
        # 文件存储
        self.history_file = 'morse_history.json'
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.history_path = os.path.join(self.data_dir, self.history_file)
        
        # 解码队列和线程控制
        self.decode_queue = Queue(maxsize=100)
        self.result_queue = Queue(maxsize=100)
        self.is_running = Event()
        self.decode_thread = None
        
        # 性能指标
        self.total_processed = 0
        self.total_time = 0

    def start(self):
        """启动解码线程"""
        if not self.is_running.is_set():
            self.is_running.set()
            self.decode_thread = Thread(target=self._decode_loop)
            self.decode_thread.daemon = True
            self.decode_thread.start()
            logger.info("摩尔斯解码器已启动")

    def stop(self):
        """停止解码线程"""
        if self.is_running.is_set():
            self.is_running.clear()
            if self.decode_thread and self.decode_thread.is_alive():
                self.decode_thread.join(timeout=1.0)
            logger.info("摩尔斯解码器已停止")

    def _decode_loop(self):
        """解码循环线程"""
        while self.is_running.is_set():
            try:
                signal_data = self.decode_queue.get(timeout=0.1)
                start_time = datetime.now()
                
                self._process_signal(signal_data)
                
                # 计算处理时间
                process_time = (datetime.now() - start_time).total_seconds()
                self.total_time += process_time
                self.total_processed += 1
                
                self.decode_queue.task_done()
            except Queue.Empty:
                pass
            except Exception as e:
                logger.error(f"解码过程中发生错误: {e}")

    def _process_signal(self, signal_data):
        """处理信号数据"""
        try:
            signal_strength = signal_data.get('strength', 0)
            signal_time = signal_data.get('time', datetime.now())
            
            # 忽略过弱的信号
            if signal_strength < 0.1:
                return
                
            # 初始化上次信号时间
            if self.last_signal_time is None:
                self.last_signal_time = signal_time
                self.current_signal.append(signal_strength)
                self.current_timing.append(0)
                return
                
            # 计算时间差
            time_diff = (signal_time - self.last_signal_time).total_seconds()
            self.last_signal_time = signal_time
            
            # 判断是否为信号间隔
            if time_diff > self.SIGNAL_GAP:
                # 分析当前缓冲区中的信号
                if self.current_signal:
                    avg_strength = np.mean(self.current_signal)
                    avg_duration = np.sum(self.current_timing)
                    
                    # 根据持续时间和强度判断是点还是划
                    if avg_duration < self.DOT_DURATION * 1.5:
                        self.signal_buffer.append('.')
                    else:
                        self.signal_buffer.append('-')
                    
                    # 自适应调整时间参数
                    self._adapt_timing_parameters(avg_duration)
                
                # 如果时间差足够大，认为是字符结束
                if time_diff > self.SIGNAL_GAP * 3:
                    self._try_decode_sequence()
                
                # 重置当前信号缓冲区
                self.current_signal = [signal_strength]
                self.current_timing = [0]
            else:
                # 继续累积当前信号
                self.current_signal.append(signal_strength)
                self.current_timing.append(time_diff)
        except Exception as e:
            logger.error(f"处理信号时发生错误: {e}")

    def _analyze_signal_buffer(self):
        """分析信号缓冲区"""
        if not self.signal_buffer:
            return None
            
        # 将信号缓冲区合并为摩尔斯电码序列
        morse_code = ''.join(self.signal_buffer)
        
        # 重置信号缓冲区
        self.signal_buffer = []
        
        return morse_code

    def _adapt_timing_parameters(self, signal_duration):
        """自适应调整时间参数"""
        # 存储到历史记录
        self.timing_history.append(signal_duration)
        
        # 至少需要5个样本才能调整
        if len(self.timing_history) < 5:
            return
            
        # 计算信号持续时间的分布
        durations = np.array(self.timing_history)
        # 使用K-means聚类来自动区分点和划
        from sklearn.cluster import KMeans
        
        # 预防异常情况
        try:
            kmeans = KMeans(n_clusters=2, random_state=0).fit(durations.reshape(-1, 1))
            centers = kmeans.cluster_centers_.flatten()
            
            # 确定哪个是点，哪个是划
            dot_center = min(centers)
            dash_center = max(centers)
            
            # 更新时间参数
            self.DOT_DURATION = dot_center
            self.DASH_DURATION = dash_center
            self.SIGNAL_GAP = (dot_center + dash_center) / 2
        except Exception as e:
            logger.warning(f"调整时间参数时发生错误: {e}")

    def _try_decode_sequence(self):
        """尝试解码当前序列"""
        morse_code = self._analyze_signal_buffer()
        
        if not morse_code:
            return
            
        # 尝试错误纠正
        corrected_code = self._correct_errors(morse_code)
        
        # 计算置信度
        confidence = self._calculate_confidence(corrected_code)
        
        # 查找对应的数字
        decoded_digit = self.MORSE_CODE.get(corrected_code, None)
        
        if decoded_digit:
            # 保存到历史记录
            self._save_to_history(corrected_code, decoded_digit)
            
            # 放入结果队列
            result = {
                'morse_code': corrected_code,
                'decoded': decoded_digit,
                'confidence': confidence,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            }
            
            self.result_queue.put(result)
            
        return decoded_digit, confidence

    def _build_trie(self):
        """构建字典树用于快速模式匹配"""
        trie = {}
        for code, digit in self.MORSE_CODE.items():
            node = trie
            for char in code:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = digit
        return trie

    def _search_trie(self, code, max_errors=2):
        """在字典树中搜索最佳匹配"""
        results = []
        
        def dfs(node, path, remaining_errors, index=0):
            # 如果到达结尾
            if index == len(code):
                if '$' in node:
                    # 找到匹配，记录路径和对应的数字
                    distance = max_errors - remaining_errors
                    results.append((path, node['$'], distance))
                return
                
            # 当前字符
            char = code[index]
            
            # 精确匹配
            if char in node:
                dfs(node[char], path + char, remaining_errors, index + 1)
                
            # 允许错误匹配
            if remaining_errors > 0:
                # 替换错误
                for next_char in ['.', '-']:
                    if next_char != char and next_char in node:
                        dfs(node[next_char], path + next_char, remaining_errors - 1, index + 1)
                
                # 插入错误
                dfs(node, path, remaining_errors - 1, index + 1)
                
                # 删除错误
                for next_char in ['.', '-']:
                    if next_char in node:
                        dfs(node[next_char], path + next_char, remaining_errors - 1, index)
        
        # 开始搜索
        dfs(self.trie, "", max_errors, 0)
        
        # 按错误距离排序
        results.sort(key=lambda x: x[2])
        
        return results[0][0] if results else code

    def _correct_errors(self, morse_code):
        """错误纠正"""
        # 如果代码完全匹配，直接返回
        if morse_code in self.MORSE_CODE:
            return morse_code
            
        # 查找最相似的代码
        min_distance = float('inf')
        best_match = morse_code
        
        for pattern in self.all_patterns:
            distance = self._levenshtein_distance(morse_code, pattern)
            if distance < min_distance:
                min_distance = distance
                best_match = pattern
                
        # 只在错误在可接受范围内时进行纠正
        if min_distance <= len(morse_code) * self.error_threshold:
            return best_match
            
        return morse_code

    def _levenshtein_distance(self, s1, s2):
        """计算Levenshtein距离"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 计算插入、删除和替换的代价
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                
                # 选择最小代价操作
                current_row.append(min(insertions, deletions, substitutions))
                
            previous_row = current_row
            
        return previous_row[-1]

    def _calculate_confidence(self, morse_code):
        """计算解码置信度"""
        # 如果代码完全匹配，置信度为1
        if morse_code in self.MORSE_CODE:
            return 1.0
            
        # 计算与最近模式的相似度
        min_distance = float('inf')
        for pattern in self.all_patterns:
            distance = self._levenshtein_distance(morse_code, pattern)
            min_distance = min(min_distance, distance)
            
        # 计算置信度，距离越小置信度越高
        max_possible_distance = max(len(morse_code), max(len(p) for p in self.all_patterns))
        confidence = 1.0 - (min_distance / max_possible_distance)
        
        return max(0.0, confidence)

    def process_audio_chunk(self, audio_data):
        """处理音频数据块"""
        # 提取信号强度
        signal_strength = np.max(np.abs(audio_data))
        
        # 放入解码队列
        self.decode_queue.put({
            'strength': signal_strength,
            'time': datetime.now()
        })

    def get_decode_result(self):
        """获取解码结果"""
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _save_to_history(self, morse_code, decoded_digit):
        """保存到历史记录"""
        try:
            # 读取现有历史记录
            history_data = []
            if os.path.exists(self.history_path):
                try:
                    with open(self.history_path, 'r') as f:
                        history_data = json.load(f)
                except json.JSONDecodeError:
                    logger.warning("历史文件格式错误，创建新文件")
                    history_data = []
                        
            # 添加新记录
            history_data.append({
                'morse_code': morse_code,
                'decoded': decoded_digit,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            # 限制历史记录大小
            if len(history_data) > 1000:
                history_data = history_data[-1000:]
                
            # 保存到文件
            with open(self.history_path, 'w') as f:
                json.dump(history_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"保存历史记录时发生错误: {e}")

    def reset(self):
        """重置解码器状态"""
        self.current_signal = []
        self.current_timing = []
        self.signal_buffer = []
        self.last_signal_time = None
        
        # 清空队列
        while not self.decode_queue.empty():
            self.decode_queue.get()
            self.decode_queue.task_done()
            
        while not self.result_queue.empty():
            self.result_queue.get()
            self.result_queue.task_done()
            
    def get_performance_metrics(self):
        """获取性能指标"""
        if self.total_processed == 0:
            return {
                'avg_process_time': 0,
                'total_processed': 0
            }
            
        avg_time = self.total_time / self.total_processed
        
        return {
            'avg_process_time': avg_time,
            'total_processed': self.total_processed
        }
        
    def __del__(self):
        """清理资源"""
        self.stop()