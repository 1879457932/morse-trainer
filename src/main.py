"""
主程序入口文件
"""

import os
import threading
import asyncio
from datetime import datetime
import numpy as np
from functools import partial
import time
import logging

from kivy.lang import Builder
from kivy.clock import Clock
from kivy.garden.graph import Graph, MeshLinePlot
from kivy.properties import StringProperty, BooleanProperty, NumericProperty, ListProperty, ObjectProperty
from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivy.graphics import Color, Rectangle
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivymd.uix.slider import MDSlider
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.label import MDLabel
from kivy.config import Config
from kivy.core.window import Window
from morse_decoder import MorseDecoder
from morse_ai import MorseAI
from audio_processor import AudioProcessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('MorseTrainer')

# 配置悬浮窗模式
Config.set('graphics', 'borderless', '1')
Config.set('graphics', 'resizable', '0')
Config.set('graphics', 'always_on_top', '1')

# 检查是否在Android平台上运行
try:
    from android.permissions import request_permissions, check_permission, Permission
    from service.foreground_service import ForegroundService
    IS_ANDROID = True
    logger.info("在Android平台上运行")
except ImportError:
    IS_ANDROID = False
    logger.info("不在Android平台上运行")

# 音频设置
CHUNK = 1024
RATE = 44100
CHANNELS = 1
WAVE_SAMPLES = 100

# 全局UI更新队列
ui_update_queue = asyncio.Queue(maxsize=10)

class GraphCanvas(Widget):
    """波形图显示组件"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.graph = Graph(
            xlabel='Time',
            ylabel='Amplitude',
            x_ticks_minor=5,
            x_ticks_major=25,
            y_ticks_major=0.5,
            y_grid_label=True,
            x_grid_label=True,
            padding=5,
            x_grid=True,
            y_grid=True,
            xmin=0,
            xmax=WAVE_SAMPLES,
            ymin=-1,
            ymax=1
        )
        self.plot = MeshLinePlot(color=[0, 0.7, 0, 1])
        self.graph.add_plot(self.plot)
        self.add_widget(self.graph)
        
        # 使用缓冲区减少刷新
        self.points_buffer = [(i, 0) for i in range(WAVE_SAMPLES)]
        self.last_update_time = time.time()
        self.update_interval = 0.05  # 50ms刷新率

    def update_plot(self, points):
        """更新波形图"""
        # 减少刷新频率以提高性能
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
            
        # 更新点坐标
        for i, p in enumerate(points[:WAVE_SAMPLES]):
            if i < len(self.points_buffer):
                self.points_buffer[i] = (i, p)
                
        # 更新绘图
        self.plot.points = self.points_buffer
        self.last_update_time = current_time

class FrequencyControl(MDBoxLayout):
    """频率控制面板"""
    target_frequency = NumericProperty(1000)
    frequency_bandwidth = NumericProperty(200)
    frequency_sensitivity = NumericProperty(1.0)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = 10
        self.padding = 10
        
        # 目标频率控制
        self.freq_label = MDLabel(text="目标频率: 1000 Hz")
        self.freq_slider = MDSlider(
            min=20,
            max=20000,
            value=1000,
            step=10
        )
        self.freq_slider.bind(value=self.on_freq_change)
        
        # 带宽控制
        self.bandwidth_label = MDLabel(text="带宽: 200 Hz")
        self.bandwidth_slider = MDSlider(
            min=10,
            max=1000,
            value=200,
            step=10
        )
        self.bandwidth_slider.bind(value=self.on_bandwidth_change)
        
        # 灵敏度控制
        self.sensitivity_label = MDLabel(text="灵敏度: 1.0")
        self.sensitivity_slider = MDSlider(
            min=0.1,
            max=2.0,
            value=1.0,
            step=0.1
        )
        self.sensitivity_slider.bind(value=self.on_sensitivity_change)
        
        # 添加组件
        self.add_widget(self.freq_label)
        self.add_widget(self.freq_slider)
        self.add_widget(self.bandwidth_label)
        self.add_widget(self.bandwidth_slider)
        self.add_widget(self.sensitivity_label)
        self.add_widget(self.sensitivity_slider)

    def on_freq_change(self, instance, value):
        """频率变化回调"""
        self.freq_label.text = f"目标频率: {int(value)} Hz"
        self.target_frequency = value

    def on_bandwidth_change(self, instance, value):
        """带宽变化回调"""
        self.bandwidth_label.text = f"带宽: {int(value)} Hz"
        self.frequency_bandwidth = value

    def on_sensitivity_change(self, instance, value):
        """灵敏度变化回调"""
        self.sensitivity_label.text = f"灵敏度: {value:.1f}"
        self.frequency_sensitivity = value

class MainScreen(MDScreen):
    """主界面"""
    # UI状态属性
    is_recording = BooleanProperty(False)
    current_sequence = StringProperty("")
    decoded_result = StringProperty("")
    ai_prediction = StringProperty("")
    noise_level = StringProperty("噪声水平: 正常")
    permission_granted = BooleanProperty(False)
    auto_adjust = BooleanProperty(False)
    current_frequency = StringProperty("当前频率: -- Hz")
    frequency_adjustment = StringProperty("")
    accuracy = NumericProperty(0.0)
    total_samples = NumericProperty(0)
    correct_samples = NumericProperty(0)
    is_floating = BooleanProperty(False)
    
    # 频率控制参数
    target_frequency = NumericProperty(1000)
    frequency_bandwidth = NumericProperty(200)
    frequency_sensitivity = NumericProperty(1.0)
    
    # 组件引用
    frequency_control = ObjectProperty(None)
    wave_graph = ObjectProperty(None)
    history_list = ObjectProperty(None)
    
    # 数据缓冲
    wave_data = ListProperty([0] * WAVE_SAMPLES)
    history_data = ListProperty([])
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.app = MDApp.get_running_app()
        self.audio_processor = None
        self.morse_decoder = None
        self.morse_ai = None
        
        # 资源管理标志
        self._resources_initialized = False
        
        # 窗口拖动变量
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        # 异步更新标志
        self.ui_update_scheduled = False
        
        # 设置监听器以响应属性变化
        self.bind(
            target_frequency=self.on_frequency_params_change,
            frequency_bandwidth=self.on_frequency_params_change,
            frequency_sensitivity=self.on_frequency_params_change
        )
        
        # 在Android上请求权限
        if IS_ANDROID:
            Clock.schedule_once(lambda dt: self.check_permissions(), 0.5)
        else:
            self.permission_granted = True
            Clock.schedule_once(lambda dt: self.initialize_components(), 1)
            
        # 设置UI更新调度器
        Clock.schedule_interval(self.process_ui_updates, 1/30)  # 30 FPS

    def on_touch_down(self, touch):
        """触摸开始事件处理"""
        if self.is_floating and self.collide_point(*touch.pos):
            self.drag_start_x = touch.x
            self.drag_start_y = touch.y
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """触摸移动事件处理"""
        if self.is_floating and hasattr(self, 'drag_start_x'):
            # 计算位移
            dx = touch.x - self.drag_start_x
            dy = touch.y - self.drag_start_y
            
            # 更新窗口位置
            Window.left += dx
            Window.top += dy
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """触摸结束事件处理"""
        if self.is_floating:
            self.drag_start_x = 0
            self.drag_start_y = 0
            return True
        return super().on_touch_up(touch)

    def toggle_floating(self):
        """切换悬浮窗模式"""
        if not self.is_floating:
            # 进入悬浮窗模式
            self._original_size = Window.size
            self.is_floating = True
            Window.size = (400, 300)
            Window.borderless = True
            Window.always_on_top = True
        else:
            # 退出悬浮窗模式
            self.is_floating = False
            Window.size = self._original_size
            Window.borderless = False
            Window.always_on_top = False

    def check_permissions(self):
        """检查并请求权限（Android平台）"""
        if IS_ANDROID:
            if check_permission(Permission.RECORD_AUDIO):
                self.permission_granted = True
                self.initialize_components()
            else:
                # 请求麦克风权限
                request_permissions(
                    [Permission.RECORD_AUDIO], 
                    self.permission_callback
                )
        else:
            # 非Android平台默认有权限
            self.permission_granted = True
            self.initialize_components()

    def permission_callback(self, permissions, grants):
        """权限请求回调"""
        if all(grants):
            self.permission_granted = True
            # 在主线程上初始化组件
            Clock.schedule_once(lambda dt: self.initialize_components(), 0.1)
        else:
            # 权限被拒绝
            self.permission_granted = False
            self.current_sequence = "需要麦克风权限"
            # 尝试再次请求权限
            Clock.schedule_once(lambda dt: self.check_permissions(), 3)

    def initialize_components(self):
        """初始化应用组件"""
        if self._resources_initialized:
            return
            
        logger.info("正在初始化组件...")
        
        # 在单独的线程中初始化以避免UI阻塞
        threading.Thread(target=self._init_resources, daemon=True).start()

    def _init_resources(self):
        """初始化资源（在后台线程中运行）"""
        try:
            # 初始化音频处理器
            self.audio_processor = AudioProcessor(
                chunk_size=CHUNK,
                rate=RATE,
                channels=CHANNELS
            )
            
            # 初始化摩尔斯解码器
            self.morse_decoder = MorseDecoder()
            
            # 初始化AI模型（使用量化版本以提高性能）
            self.morse_ai = MorseAI(quantize=True)
            self.morse_ai.load_model()
            
            # 标记资源已初始化
            self._resources_initialized = True
            
            # 在主线程中更新UI
            Clock.schedule_once(lambda dt: self._post_init_ui(), 0)
            
            logger.info("组件初始化完成")
        except Exception as e:
            logger.error(f"初始化组件时出错: {e}")
            # 在主线程中显示错误
            Clock.schedule_once(lambda dt: setattr(self, 'current_sequence', f"初始化失败: {e}"), 0)

    def _post_init_ui(self):
        """初始化完成后更新UI"""
        # 设置初始频率参数
        if self.audio_processor:
            self.audio_processor.set_target_frequency(self.target_frequency)
            self.audio_processor.set_frequency_bandwidth(self.frequency_bandwidth)
            self.audio_processor.set_frequency_sensitivity(self.frequency_sensitivity)
            
        # 更新UI状态
        self.current_sequence = "准备就绪"
        
        # 加载历史记录
        self.load_history()

    def on_frequency_params_change(self, instance, value):
        """频率参数变化响应"""
        if self.audio_processor:
            self.audio_processor.set_target_frequency(self.target_frequency)
            self.audio_processor.set_frequency_bandwidth(self.frequency_bandwidth)
            self.audio_processor.set_frequency_sensitivity(self.frequency_sensitivity)

    def toggle_recording(self):
        """切换录音状态"""
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        """开始录音"""
        if not self._resources_initialized:
            self.current_sequence = "组件未初始化，请稍候"
            return
            
        logger.info("开始录音")
        
        # 启动前台服务（Android）
        if IS_ANDROID:
            self.service = ForegroundService()
            self.service.start('摩尔斯电码训练器正在运行')
            
        # 启动音频处理器
        if self.audio_processor:
            self.audio_processor.start()
            
        # 启动摩尔斯解码器
        if self.morse_decoder:
            self.morse_decoder.start()
            
        # 更新UI状态
        self.is_recording = True
        self.current_sequence = "正在监听..."
        
        # 启动UI更新
        Clock.schedule_interval(self.update_ui, 1/30)  # 30 FPS

    def stop_recording(self):
        """停止录音"""
        logger.info("停止录音")
        
        # 停止UI更新
        Clock.unschedule(self.update_ui)
        
        # 停止摩尔斯解码器
        if self.morse_decoder:
            self.morse_decoder.stop()
            
        # 停止音频处理器
        if self.audio_processor:
            self.audio_processor.stop()
            
        # 停止前台服务（Android）
        if IS_ANDROID and hasattr(self, 'service'):
            self.service.stop()
            
        # 更新UI状态
        self.is_recording = False
        self.current_sequence = "已停止"

    async def _update_ui_async(self):
        """异步UI更新（减少主线程阻塞）"""
        if not self._resources_initialized:
            return
            
        try:
            # 获取音频处理数据
            if self.audio_processor:
                data = self.audio_processor.get_processed_data()
                if data:
                    # 更新波形显示
                    if 'thresholded' in data:
                        self.wave_data = data['thresholded'][:WAVE_SAMPLES].tolist()
                        
                    # 更新噪声级别
                    self.noise_level = self.audio_processor.get_noise_level()
                    
                    # 更新频率信息
                    freq_info = self.audio_processor.get_frequency_info()
                    if freq_info:
                        self.current_frequency = f"当前频率: {int(freq_info['current'])} Hz"
                        
                        # 显示频率差异
                        diff = freq_info['diff']
                        if abs(diff) > 10:
                            direction = "高" if diff > 0 else "低"
                            self.frequency_adjustment = f"频率{direction}了 {abs(int(diff))} Hz"
                        else:
                            self.frequency_adjustment = "频率正常"
                            
                        # 自动调整频率
                        if self.auto_adjust:
                            if self.audio_processor.auto_adjust_frequency():
                                self.target_frequency = self.audio_processor.target_frequency
                        
                    # 将数据发送到摩尔斯解码器
                    if self.morse_decoder and 'thresholded' in data:
                        self.morse_decoder.process_audio_chunk(data['thresholded'])
            
            # 获取解码结果
            if self.morse_decoder:
                result = self.morse_decoder.get_decode_result()
                if result:
                    morse_code = result.get('morse_code', '')
                    decoded = result.get('decoded', '')
                    confidence = result.get('confidence', 0.0)
                    
                    # 更新当前序列
                    self.current_sequence = morse_code
                    
                    # 更新解码结果
                    self.decoded_result = f"解码: {decoded} (置信度: {confidence:.2f})"
                    
                    # 使用AI进行预测
                    if self.morse_ai:
                        prediction, ai_confidence = self.morse_ai.predict(morse_code)
                        if prediction is not None:
                            self.ai_prediction = f"AI预测: {prediction} (置信度: {ai_confidence:.2f})"
                            
                            # 添加到历史记录
                            self.add_to_history(morse_code, decoded, prediction, ai_confidence)
        except Exception as e:
            logger.error(f"更新UI时发生错误: {e}")

    def update_ui(self, dt):
        """UI更新回调（由Clock调度）"""
        if not self.ui_update_scheduled:
            self.ui_update_scheduled = True
            # 使用asyncio更新UI
            asyncio.create_task(self._update_ui_async())
            
    async def _queue_ui_update(self, update_func, *args, **kwargs):
        """将UI更新加入队列"""
        await ui_update_queue.put((update_func, args, kwargs))
        
    def process_ui_updates(self, dt):
        """处理UI更新队列"""
        if self.ui_update_scheduled:
            self.ui_update_scheduled = False
            # 检查波形数据并更新绘图
            if hasattr(self, 'wave_graph') and self.wave_graph:
                self.wave_graph.update_plot(self.wave_data)

    def add_to_history(self, morse_code, decoded_digit, ai_prediction, confidence):
        """添加到历史记录"""
        # 创建新的历史记录条目
        entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'morse_code': morse_code,
            'decoded': decoded_digit,
            'prediction': str(ai_prediction),
            'confidence': confidence,
            'is_correct': decoded_digit == str(ai_prediction)
        }
        
        # 将条目添加到历史数据中
        self.history_data.append(entry)
        
        # 限制历史记录长度
        if len(self.history_data) > 20:
            self.history_data = self.history_data[-20:]
            
        # 更新准确率统计
        self.total_samples += 1
        if entry['is_correct']:
            self.correct_samples += 1
            
        if self.total_samples > 0:
            self.accuracy = self.correct_samples / self.total_samples
            
        # 更新UI（添加到历史列表）
        from kivymd.uix.list import OneLineListItem
        item = OneLineListItem(
            text=f"{entry['timestamp']} - {morse_code} -> {decoded_digit} (AI: {ai_prediction})"
        )
        self.ids.history_list.add_widget(item)

    def mark_result(self, index, is_correct):
        """标记结果为正确或错误"""
        if 0 <= index < len(self.history_data):
            entry = self.history_data[index]
            
            # 更新正确性标志
            old_is_correct = entry['is_correct']
            entry['is_correct'] = is_correct
            
            # 更新准确率统计
            if old_is_correct != is_correct:
                if is_correct:
                    self.correct_samples += 1
                else:
                    self.correct_samples -= 1
                    
            if self.total_samples > 0:
                self.accuracy = self.correct_samples / self.total_samples

    def toggle_auto_adjust(self):
        """切换自动频率调整"""
        self.auto_adjust = not self.auto_adjust
        if self.auto_adjust:
            self.frequency_adjustment = "自动调整已启用"
        else:
            self.frequency_adjustment = "自动调整已禁用"

    def clear_history(self):
        """清除历史记录"""
        self.history_data = []
        self.ids.history_list.clear_widgets()
        self.total_samples = 0
        self.correct_samples = 0
        self.accuracy = 0.0

    def train_ai(self):
        """训练AI模型"""
        if self.morse_ai:
            # 显示训练状态
            self.current_sequence = "正在训练AI..."
            
            # 在后台线程训练AI
            threading.Thread(target=self._train_ai_thread, daemon=True).start()

    def _train_ai_thread(self):
        """AI训练线程"""
        try:
            # 训练AI模型
            success = self.morse_ai.train(epochs=50, batch_size=32)
            
            # 更新UI
            if success:
                Clock.schedule_once(lambda dt: setattr(self, 'current_sequence', "AI训练完成"), 0)
            else:
                Clock.schedule_once(lambda dt: setattr(self, 'current_sequence', "AI训练失败，数据不足"), 0)
        except Exception as e:
            logger.error(f"训练AI时发生错误: {e}")
            Clock.schedule_once(lambda dt: setattr(self, 'current_sequence', f"AI训练错误: {e}"), 0)

    def load_history(self):
        """加载历史记录"""
        if self.morse_decoder:
            try:
                with open(self.morse_decoder.history_path, 'r') as f:
                    import json
                    data = json.load(f)
                    
                    # 处理最近的20条记录
                    for entry in data[-20:]:
                        # 创建列表项
                        from kivymd.uix.list import OneLineListItem
                        morse_code = entry.get('morse_code', '')
                        decoded = entry.get('decoded', '')
                        item = OneLineListItem(
                            text=f"{entry.get('timestamp', '')} - {morse_code} -> {decoded}"
                        )
                        self.ids.history_list.add_widget(item)
            except Exception as e:
                logger.warning(f"加载历史记录失败: {e}")

    def export_data(self):
        """导出数据"""
        if self.morse_decoder and hasattr(self.morse_decoder, 'history_path'):
            # 显示导出状态
            source_path = self.morse_decoder.history_path
            
            if IS_ANDROID:
                # 在Android上复制到下载目录
                from android.storage import primary_external_storage_path
                import shutil
                
                dest_path = os.path.join(primary_external_storage_path(), 'Download', 'morse_history.json')
                try:
                    shutil.copy2(source_path, dest_path)
                    self.current_sequence = f"已导出到: {dest_path}"
                except Exception as e:
                    self.current_sequence = f"导出失败: {e}"
            else:
                # 在桌面平台上直接显示文件路径
                self.current_sequence = f"数据文件位置: {source_path}"

    def on_stop(self):
        """应用停止时的清理工作"""
        # 停止录音
        if self.is_recording:
            self.stop_recording()
            
        # 清理资源
        if self.audio_processor:
            del self.audio_processor
            
        if self.morse_decoder:
            del self.morse_decoder
            
        if self.morse_ai:
            del self.morse_ai

class MorseTrainerApp(MDApp):
    """摩尔斯电码训练应用"""
    def build(self):
        """构建应用"""
        self.theme_cls.primary_palette = "Green"
        self.theme_cls.accent_palette = "Blue"
        self.theme_cls.theme_style = "Dark"
        
        # 加载KV文件
        Builder.load_file('morse.kv')
        
        # 创建主屏幕
        return MainScreen()
        
    def on_start(self):
        """应用启动"""
        logger.info("应用已启动")
        
    def on_stop(self):
        """应用停止"""
        logger.info("应用已停止")
        screen = self.root
        if hasattr(screen, 'on_stop'):
            screen.on_stop()

def main():
    """应用入口点"""
    # 启动asyncio事件循环
    loop = asyncio.get_event_loop()
    
    try:
        app = MorseTrainerApp()
        app.run()
    except Exception as e:
        logger.error(f"应用运行错误: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    main() 