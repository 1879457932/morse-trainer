from jnius import autoclass, cast
from android.runnable import run_on_ui_thread
import logging

# 设置日志
logger = logging.getLogger('ForegroundService')

# Android类
Context = autoclass('android.content.Context')
NotificationManager = autoclass('android.app.NotificationManager')
NotificationChannel = autoclass('android.app.NotificationChannel')
NotificationCompat = autoclass('androidx.core.app.NotificationCompat')
PendingIntent = autoclass('android.app.PendingIntent')
Intent = autoclass('android.content.Intent')
String = autoclass('java.lang.String')
PythonActivity = autoclass('org.kivy.android.PythonActivity')
PowerManager = autoclass('android.os.PowerManager')
Build = autoclass('android.os.Build')
Uri = autoclass('android.net.Uri')
Settings = autoclass('android.provider.Settings')

class ForegroundService:
    """Android前台服务管理类"""
    
    def __init__(self):
        self.service_active = False
        self.notification_id = 1
        self.channel_id = "morse_trainer_channel"
        self.notification = None
        self.wakelock = None
        
        # 创建通知通道
        self._create_notification_channel()
        
        # 获取电源服务
        self._get_power_service()
    
    @run_on_ui_thread
    def _create_notification_channel(self):
        """创建通知通道（Android 8.0及以上需要）"""
        try:
            activity = PythonActivity.mActivity
            if not activity:
                logger.error("无法获取Activity")
                return
                
            notification_manager = activity.getSystemService(Context.NOTIFICATION_SERVICE)
            
            # 只需在Android 8.0及以上创建通知渠道
            if Build.VERSION.SDK_INT >= 26:  # Build.VERSION_CODES.O
                channel = NotificationChannel(
                    self.channel_id,
                    String('Morse Trainer Service'),
                    NotificationManager.IMPORTANCE_LOW
                )
                channel.setDescription('Morse code training service notification channel')
                channel.enableVibration(False)  # 禁用振动
                channel.setSound(None, None)  # 禁用声音
                notification_manager.createNotificationChannel(channel)
                logger.info("通知渠道已创建")
        except Exception as e:
            logger.error(f"创建通知渠道时出错: {e}")
    
    def _get_power_service(self):
        """获取电源管理服务"""
        try:
            activity = PythonActivity.mActivity
            if not activity:
                logger.error("无法获取Activity")
                return
                
            power_manager = activity.getSystemService(Context.POWER_SERVICE)
            self.power_manager = cast('android.os.PowerManager', power_manager)
            logger.info("电源管理服务已初始化")
        except Exception as e:
            logger.error(f"获取电源服务时出错: {e}")
            self.power_manager = None
    
    def _acquire_wakelock(self):
        """获取唤醒锁"""
        if self.power_manager:
            try:
                if self.wakelock and self.wakelock.isHeld():
                    self.wakelock.release()
                    
                # 创建部分唤醒锁，保持CPU运行但允许屏幕关闭
                self.wakelock = self.power_manager.newWakeLock(
                    PowerManager.PARTIAL_WAKE_LOCK,
                    String('MorseTrainerWakeLock')
                )
                self.wakelock.acquire()
                logger.info("已获取唤醒锁")
            except Exception as e:
                logger.error(f"获取唤醒锁时出错: {e}")
    
    def _release_wakelock(self):
        """释放唤醒锁"""
        if self.wakelock and self.wakelock.isHeld():
            try:
                self.wakelock.release()
                self.wakelock = None
                logger.info("已释放唤醒锁")
            except Exception as e:
                logger.error(f"释放唤醒锁时出错: {e}")
    
    def check_battery_optimization(self):
        """检查电池优化状态"""
        try:
            activity = PythonActivity.mActivity
            if not activity:
                return False
                
            if Build.VERSION.SDK_INT >= 23:  # Build.VERSION_CODES.M
                package_name = activity.getPackageName()
                is_ignored = self.power_manager.isIgnoringBatteryOptimizations(package_name)
                
                if not is_ignored:
                    logger.warning("应用不在电池优化白名单中")
                    return False
                    
                logger.info("应用已在电池优化白名单中")
                return True
        except Exception as e:
            logger.error(f"检查电池优化时出错: {e}")
            
        return False
    
    def request_ignore_battery_optimization(self):
        """请求忽略电池优化"""
        try:
            activity = PythonActivity.mActivity
            if not activity or Build.VERSION.SDK_INT < 23:
                return False
                
            package_name = activity.getPackageName()
            if not self.power_manager.isIgnoringBatteryOptimizations(package_name):
                intent = Intent()
                intent.setAction(Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS)
                intent.setData(Uri.parse("package:" + package_name))
                activity.startActivity(intent)
                logger.info("已请求忽略电池优化")
                return True
        except Exception as e:
            logger.error(f"请求忽略电池优化时出错: {e}")
            
        return False
    
    def start(self, notification_text="正在运行..."):
        """启动前台服务"""
        if self.service_active:
            return
            
        try:
            activity = PythonActivity.mActivity
            if not activity:
                logger.error("无法获取Activity")
                return
                
            # 检查电池优化状态
            self.check_battery_optimization()
            
            # 获取唤醒锁
            self._acquire_wakelock()
            
            # 创建返回应用的Intent
            intent = Intent(activity, PythonActivity)
            flags = 0
            if Build.VERSION.SDK_INT >= 23:  # Build.VERSION_CODES.M
                flags = PendingIntent.FLAG_IMMUTABLE
                
            pending_intent = PendingIntent.getActivity(
                activity,
                0,
                intent,
                flags
            )
            
            # 构建通知
            builder = NotificationCompat.Builder(activity, self.channel_id)
            builder.setContentTitle('摩尔斯电码训练器')
            builder.setContentText(notification_text)
            builder.setSmallIcon(activity.getApplicationInfo().icon)
            builder.setContentIntent(pending_intent)
            builder.setOngoing(True)
            builder.setPriority(NotificationCompat.PRIORITY_LOW)  # 降低优先级减少系统资源使用
            
            # 显示通知
            self.notification = builder.build()
            
            # 在某些设备上，startForeground方法可能不在主Activity上
            service = activity.getApplicationContext()
            service_wrapper = cast('android.app.Service', service)
            
            try:
                service_wrapper.startForeground(self.notification_id, self.notification)
            except:
                # 回退到旧方法
                activity.startForeground(self.notification_id, self.notification)
                
            self.service_active = True
            logger.info("前台服务已启动")
        except Exception as e:
            logger.error(f"启动前台服务时出错: {e}")
    
    def stop(self):
        """停止前台服务"""
        if not self.service_active:
            return
            
        try:
            activity = PythonActivity.mActivity
            if not activity:
                logger.error("无法获取Activity")
                return
                
            # 释放唤醒锁
            self._release_wakelock()
            
            # 停止前台服务
            service = activity.getApplicationContext()
            service_wrapper = cast('android.app.Service', service)
            
            try:
                service_wrapper.stopForeground(True)
            except:
                # 回退到旧方法
                activity.stopForeground(True)
                
            self.service_active = False
            logger.info("前台服务已停止")
        except Exception as e:
            logger.error(f"停止前台服务时出错: {e}")
    
    def update_notification(self, text):
        """更新通知内容"""
        if not self.service_active:
            return
            
        try:
            activity = PythonActivity.mActivity
            if not activity:
                return
                
            # 更新通知文本
            builder = NotificationCompat.Builder(activity, self.channel_id)
            builder.setContentTitle('摩尔斯电码训练器')
            builder.setContentText(text)
            builder.setSmallIcon(activity.getApplicationInfo().icon)
            builder.setOngoing(True)
            
            # 更新通知
            notification_manager = activity.getSystemService(Context.NOTIFICATION_SERVICE)
            notification_manager = cast('android.app.NotificationManager', notification_manager)
            notification_manager.notify(self.notification_id, builder.build())
            logger.info(f"通知已更新: {text}")
        except Exception as e:
            logger.error(f"更新通知时出错: {e}")
    
    def __del__(self):
        """析构函数"""
        self.stop()
        if self.wakelock and self.wakelock.isHeld():
            self._release_wakelock() 