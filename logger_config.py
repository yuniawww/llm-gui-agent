# logger_config.py
import logging
import os
from datetime import datetime

def setup_logger(name="agent", log_level=logging.INFO, log_to_file=True):
    """设置日志记录器
    
    Args:
        name: 日志器名称
        log_level: 日志级别
        log_to_file: 是否写入文件
    
    Returns:
        配置好的logger对象
    """
    
    # 创建logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除已有的处理器
    logger.handlers.clear()
    
    # 创建格式化器
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_to_file:
        # 创建logs目录
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # 生成日志文件名
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = os.path.join(log_dir, f"agent_{timestamp}.log")
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_api_call(logger, api_name, success, response_time=None, error=None, details=None):
    """记录API调用日志
    
    Args:
        logger: 日志器对象
        api_name: API名称
        success: 是否成功
        response_time: 响应时间(秒)
        error: 错误信息
        details: 额外详情
    """
    
    status = "SUCCESS" if success else "FAILED"
    time_str = f" ({response_time:.2f}s)" if response_time else ""
    
    message = f"API Call: {api_name} - {status}{time_str}"
    
    if details:
        message += f" | Details: {details}"
    
    if success:
        logger.info(message)
    else:
        error_msg = f"{message} | Error: {error}" if error else message
        logger.error(error_msg)

def log_state_change(logger, from_state, to_state, reason=None):
    """记录状态变化日志"""
    message = f"State Change: {from_state} -> {to_state}"
    if reason:
        message += f" | Reason: {reason}"
    logger.info(message)

def log_task_progress(logger, task_type, task_name, status, details=None):
    """记录任务进度日志"""
    message = f"Task {status}: [{task_type}] {task_name}"
    if details:
        message += f" | {details}"
    
    if status in ["COMPLETED", "SUCCESS"]:
        logger.info(message)
    elif status in ["FAILED", "ERROR"]:
        logger.error(message)
    else:
        logger.info(message)
