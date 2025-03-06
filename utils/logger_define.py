import logging
import os
from datetime import datetime


class CustomLogger:
    def __init__(self, model_name, logger_dir):
        self.logger_dir = logger_dir
        self.model_name = model_name
        self.log_file = self._generate_log_filename()
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.INFO)

        # 创建文件处理器，将日志写入文件
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)

        # 创建控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # 将格式化器添加到处理器
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def _generate_log_filename(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(self.logger_dir, f'train_{self.model_name}_{timestamp}.log')
        return log_filename

    def get_logger(self):
        return self.logger
