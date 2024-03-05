import subprocess
import re
import logging
import inspect


class GPUConsoleLogger:
    def __init__(self):
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger("GPU_Logger")
        logger.setLevel(logging.INFO)

        # 创建一个文件处理器，将日志写入文件中，并设置编码为UTF-8
        file_handler = logging.FileHandler("gpu_usage.log", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 创建一个控制台处理器，将日志输出到控制台
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 创建一个格式化器，定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 将处理器添加到日志记录器中
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def get_total_memory(self):
        # 获取显卡总显存
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.total --format=csv,noheader", shell=True)
        total_memory = int(output.decode().strip().split()[0])
        return total_memory

    def get_current_memory_usage(self):
        # 获取当前显卡占用显存
        output = subprocess.check_output("nvidia-smi --query-gpu=memory.used --format=csv,noheader", shell=True)
        current_memory = int(output.decode().strip().split()[0])
        return current_memory

    def use(self, *args):
        description = args[0] if args else ""
        total_memory = self.get_total_memory()
        current_memory = self.get_current_memory_usage()
        percentage = (current_memory / total_memory) * 100

        # 获取当前文件的绝对路径和行号
        frame = inspect.currentframe().f_back
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        output = f"{description} 显存占用百分比：{percentage:.2f}%，当前占用/总显存：{current_memory}/{total_memory}MB。 File \"{filename}\", line {lineno}"
        self.logger.info(output)




# 示例用法
if __name__ == "__main__":
    logger = GPUConsoleLogger()
    logger.use("自定义描述文本")
