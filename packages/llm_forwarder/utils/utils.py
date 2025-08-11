import os

from loguru import logger
from pydub import AudioSegment

from .config import Config

config_data = Config().get_config()

target_sample_rate = 16000
target_num_channels = 1


def convert_mp3_to_wav(mp3_path, wav_path, target_sample_rate=16000, target_num_channels=1):
    """
    将 MP3 音频文件转换为 WAV 格式。
    Args:
        mp3_path (str): 输入 MP3 文件的路径。
        wav_path (str): 输出 WAV 文件的路径。
    """
    try:
        # 1. 加载 MP3 文件
        audio = AudioSegment.from_mp3(mp3_path)
        print(f"原始音频信息: 采样率={audio.frame_rate} Hz, 声道数={audio.channels}")
        # 2. 强制设置采样率
        if audio.frame_rate != target_sample_rate:
            print(f"正在将采样率从 {audio.frame_rate} Hz 转换为 {target_sample_rate} Hz...")
            audio = audio.set_frame_rate(target_sample_rate)
        # 3. 强制设置声道数
        if audio.channels != target_num_channels:
            print(f"正在将声道数从 {audio.channels} 转换为 {target_num_channels}...")
            audio = audio.set_channels(target_num_channels)
        # 4. 导出为 WAV 文件
        audio.export(wav_path, format="wav")
        logger.info(f"成功将 '{mp3_path}' 转换为 '{wav_path}'")
        logger.info(f"输出 WAV 文件信息: 采样率={target_sample_rate} Hz, 声道数={target_num_channels}")
    except Exception as e:
        logger.error(f"转换过程中发生错误: {e}")
        logger.error("请确保已安装 ffmpeg 并将其添加到系统 PATH 环境变量中。")
        logger.error("如果问题依然存在，请检查 MP3 文件是否损坏。")


def convert_all():
    input_mp3_file_dir = r"D:\sourcecode\open-llm-vtuber-zhiwen\cache"
    for file_name in os.listdir(input_mp3_file_dir):
        if file_name.endswith(".mp3"):
            input_mp3_file = os.path.join(input_mp3_file_dir, file_name)
            output_wav_file = os.path.join(input_mp3_file_dir, file_name.replace(".mp3", ".wav"))
            convert_mp3_to_wav(input_mp3_file, output_wav_file)

    # test_mp3_path = os.path.join(input_mp3_file_dir, "我赢啦.mp3")
    # convert_mp3_to_wav(test_mp3_path, test_mp3_path.replace(".mp3", ".wav"))
