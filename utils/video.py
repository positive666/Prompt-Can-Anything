import asyncio
import os
import cv2
import time
import sys
import subprocess
from multiprocessing import Pool, current_process
from threading import Thread

dataset_path = sys.argv[1]
output_folder = sys.argv[2]

def process_video(video_path):
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    # Create output folder
    output_path = os.path.join(output_folder, video_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # Decode video
    ffmpeg_command = ['ffmpeg', '-i', video_path, os.path.join(output_path, video_name + '_%04d.png')]
    process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print('Error processing video {}:\n{}'.format(video_path, stderr))

if __name__ == '__main__':
    # Get video list
    video_list = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                video_list.append(os.path.join(root, file))
    # Process videos using pool of workers
    pool_size = 4 # Change this to number of CPU cores you want to use
    pool = Pool(processes=pool_size)
    for video_path in video_list:
        pool.apply_async(process_video, args=(video_path,))
    pool.close()
    pool.join()


    print('All videos processed!')
async def process_video_chunk(chunk_index, video_chunk, output_folder):
    # 这里输入您的图像处理操作，例如人脸检测等等。
    print(f"Processing chunk index {chunk_index}...")

    # 该函数可以根据您的需求来设置输出文件名，这里我们按照chunk_index进行计数:
    output_file = os.path.join(output_folder, f"{chunk_index}.mp4")
    # 实际上，通过更改文件扩展名，您可以保存为不同的文件格式，例如.avi, .mkv等

    # 再次调用VideoWrite对象以创建一个新的输出文件:
    output_writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), 25, (video_chunk.shape[1], video_chunk.shape[0]), True)

    # 写入切片中的帧:
    for frame in video_chunk:
        output_writer.write(frame)

    output_writer.release()

async def process_video(input_file, chunk_size_secs=10, output_folder="output"):
    # 创建输出目录:
    os.makedirs(output_folder, exist_ok=True)

    # 打开输入视频:
    input_video = cv2.VideoCapture(input_file)

    # 获取视频帧数和每一帧的间隔时间:
    frame_count = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    frame_duration_ms = int(1 / fps * 1000)

    #创建任务列表
    tasks = []

    # 分割视频:
    chunk_idx = 0
    while True:
        # 读取chunk_size_secs秒钟的帧:
        frames_chunk = []
        chunk_start_frame_index = chunk_idx * chunk_size_secs * fps
        chunk_end_frame_index = (chunk_idx + 1) * chunk_size_secs * fps
        end_of_video_reached = False

        # 获取每一帧并将其添加到切片中
        for i in range(chunk_start_frame_index, chunk_end_frame_index):
            # 寻找下一帧:
            input_video.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = input_video.read()

            # 如果到达视频末尾，停止读取:
            if not success:
                end_of_video_reached = True
                break

            frames_chunk.append(frame)

        # 发现最后一个文件不足，跳出循环
        if len(frames_chunk) < 1:
            break

        # 创建任务并发处理该切片:
        tasks.append(asyncio.create_task(process_video_chunk(chunk_idx, frames_chunk, output_folder)))

        # 如果到达视频末尾，停止分割:
        if end_of_video_reached:
            break

        # 增加计数器以准备下一个切片:
        chunk_idx += 1

    # 等待所有任务完成
    await asyncio.gather(*tasks)

    input_video.release()

if __name__ == "__main__":
    start_time = time.monotonic()

    asyncio.run(process_video("input_video.mp4"))

    end_time = time.monotonic()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds.")