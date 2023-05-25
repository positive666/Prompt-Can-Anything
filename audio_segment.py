

import os
import gradio as gr
from pydub import AudioSegment

# function to crop audio according to the given start and end time
def crop_audio(file_path, start_time, end_time):
    audio = AudioSegment.from_file(file_path)
    cropped_audio = audio[start_time:end_time]
    filename = os.path.splitext(os.path.basename(file_path))[0]
    if not os.path.exists("cropped_audio"):
        os.makedirs("cropped_audio")
    cropped_file_path = os.path.join("cropped_audio",f"{filename}_{start_time//1000}_{end_time//1000}.wav")
    cropped_audio.export(cropped_file_path, format="wav")
    return cropped_file_path

# function to split audio file into segments
def split_audio_file(file_path, output_path, segment_time=3000):
    audio = AudioSegment.from_file(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Calculating total segments that will be created.
    total_segments = int(audio.duration_seconds // (segment_time/1000)) + 1

    # Creating each segment and saving to the output folder
    for segment_number in range(total_segments):
        start_time = segment_number * segment_time
        end_time = start_time + segment_time
        segment_file_path = os.path.join(output_path, f"{file_name}_{start_time//1000}_{end_time//1000}.wav")
        segment = audio[start_time:end_time]
        segment.export(segment_file_path, format="wav")

    return output_path

# main function
def audio_processing(file_path, output_path, label):
    # 分割音频文件
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    split_audio_file(file_path, output_path)

    # 获取手动选择的音频段并裁剪
    cropped_files_paths = []
    for root, dirs, files in os.walk(output_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.abspath(os.path.join(root, file))
                cropped_file_path = crop_audio(file_path, 0, 1000) # 注意此处仅提供示例裁剪了1s的音频 
                cropped_files_paths.append(cropped_file_path)

    # 生成txt文件
    txt_file = open('file_labels.txt', 'a')
    for index, cropped_file_path in enumerate(cropped_files_paths):
        segment_label = label + '_' + str(index)
        # 将文件路径和标签写入txt文件
        txt_file.write(f"{cropped_file_path}\t{segment_label}\n")
    txt_file.close()

    print("处理完成！")
# 定义输入界面, 接收音频文件、输出文件夹和标签
iface = gr.Interface(
            fn=audio_processing,
            inputs=[gr.inputs.File(label="上传音频文件"),
                    gr.inputs.Textbox(label="输出文件夹路径"),
                    gr.inputs.Textbox(label="标签")],
            outputs="text",
            title="音频处理工具",
            description="通过鼠标点击音频的任意区间保存片段")

iface.launch()    