import numpy as np
import json
import os

def youtube_video_to_wav_file(video_id, start_time, end_time, wav_files_output_path):
    duration = end_time - start_time
    os.system(f"cd {wav_files_output_path} && ffmpeg $(./yt-dlp/yt-dlp -g 'https://www.youtube.com/watch?v={video_id}' | sed \"s/.*/-ss {start_time} -i &/\") -t {duration} -c copy {video_id}.mkv && ffmpeg -i {video_id}.mkv -acodec pcm_s16le -ac 2 {video_id}.wav && rm {video_id}.mkv")
    os.system(f"cd {wav_files_output_path} && sox {video_id}.wav -r 16000 {video_id}_16k.wav && rm {video_id}.wav")

def prep_data(csv_file_path, wav_files_output_path, sample_data_json_file_path):
    os.mkdir(wav_files_output_path)
    meta = np.loadtxt(csv_file_path, delimiter=',', dtype='str', skiprows=1, quotechar='"')
    wav_list = []
    for i in range(0, len(meta)):
        video_id = meta[i][0]
        start_time = meta[i][1]
        end_time = meta[i][2]
        lables = meta[i][3]
        youtube_video_to_wav_file(video_id, start_time, end_time, wav_files_output_path)
        
        wav_file_path = wav_files_output_path + video_id + "_16k.wav"
        cur_dict = {"wav": wav_file_path, "labels": lables}
        wav_list.append(cur_dict)
    with open(sample_data_json_file_path, 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)

os.mkdir("./data/datafiles/audio_16k")

print("Preparing training data")
prep_data("./data/balanced_train_segments.csv", "./data/datafiles/audio_16k/train", "./data/datafiles/train_data.json")

print("Preparing eval data")
prep_data("./data/eval_segments.csv", "./data/datafiles/audio_16k/eval", "./data/datafiles/eval_data.json")

print('Finished AudioSet Preparation')
