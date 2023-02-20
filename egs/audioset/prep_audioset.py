import numpy as np
import json
import os


BASE_DIR = "./data/datafiles"
WAV_DIR = os.path.join(BASE_DIR, "wav_files")


def youtube_video_to_wav_file(video_id, start_time, end_time, wav_files_output_path):
    duration = float(end_time) - float(start_time)
    
    os.system(
        f"ffmpeg $(./yt-dlp/yt-dlp -g 'https://www.youtube.com/watch?v={video_id}' | sed \"s/.*/-ss {start_time} -i &/\") -t {duration} -c copy {wav_files_output_path}/{video_id}.mkv && ffmpeg -i {wav_files_output_path}/{video_id}.mkv -acodec pcm_s16le -ac 2 {wav_files_output_path}/{video_id}.wav && rm {wav_files_output_path}/{video_id}.mkv")
    # for mac
    """os.system(
        f"ffmpeg $(./yt-dlp/yt-dlp_macos -g 'https://www.youtube.com/watch?v={video_id}' | sed \"s/.*/-ss {start_time} -i &/\") -t {duration} -c copy {wav_files_output_path}/{video_id}.mkv && ffmpeg -i {wav_files_output_path}/{video_id}.mkv -acodec pcm_s16le -ac 2 {wav_files_output_path}/{video_id}.wav && rm {wav_files_output_path}/{video_id}.mkv")
    """
    
    # resample 16k
    # os.system(f"sox {wav_files_output_path}/{video_id}.wav -r 16000 {wav_files_output_path}/{video_id}_16k.wav && rm {wav_files_output_path}/{video_id}.wav")

    return f"{wav_files_output_path}/{video_id}.wav"


def prep_data(csv_file_path, wav_files_output_path, sample_data_json_file_path):
    os.makedirs(wav_files_output_path, exist_ok=True)

    meta = np.loadtxt(csv_file_path, delimiter=',',
                      dtype='str', skiprows=1, quotechar='"')
    wav_list = []

    for i in range(len(meta)):
        video_id = meta[i][0]
        start_time = meta[i][1]
        end_time = meta[i][2]
        labels = meta[i][3]
        wav_file_path = youtube_video_to_wav_file(
            video_id, start_time, end_time, wav_files_output_path)

        if os.path.exists(wav_file_path):
            cur_dict = {"wav": wav_file_path, "labels": labels}
            wav_list.append(cur_dict)

    with open(sample_data_json_file_path, 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)


print("Preparing training data")
prep_data("./data/balanced_train_segments.csv",
          os.path.join(WAV_DIR, "train"), os.path.join(BASE_DIR, "train_data.json"))

print("Preparing eval data")
prep_data("./data/eval_segments.csv", os.path.join(WAV_DIR, "eval"),
          os.path.join(BASE_DIR, "eval_data.json"))

print('Finished AudioSet Preparation')
