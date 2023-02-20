import numpy as np
import json
import pandas as pd
import os
import multiprocessing as mp


BASE_DIR = "./data/datafiles"
WAV_DIR = os.path.join(BASE_DIR, "wav_files")


def youtube_video_to_wav_file(video_id, start_time, end_time, wav_files_output_path):
    duration = float(end_time) - float(start_time)
    os.system(
        f"ffmpeg $(./yt-dlp/yt-dlp -g 'https://www.youtube.com/watch?v={video_id}' | sed \"s/.*/-ss {start_time} -i &/\") -t {duration} -c copy {wav_files_output_path}/{video_id}.mkv && ffmpeg -i {wav_files_output_path}/{video_id}.mkv -acodec pcm_s16le -ac 2 {wav_files_output_path}/{video_id}.wav && rm {wav_files_output_path}/{video_id}.mkv")
    
    """
    # for mac
    os.system(
        f"ffmpeg $(./yt-dlp/yt-dlp_macos -g 'https://www.youtube.com/watch?v={video_id}' | sed \"s/.*/-ss {start_time} -i &/\") -t {duration} -c copy {wav_files_output_path}/{video_id}.mkv && ffmpeg -i {wav_files_output_path}/{video_id}.mkv -acodec pcm_s16le -ac 2 {wav_files_output_path}/{video_id}.wav && rm {wav_files_output_path}/{video_id}.mkv")
    """

    # resample 16k
    os.system(f"sox {wav_files_output_path}/{video_id}.wav -r 16000 {wav_files_output_path}/{video_id}_16k.wav && rm {wav_files_output_path}/{video_id}.wav")

    return f"{wav_files_output_path}/{video_id}_16k.wav"


def save_chunk_to_file(df_chunk, chunk_num, file_name):
    sub_file_name = file_name.replace('.csv', f'_{chunk_num}.csv')
    df_chunk.to_csv(sub_file_name, index=False)


def divide_file_to_sub_files(csv_file, num_cpus):
    df = pd.read_csv(csv_file, header=0)
    df_len = len(df)
    
    df_chunks = []
    chunks_indices = np.array_split(np.arange(df_len), num_cpus)
    for chunk_indices in chunks_indices:
        chunk = df.iloc[chunk_indices]
        df_chunks.append(chunk)
    
    proccesses = []
    for chunk_num, df_chunk in enumerate(df_chunks):
        p = mp.Process(target=save_chunk_to_file,
                       args=(df_chunk, chunk_num, csv_file,))
        proccesses.append(p)
        p.start()

    for p in proccesses:
        p.join()


def create_wav_files_from_csv(csv_file_path, wav_files_output_path, sample_data_json_file_path):
    meta = np.loadtxt(csv_file_path, delimiter=',',
                      dtype='str', skiprows=1, quotechar='"')

    os.remove(csv_file_path)

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


def merge_json_files(files_path, sample_data_json_file_path):
    wav_list = []

    # concat all files
    for file_path in files_path:
        with open(file_path) as f:
            # get data list
            file_data = json.load(f)['data']

        os.remove(file_path)

        # extend list
        wav_list.extend(file_data)

    with open(sample_data_json_file_path, 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)


def prep_data(csv_file_path, wav_files_output_path, sample_data_json_file_path):
    if os.path.exists(wav_files_output_path):
        print(f"{wav_files_output_path} is already exist. Use existing files.")

    os.makedirs(wav_files_output_path)

    num_cpus = mp.cpu_count()
    divide_file_to_sub_files(csv_file_path, num_cpus)
    proccesses = []
    files_path = []
    for i in range(num_cpus):
        csv_file = csv_file_path.replace('.csv', f'_{i}.csv')
        json_file = sample_data_json_file_path.replace('.json', f'_{i}.json')
        files_path.append(json_file)

        # start process
        p = mp.Process(target=create_wav_files_from_csv, args=(
            csv_file, wav_files_output_path, json_file,))
        proccesses.append(p)
        p.start()

    for p in proccesses:
        p.join()

    merge_json_files(files_path, sample_data_json_file_path)


def main():
    print("Preparing training data")
    prep_data("./data/balanced_train_segments.csv",
              os.path.join(WAV_DIR, "train"), os.path.join(BASE_DIR, "train_data.json"))

    print("Preparing eval data")
    prep_data("./data/eval_segments.csv", os.path.join(WAV_DIR, "eval"),
              os.path.join(BASE_DIR, "eval_data.json"))

    print('Finished AudioSet Preparation')


if __name__ == '__main__':
    main()
