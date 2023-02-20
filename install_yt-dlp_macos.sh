#!/bin/bash
mkdir yt-dlp
cd yt-dlp
wget https://github.com/ytdl-patched/yt-dlp/releases/download/2023.02.17.334/yt-dlp_macos.zip
unzip -o yt-dlp_macos.zip
rm -f yt-dlp_macos.zip
cd ..
cp -R yt-dlp egs/audioset/
