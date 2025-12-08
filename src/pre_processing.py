from pytube import YouTube

url = "https://www.youtube.com/watch?v=GgXAs7UVpig"

yt = YouTube(url)
stream = yt.streams.filter(file_extension="mp4").get_highest_resolution()
stream.download("data/raw", filename="full_video.mp4")

print("Downloaded!")