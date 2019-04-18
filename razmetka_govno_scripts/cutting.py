# ffmpeg -i video.webm -ss 00:00:07.000 -vframes 1 thumb.jpg
import subprocess
def get_image(video_name, timestamp, filename):
    subprocess.check_output(['ffmpeg','-i',video_name,
                                   "-ss",timestamp,"-vframes",
                                   str(1),filename])
video_name = "videos/"
video_name+= "ufc234_gastelum_bisping_1080p_nosound.mp4"
timestamp = "00:05:41.000"
filename = "test.png"
# For testing
#get_image(video_name, timestamp, filename)
