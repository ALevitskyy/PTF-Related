# youtube-dl -f 22 'http://www.youtube.com/watch?v=P9pzm5b6FFY'
from datetime import datetime, timedelta
from random import randrange
import numpy as np
# timestamp_list = [["00:03:00","00:03:35"],
#              ["00:05:00","00:06:35"]]
def random_date(start, end):
    """
        This function will return a random datetime between two datetime
        objects.
        """
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.uniform(0,int_delta)
    return str(datetime.time(start + timedelta(seconds=random_second)))

def random_from_one_duplet(duplet):
    result = []
    date1 = duplet[0]
    date2 = duplet[1]
    date1 = datetime.strptime(date1, "%H:%M:%S")
    date2 = datetime.strptime(date2, "%H:%M:%S")
    diff = date2 - date1
    seconds = diff.seconds
    number_of_timestamps = int(seconds/3) + 1
    for i in range(number_of_timestamps):
        result.append(random_date(date1,date2))
    return result

def random_time_stamps(timestamp_list):
    result = []
    for duplet in timestamp_list:
        result.append(random_from_one_duplet(duplet))
    result = [z for i in result for z in i]
    return result
#print(random_time_stamps(timestamp_list))

