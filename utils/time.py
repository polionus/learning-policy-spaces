import datetime

def get_time_stamp() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")