import datetime

class Logger:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def write_log(self, log_content, *args):
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime('%Y-%m-%d %H:%M:%S')
        if args:
            log_content = log_content.format(*args)
        with open(self.log_file_path, 'a') as file:
            file.write(f'{formatted_time} {log_content}\n')