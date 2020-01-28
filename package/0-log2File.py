import logging
import sys
import warnings


class LogWriter(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        '''l = self.log_level
        if self.log_level == logging.WARNING:
            if 'Traceback' in buf:
                for line in buf.rstrip().splitlines():
                    self.logger.log(logging.ERROR, line.rstrip())
        else:'''
        for line in buf.rstrip().splitlines():
            if self.log_level == logging.WARNING and 'Traceback' in line.rstrip():
                self.log_level = logging.ERROR
                break
            else:
                break
        for line in buf.rstrip().splitlines():    
            self.logger.log(self.log_level, line.rstrip())
        
                
    def flush(self):
        pass


logging.basicConfig(
       level=logging.DEBUG,
       format='[%(asctime)s | %(name)-12s ] |%(levelname)-8s| %(message)s',
       filename="log_file.log",
       filemode='a')


stderr_logger = logging.getLogger('STDERR')
sys.stderr = LogWriter(stderr_logger, logging.WARNING)

def log2File(proc_name):
    
    stdout_logger = logging.getLogger(proc_name)
    sys.stdout = LogWriter(stdout_logger, logging.INFO)



if __name__ == "__main__":
    
    log2File("log2File")
    
    print('I\'m fine')
    warnings.warn(r"I'm Warning!")
    1/0