import fcntl
import pandas as pd
import threading

# read file from csv by pandas
id = threading.current_thread().getName()


data = pd.read_csv('./result.csv')
#
print(data)