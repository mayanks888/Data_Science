import datetime
import time
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
st = datetime.datetime.fromtimestamp(ts).strftime('%d_%m_%Y_%H_%M_%S')
print (st)