from __future__ import print_function

from enum import Enum
from argparse import ArgumentParser
import time
import os, shutil
from glob import glob

import tarfile
import urllib.request

from dask import delayed
from dask import compute
import pandas as pd



#Namespace containing  dafault data
class DafaultData:
    source_url = "https://storage.googleapis.com/dask-tutorial-data/nycflights.tar.gz"
    data_dir = 'data'
    zip_name = 'nycflights.tar.gz'
    flight_dir = 'nycflights'
    json_dir = 'flightjson'

    round_tol = 3

#Enum containing the texts of output
class Log(Enum):
    MAINSTART = 'Setting up data directory\n' \
                '-------------------------'
    DOWNLOAD = "- Downloading NYC Flights dataset... "
    EXTRACT = "- Extracting flight data... "
    CREAT_JSON = "- Creating json data... "
    WORK_ON = 'Working on ' + os.path.curdir
    WORK_END = 'Ended working on ' + os.path.curdir
    FINISH_DOWNLOAD = "** Finished! **"
    DONE = 'done!'
    DONE_TIME = 'done! Time in second ='
    MAINEND = 'Finished!'

#Loading flight function
def flights(update_flag, json_flag):

    #taking necessary parameters from DafaultData
    source_url = DafaultData.source_url
    data_dir = DafaultData.data_dir
    zip_name = DafaultData.zip_name
    flight_dir_name = DafaultData.flight_dir
    json_dir_name = DafaultData.json_dir

    round_tol = DafaultData.round_tol

    #main part
    flights_raw = os.path.join(data_dir, zip_name)
    flight_dir = os.path.join(data_dir, flight_dir_name)
    json_dir = os.path.join(data_dir, json_dir_name)

    if update_flag == True and os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    if not os.path.exists(flights_raw):
        print(Log.DOWNLOAD.value, end='', flush=True)
        tic = time.time()
        urllib.request.urlretrieve(source_url, flights_raw)
        print(Log.DONE_TIME.value, round(time.time() - tic,round_tol), flush=True)

    if not os.path.exists(flight_dir):
        tic = time.time()
        print(Log.EXTRACT.value, end='', flush=True)
        tar_path = os.path.join(data_dir, zip_name)
        with tarfile.open(tar_path, mode='r:gz') as flights:
            flights.extractall(os.path.join(data_dir, ''))
            print(Log.DONE_TIME.value, round(time.time() - tic,round_tol), flush=True)


    if json_flag == True:
        if not os.path.exists(json_dir):
            tic = time.time()
            print(Log.CREAT_JSON.value, flush=True)
            os.mkdir(json_dir)

            # Parallel performing with Dask
            l = []
            for path in glob(os.path.join(flight_dir, '*.csv')):
                l.append(delayed(convert_to_json)(path, json_dir))
            compute(l)

            print(Log.CREAT_JSON.value, Log.DONE_TIME.value,round(time.time() - tic,round_tol), flush=True)
    print(Log.FINISH_DOWNLOAD.value)


#Seperated function for parallel performing
def convert_to_json(path, json_dir):
    print(os.path.join(Log.WORK_ON.value, path))

    prefix = os.path.splitext(os.path.basename(path))[0]
    df = pd.read_csv(path)
    df.to_json(os.path.join(json_dir, prefix + '.json'),
               orient='records', lines=True)

    print(os.path.join(Log.WORK_END.value, path))



def get_args():
    parser = ArgumentParser(description='Load flights data')

    parser.add_argument('-j', '--json',
                        action='store_true',
                        help='Duplicates the uploaded data in json format')
    parser.add_argument('-u', '--update',
                        action= 'store_true',
                        help='Update all data')

    args = parser.parse_args()
    return args

def random_array():
    pass

def weather():
    pass


def main(args):
    print(Log.MAINSTART.value)

    flights(args.update, args.json)
    # random_array()
    # weather()
    print(Log.MAINEND.value)


if __name__ == '__main__':
    args = get_args()
    main(args)