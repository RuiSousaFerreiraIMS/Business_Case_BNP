# src/code/data_functions.py
from src.code.io_utils import load, data_path

def load_bdoss(parse_dates=None):
    return load(data_path("bdoss.csv"), parse_dates=parse_dates)

def load_crc(parse_dates=None):
    return load(data_path("crc.csv"), parse_dates=parse_dates)

def load_credscore(parse_dates=None):
    return load(data_path("credscore.csv"), parse_dates=parse_dates)

def load_fama(parse_dates=None):
    return load(data_path("fama.csv"), parse_dates=parse_dates)