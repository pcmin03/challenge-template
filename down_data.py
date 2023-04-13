import pandas as pd
pd.options.mode.chained_assignment = None
import urllib.request
import tarfile
import os

dataset_download_path="./asl_dataset"
## Replace the download_url variable with the url you get from MSR site
download_url="https://msropendataset01.blob.core.windows.net/msaslamericansignlanguagedataset-x/msaslamericansignlanguagedataset.tar.gz?sv=2019-02-02&sr=b&sig=Tf%2BAh2R7grSsBDdicopSlNSfrP8DoqBvC8pO3El9PxQ%3D&st=2022-05-12T03%3A22%3A15Z&se=2022-06-11T03%3A27%3A15Z&sp=r"
urllib.request.urlretrieve(download_url,"msaslamericansignlanguagedataset.tar.gz" )
file = tarfile.open("msaslamericansignlanguagedataset.tar.gz", mode="r|gz")
file.extractall(path=dataset_download_path)
