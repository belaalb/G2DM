
from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-',
                                    dest_path='./alexnet_caffe.pth.tar',
                                    unzip=False,
                                    showsize=True,
                                    overwrite=False)



## Download vlcs 
from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='13qTR2jJqCQgfdbI3E_a6dLU7rEWpF37T',
                                    dest_path='./data/vlcs/vlcs.tar.gz',
                                    unzip=False,
                                    showsize=True,
                                    overwrite=False)
## unzip

import tarfile

fname= './data/vlcs/vlcs.tar.gz'
tar = tarfile.open(fname, "r:gz")
tar.extractall()
tar.close()

## Move to ./data/vlcs/prepared_data

import shutil, glob

for filePath in glob.glob('./VLCS' + '/*'):
    shutil.move(filePath, './data/vlcs/prepared_data/')
