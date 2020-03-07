
from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-',
                                    dest_path='./alexnet_caffe.pth.tar',
                                    unzip=False,
                                    showsize=True,
                                    overwrite=False)
                                    