import os
import sys
import urllib.error
import urllib.request
from multiprocessing.pool import ThreadPool


def fetch_image_url(line):
    """ WARNING => please read the DISCLAIMER before running this script
        url => this is the URL to download
        model_type => this the car model type, a folder will be created for each model type
        image_name => this is the name of the image renamed, this can be used a label

        global variables

        home_dir ==> this is set to be the current dir 
        master_dir ==> for the toyota dataset this will be toyota_master
        threadpool_size ==> this is set to 120 by default, please reduce according you instance thread count 
        image_source_link_jpg => this is csv file with iamge links 

    """
    try:
        split_array = line.strip().split('|')
        url = split_array[0]
        model_type = split_array[1]
        image_name = split_array[2]

        extension = url[-4:]
        asis_image_path = as_is_download_dir + model_type + "/"

        if not os.path.exists(asis_image_path):
            os.makedirs(asis_image_path)

        image_path_with_name = asis_image_path + str(image_name) + extension

        urllib.request.urlopen(url)
        urllib.request.urlretrieve(url, image_path_with_name)

    except urllib.error.HTTPError as e:
        print(str(e))
    except OSError as e:
        print(str(e))
    except Exception as e:
        print(str(e))
        print("Unexpected error:", sys.exc_info()[0])
    raise


# global variables
home_dir = os.getcwd() + '/'
master_dir = 'toyota_master/'
# should be in a data structure
as_is_download_dir = home_dir + master_dir
threadpool_size = 120

# should be in a data structure
image_source_link = 'toyota_links.csv'

# directory creation
# should loop over the data structure with the directories and create them if they don't exists
directory_list = [as_is_download_dir]
for directory in directory_list:
    if not os.path.exists(directory):
        os.makedirs(directory)

tp = ThreadPool(threadpool_size)

try:
    with open(image_source_link) as f:
        headers = next(f)
        url = f.readlines()
        tp.imap_unordered(fetch_image_url, url)
        tp.close()
        tp.join()
except Exception as e:
    print(str(e))
except OSError as e:
    print(str(e))