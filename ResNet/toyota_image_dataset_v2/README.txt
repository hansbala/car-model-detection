#################################
Please read the Disclaimer first
#################################
This dataset is prepared by Occulta Insights. 
https://occultainsights.io

Images are downloaded following the below steps. Pre-downloaded images are available in the toyota_cars folder.

This package contains a python script => download_images.py

This package contains a file with image links => toyota_validated_links.csv

Using the python script you will have to download the images yourself. This might take a few hours depending your internet speed and thread count. 

you can change the following global variables to suit your needs 

global variables
home_dir ==> this is set to be the current dir 
master_dir ==> for the toyota dataset this will be toyota_master
threadpool_size ==> this is set to 120 by default, please increae/reduce according you instance thread count e.g 12 to 120 
image_source_link_jpg => this is csv file with image links 

To run as below you will need python3
>python3 download_images.py


for any queries please contact 
support@occultainsights.io





