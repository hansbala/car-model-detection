from glob import glob
import os
import random

# separate into train and test dataset
DIR = './toyota_image_dataset_v2/toyota_cars/'
TEST_RATIO = 0.13

def split_dataset():
    test_files = []
    train_files = []
    # iterate over all folders
    car_directories = glob(DIR + "*/")
    for directory in car_directories:
        files = glob(directory + '*.jpg')
        # pick like 70 images from each folder for test, rest train
        curr_test_set = random.sample(files, int(TEST_RATIO * len(files)))
        curr_train_set = [x for x in files if x not in curr_test_set]
        # add it to the master lists
        test_files.extend(curr_test_set)
        train_files.extend(curr_train_set)
    # now that we have the split files, return as tuple
    return (train_files, test_files)


def get_model_name(model_file_path):
    slash_pos = model_file_path.rfind('/')
    dash_pos = model_file_path.rfind('-')
    return model_file_path[slash_pos + 1 : dash_pos]

# TODO: Open and store the image inside each tuple
def assign_labels(dataset):
    labels = []
    for file_path in dataset:
        # get the label from the image
        car_name = get_model_name(file_path)
        labels.append(car_name)
    return (dataset, labels)


#######################
##  ONLY FOR TESTING ##
#######################
#######################
def main():
    train_set, test_set = split_dataset()
    assign_labels(train_set)

if __name__ == '__main__':
    main()
#######################
#######################
#######################