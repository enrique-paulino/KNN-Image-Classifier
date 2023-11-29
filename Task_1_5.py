import argparse
import csv
import os

import cv2 # Imported for reading and transforming images
from sewar import * # Imported for conducting image quality assessments
import numpy as np

student_id = '2033700'

# Provided Classification System
classification_scheme = ['Female','Male','Primate','Rodent','Food']

# INPUT: training_data      : a list of lists that was read from the training data csv (see parse_arguments function)
#        k                  : the value of k neighbours
#        sim_id             : value from 1 to 5 which says what similarity should be used;
#                             values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                             values from 4 and 5 denote similarities from Task 5
#        data_to_classify   : a list of lists that was read from the data to classify csv;
#                             this data is NOT used for training the classifier, but for running and testing it
#                             (see parse_arguments function)
# OUTPUT: processed         : a list of lists which expands the data_to_classify with the results on how the
#                             classifier has classified a given image
def kNN(training_data, k, sim_id, data_to_classify):
    processed = [data_to_classify[0] + [student_id]]

    if training_data[0][0] == 'Path':
        training_data.pop(0)
        data_to_classify.pop(0)
        
    for i in range(len(data_to_classify)):
        data_classes = [None] * len(training_data)
        for j in range(len(training_data)):

            # Below are the two images currently being compared scaled to 64x64
            img1 = cv2.resize(cv2.imread(data_to_classify[i][0]), (64,64))
            img2 = cv2.resize(cv2.imread(training_data[j][0]), (64,64))
            

            # Check which simulation the user has selected
            
            if sim_id == 1: # MSE -> Mean Squared Error
                data_classes[j] = [mse(img1,img2), training_data[j][1]]

            elif sim_id == 2: # RMSE -> Root Mean Squared Error
                data_classes[j] = [rmse(img1,img2), training_data[j][1]]

            elif sim_id == 3: # SAM -> Spectral Angle Mapper
                data_classes[j] = [sam(img1,img2), training_data[j][1]]

            elif sim_id == 4: # Question 5 (Part 1): Remaking MSE -> Mean Squared Error
                data_classes[j] = [q5_similarity_1(img1,img2), training_data[j][1]]

            elif sim_id == 5:# Question 5 (Part 2): Remaking RMSE -> Root Mean Squared Error
                data_classes[j] = [q5_similarity_2(img1,img2), training_data[j][1]]

        # Sort by the number (ascending) -> sorting would not work for other measures as a larger number could mean more similar
        data_classes = sorted(data_classes, key=lambda x: x[0])
        data_classes = data_classes[:k]
        
        # Creates a new list with only the classes and then finds the most common one
        estimated_class = max(set([x[1] for x in data_classes]), key=[x[1] for x in data_classes].count)
            
        processed.append([data_to_classify[i][0], data_to_classify[i][1], estimated_class])
    return processed

# Question 5: MSE -> Mean Squared Error.
def q5_similarity_1(img1, img2): 

    # Recasting the images as a float rather than an int
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Gets the mean of the difference squared
    return np.mean((img1-img2) ** 2) 

# Question 5: RMSE -> Root Mean Squared Error
def q5_similarity_2(img1, img2): 

    # Recasting the images as a float rather than an int
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    # Gets the square root of the mean of the difference squared
    return np.sqrt(np.mean((img1-img2) ** 2)) 

# This function reads the necessary arguments (see parse_arguments function), and based on them executes
# the kNN classifier. If the "unseen" mode is on, the results are written to a file.
def main():
    opts = parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = read_csv_file(opts['training_data'])
    data_to_classify = read_csv_file(opts['data_to_classify'])
    unseen = opts['mode']
    print('Running kNN')
    result = kNN(training_data, opts['k'], opts['sim_id'], data_to_classify)
    if unseen:
        path = os.path.dirname(os.path.realpath(opts['data_to_classify']))
        out = f'{path}/{student_id}_classified_data.csv'
        print(f'Writing data to {out}')
        write_csv_file(out, result)


# Straightforward function to read the data contained in the file "filename"
def read_csv_file(filename):
    lines = []
    with open(filename, newline='') as infile:
        reader = csv.reader(infile)
        for line in reader:
            lines.append(line)
    return lines


# Straightforward function to write the data contained in "lines" to a file "filename"
def write_csv_file(filename, lines):
    with open(filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)


# This function simply parses the arguments passed to main. It looks for the following:
#       -k              : the value of k neighbours
#       -f              : the number of folds to be used for cross-validation
#                         (needed in Task 3)
#       -sim_id         : value from 1 to 5 which says what similarity should be used;
#                         values from 1, 2 and 3 denote similarities from Task 1 that can be called from libraries
#                         values from 4 and 5 denote similarities from Task 5
#       -u              : flag for how to understand the data. If -u is used, it means data is "unseen" and
#                         the classification will be written to the file. If -u is not used, it means the data is
#                         for training purposes and no writing to files will happen.
#       training_data   : csv file to be used for training the classifier, contains two columns: "Path" that denotes
#                         the path to a given image file, and "Class" that gives the true class of the image
#                         according to the classification scheme defined at the start of this file.
#       data_to_classify: csv file formatted the same way as training_data; it will NOT be used for training
#                         the classifier, but for running and testing it
#
def parse_arguments():
    parser = argparse.ArgumentParser(description='Processes files ')
    parser.add_argument('-k', type=int)
    parser.add_argument('-f', type=int)
    parser.add_argument('-s', '--sim_id', nargs='?', type=int)
    parser.add_argument('-u', '--unseen', action='store_true')
    parser.add_argument('training_data')
    parser.add_argument('data_to_classify')
    params = parser.parse_args()

    if params.sim_id < 0 or params.sim_id > 5:
        print('Argument sim_id must be a number from 1 to 5')
        return None

    opt = {'k': params.k,
           'f': params.f,
           'sim_id': params.sim_id,
           'training_data': params.training_data,
           'data_to_classify': params.data_to_classify,
           'mode': params.unseen
           }
    return opt


if __name__ == '__main__':
    main()
