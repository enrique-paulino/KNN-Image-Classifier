# kNN Image Classifier

This repository contains Python scripts for implementing a k-nearest neighbours (kNN) image classifier. The classifier is designed to work with image data and provides various similarity measures for comparison.

## Contents

1. **Task_1_5.py**
   - Main implementation of the kNN classifier.
   - Provides functions for calculating Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Spectral Angle Mapper (SAM) as similarity measures.
   - Includes additional functions for remaking MSE and RMSE (Question 5).

2. **Task_2.py**
   - Evaluates the kNN classifier by computing precision, recall, F-measure, and accuracy metrics.
   - Utilises the kNN implementation from Task_1_5 for processing.

3. **Task_3.py**
   - Implements cross-validation with a specified number of folds (f).
   - Evaluates the kNN classifier's performance across multiple rounds of cross-validation.
   - Outputs a comprehensive CSV file with detailed results and average metrics.

## Usage

### Task 1-5 (kNN Classifier)

To run the kNN classifier, execute the following command:

```
python Task_1_5.py -k <k_value> -s <similarity_id> -u <unseen_flag> <training_data_file> <data_to_classify_file>
```

- `<k_value>`: Number of neighbours to consider.
- `<similarity_id>`: ID specifying the similarity measure to use.
- `<unseen_flag>`: Use this flag if the data is "unseen" and results need to be written to a file.
- `<training_data_file>`: CSV file with training data.
- `<data_to_classify_file>`: CSV file with data to classify.

### Task 2 (Evaluation)

To evaluate the kNN classifier, use the following command:

```
python Task_2.py -k <k_value> -s <similarity_id> <training_data_file> <data_to_classify_file>
```

- `<k_value>`: Number of neighbours to consider.
- `<similarity_id>`: ID specifying the similarity measure to use.
- `<training_data_file>`: CSV file with training data.
- `<data_to_classify_file>`: CSV file with data to classify.

### Task 3 (Cross-validation)

To perform cross-validation, execute the following command:

```
python Task_3.py -k <k_value> -s <similarity_id> -f <num_folds> <training_data_file>
```

- `<k_value>`: Number of neighbours to consider.
- `<similarity_id>`: ID specifying the similarity measure to use.
- `<num_folds>`: Number of folds for cross-validation.
- `<training_data_file>`: CSV file with training data.

The results will be written to a CSV file named `<student_id>_cross_validation.csv` in the same directory as the training data file.

## CSV File Format

The input CSV files for training and classification should follow a specific format with two columns:

1. **Path to Image (Column 1):**
   - This column should contain the path to each image file. It is recommended to use absolute paths or paths relative to the location of the CSV file.

2. **Class (Column 2):**
   - The second column represents the class label for each corresponding image in the first column. Classes should adhere to the predefined classification scheme, including categories such as 'Female,' 'Male,' 'Primate,' 'Rodent,' and 'Food.'

### Example:

```csv
Path,Class
/path/to/image1.jpg,Female
/path/to/image2.jpg,Male
/path/to/image3.jpg,Primate
/path/to/image4.jpg,Rodent
/path/to/image5.jpg,Food
```
Ensure that your CSV files strictly adhere to this format for the kNN classifier to function correctly. Feel free to adjust the wording or add any additional details based on your specific use case or requirements.



