import os
import Task_1_5
import Task_2
import numpy as np

# Conducts cross-validation with 'f' defining the number of folds.
# Updates the 'processed' data structure with details for each round in the cross-validation.
# Appends columns indicating the fold number of each image and its assigned class during testing (if in the testing fold).
# After completing all rounds, adds average measures at the end.
# CSV writing is handled in a separate function, and the expected structure resembles:
# Example:
# path                  class                  round_1                        class_round_1                     round_2      class_round_2...
# <from training data>  <from training data>   in what fold it is in round 1  how it was classified in round 1
# ...
#
# avg_precision <value>
# avg_recall   <value>
# ...
def cross_evaluate_knn(training_data, k, sim_id, f):
    # Initialize a matrix for round-class identifiers
    rc = [[f'Round_{i}', f'Class_{i}'] for i in range(1, f + 1)]
    processed = [training_data[0] + [y for tp in rc for y in tp]]

    # Remove the ['Class','Path'] entry
    training_data.pop(0)
    
    # Shuffle the training data
    np.random.shuffle(training_data)
    
    # Split the list into f sublists
    training_data = list(split_list(training_data, int((len(training_data)/f))))

    # Initialize total evaluation metrics
    t_precision = float(0)
    t_recall = float(0)
    t_f_measure = float(0)
    t_accuracy = float(0)

    data_to_add = []
    
    # Iterate through the training data
    for i in training_data:
        # Get all lists except the current one as other_data
        other_data = [x for x in training_data if x != i]
        # Flatten the other_data
        other_data = [x for sublist in other_data for x in sublist]
        
        # Apply kNN algorithm to obtain temp_process
        temp_process = Task_1_5.kNN(other_data, k, sim_id, i)
        temp_process.pop(0)  # Remove the first image as it is duplicated

        # Iterate through images in i
        for j in range(len(i)):
            unadded_data = [i[j][0], i[j][1]]
            for x in range(f*2):
                if x % 2 == 0:
                    unadded_data.append(training_data.index(i)+1)
                else:
                    unadded_data.append(" ")
            
            # Update unadded_data with kNN results
            unadded_data[(training_data.index(i)*2)+3] = temp_process[j][2]
            processed.append(unadded_data)

        # Extract unique class options from temp_process
        options = set(x[1] for x in temp_process)
        
        # Evaluate the performance using temp_process and update total metrics
        precision, recall, f_measure, accuracy = Task_2.evaluate(temp_process, options)
        t_precision += precision
        t_recall += recall
        t_f_measure += f_measure
        t_accuracy += accuracy

    # Calculate average metrics
    avg_precision = t_precision / f
    avg_recall = t_recall / f
    avg_f_measure = t_f_measure / f
    avg_accuracy = t_accuracy / f

    # Create a matrix with average metrics
    h = ['avg_precision', 'avg_recall', 'avg_f_measure', 'avg_accuracy']
    v = [avg_precision, avg_recall, avg_f_measure, avg_accuracy]
    r = [[h[i], v[i]] for i in range(len(h))]

    # Return the processed data along with average metrics
    return processed + r

# Loops through the list, splitting each list into a size of n
def split_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]
    
# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier using the cross-validation technique. The results
# are written into an appropriate csv file.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    print('Evaluating kNN')
    result = cross_evaluate_knn(training_data, opts['k'], opts['sim_id'], opts['f'])
    path = os.path.dirname(os.path.realpath(opts['training_data']))
    out = f'{path}/{Task_1_5.student_id}_cross_validation.csv'
    print(f'Writing data to {out}')
    Task_1_5.write_csv_file(out, result)


if __name__ == '__main__':
    main()
