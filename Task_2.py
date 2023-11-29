import Task_1_5

# Computes the precision, recall, f-measure, and accuracy metrics of the kNN classifier
def evaluate_knn(training_data, k, sim_id, data_to_classify):
    # Initialize evaluation metrics
    precision = float(0)
    recall = float(0)
    f_measure = float(0)
    accuracy = float(0)

    # Apply kNN algorithm to obtain processed results
    processed = Task_1_5.kNN(training_data, k, sim_id, data_to_classify)
    
    # Extract unique class options from processed data
    options = set([x[1] for x in processed])
    processed.pop(0)  # Remove the header row
    options.remove('Class')  # Remove the 'Class' label from options

    # Evaluate the performance using the processed results
    precision, recall, f_measure, accuracy = evaluate(processed, options)
    
    # Return the calculated evaluation metrics
    return precision, recall, f_measure, accuracy

def evaluate(processed, options):
    # Initialize evaluation metrics
    precision = float(0)
    recall = float(0)
    f_measure = float(0)
    accuracy = float(0)
    
    # Loop through all classes
    for y in options: 
        
        # Initialize counters for precision and recall
        precision_counter = 0
        precision_total_counter = 0
        recall_counter = 0
        recall_total_counter = 0
            
        for x in processed:
            # Counter for precision
            if x[2] == x[1] == y:  # x[2] is the predicted class
                precision_counter += 1
            else:
                precision_total_counter += 1

            # Counter for Recall
            if x[1] == y:  # x[1] is the actual class
                recall_counter += 1
            else:
                recall_total_counter += 1
                
        # Calculate precision and recall for the current class
        precision += (precision_counter / (precision_counter + precision_total_counter))
        recall += (recall_counter / (recall_counter + recall_total_counter))

    # Calculate accuracy
    accuracy_counter = sum(1 for x in processed if x[1] == x[2])
    accuracy = accuracy_counter / len(processed)

    # Calculate final precision, recall, accuracy, and F-measure
    precision = precision / len(options)  # Divide the total precisions by the number of classes
    recall = recall / len(options)  # Divide the total recalls by the number of classes
    f_measure = (2 * precision * recall) / (precision + recall)  # F-Measure algorithm
    
    return precision, recall, f_measure, accuracy


# This function reads the necessary arguments (see parse_arguments function in Task_1_5),
# and based on them evaluates the kNN classifier.
def main():
    opts = Task_1_5.parse_arguments()
    if not opts:
        exit(1)
    print(f'Reading data from {opts["training_data"]} and {opts["data_to_classify"]}')
    training_data = Task_1_5.read_csv_file(opts['training_data'])
    data_to_classify = Task_1_5.read_csv_file(opts['data_to_classify'])
    print('Evaluating kNN')
    result = evaluate_knn(training_data, opts['k'], opts['sim_id'], data_to_classify)
    print('Result: precision {}; recall {}; f-measure {}; accuracy {}'.format(*result))


if __name__ == '__main__':
    main()
