import json

def calculate_accuracy(output_json_path):
    # Read output JSON file
    with open(output_json_path, "r") as f:
        results = json.load(f)
    
    # Calculate total number of samples and correct prediction
    total_samples = len(results)
    correct_predictions = sum(1 for item in results if item[3] == 1)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    
    print(f"Total Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
    
    return accuracy

if __name__ == "__main__":
    # Example
    output_json_path = "data/output/out_easy_no_ambi.json"
    sr = calculate_accuracy(output_json_path)