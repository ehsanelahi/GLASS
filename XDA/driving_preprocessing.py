import pandas as pd
import ast  # for safely evaluating strings as lists

def split_confidence_columns(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Function to convert string representation of list to actual list
    def parse_confidence(conf_str):
        try:
            # Remove brackets and split by whitespace
            conf_values = conf_str.strip('[]').split()
            # Convert to floats
            return [float(x) for x in conf_values]
        except:
            return [None, None, None]
    
    # Apply the parsing function to the confidence column
    confidence_values = df['confidence'].apply(parse_confidence)
    
    # Create new columns for each requirement
    df['req_0'] = confidence_values.str[0]
    df['req_1'] = confidence_values.str[1]
    df['req_2'] = confidence_values.str[2]
    #df['req_3'] = confidence_values.str[3]
    
    # Optionally drop the original confidence column
    # df = df.drop('confidence', axis=1)
    
    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)
    print(f"Processed file saved to {output_file}")

# Example usage:
input_csv = 'drive_custom.csv'
output_csv = 'drive_custom_processed.csv'
split_confidence_columns(input_csv, output_csv)