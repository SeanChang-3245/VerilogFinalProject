import os

def combine_coe_files(input_files, output_file):
    initial_lines = []
    data_lines = []

    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            if not initial_lines:
                initial_lines = lines[:2]  # Assuming the first two lines are the initial strings
            data_lines.extend(lines[2:])

    with open(output_file, 'w') as f:
        f.writelines(initial_lines)
        f.writelines(data_lines)

if __name__ == "__main__":
    input_files = []
    
    for i in range(13) :
        input_files.append(f"./{i+40}.coe")
    
    output_file = "c:/Projects/VerilogFinalProject/COE/orange.coe"
    combine_coe_files(input_files, output_file)
