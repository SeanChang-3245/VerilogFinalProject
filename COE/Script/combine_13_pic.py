import os

def combine_coe_files(input_files, output_file):
    combined_lines = []
    initialization_vector = []

    for file in input_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            if not combined_lines:
                combined_lines.extend(lines[:2])  # Add the initial string only once
            initialization_vector.extend(lines[2:])  # Combine the number rows

    # Ensure each row is separated by a comma, and the last one ends with a semicolon
    initialization_vector = [line.strip() for line in initialization_vector]
    initialization_vector = ',\n'.join(initialization_vector[:-1]) + ',\n' + initialization_vector[-1] + ';\n'

    with open(output_file, 'w') as f:
        f.writelines(combined_lines)
        f.write(initialization_vector)

if __name__ == "__main__":
    input_files = []
    for i in range(13):
        input_files.append(f"c:/Projects/VerilogFinalProject/COE/{i+40}_combined.coe")
    print(len(input_files))
    output_file = "c:/Projects/VerilogFinalProject/COE/orange_output.coe"
    combine_coe_files(input_files, output_file)
