def combine_coe_file(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    header = lines[:2]
    data = ''.join(line.strip().replace(',', '') for line in lines[2:])

    with open(output_file, 'w') as file:
        file.writelines(header)
        file.write(data)

# Example usage
# combine_coe_file('c:/Projects/VerilogFinalProject/COE/1.coe', 'c:/Projects/VerilogFinalProject/COE/1_combined.coe')

for i in range (54):
    input_file = f"./{i+1}.coe"
    output_file = f"./{i+1}_combined.coe"
    
    combine_coe_file(input_file, output_file)