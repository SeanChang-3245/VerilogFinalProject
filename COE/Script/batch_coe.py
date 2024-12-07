import subprocess
import os

def run_exe_with_input(exe_path, input1, input2, input3):
    # Prepare the input string
    input_data = f"{input1}\n{input2}\n{input3}\n"
    
    # Run the .exe file with the input data
    process = subprocess.Popen(exe_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate(input=input_data)
    
    # Print the output and error (if any)
    print("Output:\n", stdout)
    if stderr:
        print("Error:\n", stderr)

if __name__ == "__main__":
    exe_path = "./PicTrans.exe"

    for i in range(54):
        input1 = 32
        input2 = 46
        input3 = f"./cards/{i+1}.jpg"
        
        run_exe_with_input(exe_path, input1, input2, input3)
        output_filename = f"./out.coe"
        new_filename = f"./{i+1}.coe"
        
        try:
            os.rename(output_filename, new_filename)
            print(f"Renamed {output_filename} to {new_filename}")
        except FileNotFoundError:
            print(f"File {output_filename} not found.")
        except Exception as e:
            print(f"Error renaming file: {e}")