import subprocess

def run_fit(script_name):
    process = subprocess.Popen(["bash", f"ml/scripts/{script_name}.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")

    process.wait()
    print("Script finished.")