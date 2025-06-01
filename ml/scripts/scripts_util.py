import subprocess

def run_fit(script_name):
    try:
        result = subprocess.run(
            ["bash", f"ml/scripts/{script_name}.sh"],
            text=True,
            capture_output=False  # Print directly to terminal
        )
        print("Script finished.")
        return result.returncode
    except Exception as e:
        print(f"Error running script: {e}")
        return -1