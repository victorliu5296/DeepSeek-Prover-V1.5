import os
import subprocess

# Test running the command with relative path
lean_workspace = 'mathlib4'
lake_path = '/home/defaultuser/.elan/bin/lake'

try:
    outputs = subprocess.run(
        [lake_path, "exe", "repl"], 
        capture_output=True, 
        text=True, 
        cwd=lean_workspace
    )
    outputs.check_returncode()
    print("Command executed successfully:", outputs.stdout)
except subprocess.CalledProcessError as e:
    print("Command failed with error:", e)
