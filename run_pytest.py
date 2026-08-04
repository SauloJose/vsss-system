import subprocess
import sys

cmd = [r"C:\Users\saulo\AppData\Local\Programs\Python\Python311\python.exe", '-m', 'pytest', '-q', 'tests/test_interface_app.py', 'tests/test_pipeline_flow.py']
result = subprocess.run(cmd, cwd=r'c:\Users\saulo\Documents\GitHub\vsss-system', capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
print(f'EXIT_CODE={result.returncode}')
