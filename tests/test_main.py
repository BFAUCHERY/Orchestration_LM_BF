import subprocess

def test_kedro_main_runs():
    result = subprocess.run(["python", "src/orchestration_lm_bf/__main__.py", "help"], capture_output=True)
    assert result.returncode == 0