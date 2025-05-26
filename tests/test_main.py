from orchestration_lm_bf.__main__ import main

def test_main_executes_without_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["kedro", "run"])
    try:
        main()
    except SystemExit:
        pass  # Some CLI apps exit after running