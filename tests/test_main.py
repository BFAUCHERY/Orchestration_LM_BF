from orchestration_lm_bf.__main__ import main

def test_main_executes_without_error(monkeypatch):
    monkeypatch.setattr("sys.argv", ["kedro", "run"])
    try:
        main()
    except SystemExit:
        pass  # Some CLI apps exit after running
    
def test_main_as_script(monkeypatch):
    import builtins
    monkeypatch.setattr("sys.argv", ["__main__.py"])

    __name__original = builtins.__name__
    builtins.__name__ = "__main__"
    try:
        from orchestration_lm_bf import __main__  # force l'exécution du bloc if __name__ == "__main__"
    except SystemExit:
        pass
    finally:
        builtins.__name__ = __name__original