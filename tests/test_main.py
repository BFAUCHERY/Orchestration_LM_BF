from orchestration_lm_bf.__main__ import main

def test_main_executes_without_error():
    # Just test the entry point doesn't raise
    try:
        main()
    except SystemExit:
        pass