def pytest_terminal_summary(terminalreporter):
    total = terminalreporter._numcollected
    passed = len(terminalreporter.stats.get('passed', []))
    rate = (passed / total) * 100 if total > 0 else 0
    terminalreporter.write_sep(
        "-", f"Taux de réussite des tests : {rate:.2f}% ({passed}/{total})"
    )