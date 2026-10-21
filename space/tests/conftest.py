def pytest_configure(config):
    config.addinivalue_line("markers", "slow: loads the real Falcon weights (cached from Hugging Face)")
