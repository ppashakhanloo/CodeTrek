def sample_function():
    ages = {'Jim': 30, 'Pam': 28, 'Kevin': 33}
    try:
        a = ages['Michael']
    except KeyError:
        raise KeyError
    return 0
