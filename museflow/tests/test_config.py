from museflow import config


def test_configure_function():
    @config.configurable(['obj'])
    def f(cfg, a, b, c, d, e=5):
        return cfg['obj'].configure(dict, a=a, b=b, c=c, d=d, e=e)

    result = config.Configuration({
        'a': 10, 'b': 2, 'c': 3,
        'obj': {'a': 1, 'f': 6}
    }).configure(f, a=0, c=300, d=4)
    expected_result = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    assert result == expected_result


def test_configure_list():
    @config.configurable(['items'])
    def f(cfg):
        return cfg['items'].configure(f)

    result = config.Configuration({
        'items': [
            {'class': dict, 'x': 1},
            {'items': [{'class': dict, 'y': 2}, {'class': dict, 'z': 3}]}
        ]
    }).configure(f)
    expected_result = [{'x': 1}, [{'y': 2}, {'z': 3}]]
    assert result == expected_result
