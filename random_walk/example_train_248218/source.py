

def test_run_should_pass_when_all_examples_pass(self):
    outline = ScenarioOutline('foo.feature', 17, 'Scenario Outline', 'foo')
    outline._scenarios = [Mock(), Mock(), Mock()]
    for scenario in outline._scenarios:
        scenario.run.return_value = False
    runner = Mock()
    context = runner.context = Mock()
    config = runner.config = Mock()
    config.stop = True
    resultFailed = outline.run(runner)
    eq_(resultFailed, False)
