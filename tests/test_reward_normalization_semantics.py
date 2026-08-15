from openrlhf.cli.reward_normalization import (
    AGENT_FUNC,
    HTTP_REWARD_API,
    LOCAL_REWARD_MODEL,
    PYTHON_REWARD_FUNC,
    as_reward_url_list,
    classify_reward_source,
    reward_normalization_warning,
)


def test_as_reward_url_list_normalizes_supported_shapes():
    assert as_reward_url_list(None) == []
    assert as_reward_url_list("http://host:5000/get_reward") == ["http://host:5000/get_reward"]
    assert as_reward_url_list(" http://host:5000/get_reward ") == ["http://host:5000/get_reward"]
    assert as_reward_url_list([" a.py ", "", "  ", "http://host "]) == ["a.py", "http://host"]


def test_classify_reward_source_local_reward_model():
    assert classify_reward_source() == LOCAL_REWARD_MODEL


def test_classify_reward_source_python_reward_func():
    assert classify_reward_source("/tmp/reward_func.py") == PYTHON_REWARD_FUNC
    assert classify_reward_source(" /tmp/reward_func.py ") == PYTHON_REWARD_FUNC


def test_classify_reward_source_preserves_case_sensitive_python_suffix():
    assert classify_reward_source("/tmp/reward_func.PY") == HTTP_REWARD_API


def test_classify_reward_source_http_reward_api():
    assert classify_reward_source("http://host:5000/get_reward") == HTTP_REWARD_API


def test_classify_reward_source_agent_func_takes_precedence():
    assert classify_reward_source(["agent"], agent_func_path="/tmp/agent_func.py") == AGENT_FUNC


def test_classify_reward_source_uses_first_split_reward_endpoint():
    assert classify_reward_source(["/tmp/reward_func.py", "http://host:5000/get_reward"]) == PYTHON_REWARD_FUNC
    assert classify_reward_source(["http://host:5000/get_reward", "/tmp/reward_func.py"]) == HTTP_REWARD_API


def test_reward_normalization_warning_is_only_for_enabled_remote_or_custom_rewards():
    assert reward_normalization_warning(False, PYTHON_REWARD_FUNC) is None
    assert reward_normalization_warning(True, LOCAL_REWARD_MODEL) is None

    warning = reward_normalization_warning(True, PYTHON_REWARD_FUNC)

    assert "--reward.normalize_enable does not transform rewards" in warning
    assert "Python reward_func files" in warning
    assert "local reward/critic model heads" in warning
    assert "critic values may still use this setting" in warning


def test_reward_normalization_warning_describes_agent_and_http_sources():
    agent_warning = reward_normalization_warning(True, AGENT_FUNC)
    http_warning = reward_normalization_warning(True, HTTP_REWARD_API)

    assert "custom agent_func executors" in agent_warning
    assert "inside the agent executor" in agent_warning
    assert "remote reward APIs" in http_warning
    assert "pass --reward.normalize_enable to that server command" in http_warning
