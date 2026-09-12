from openrlhf.env_utils import configure_vllm_allocator_env


def test_removes_sole_legacy_expandable_segments_option():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    updated = configure_vllm_allocator_env(True, env)

    assert updated == ["PYTORCH_CUDA_ALLOC_CONF"]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env


def test_preserves_other_allocator_options():
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128, expandable_segments:True, garbage_collection_threshold:0.8"
    }

    configure_vllm_allocator_env(True, env)

    expected = "max_split_size_mb:128,garbage_collection_threshold:0.8"
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == expected


def test_preserves_disabled_and_malformed_options():
    original = "expandable_segments:False,expandable_segments,max_split_size_mb:64"
    env = {"PYTORCH_CUDA_ALLOC_CONF": original}

    updated = configure_vllm_allocator_env(True, env)

    assert updated == []
    assert env["PYTORCH_CUDA_ALLOC_CONF"] == original


def test_sanitizes_legacy_and_current_environment_names():
    env = {
        "PYTORCH_CUDA_ALLOC_CONF": "EXPANDABLE_SEGMENTS : 1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:true,backend:cudaMallocAsync",
    }

    updated = configure_vllm_allocator_env(True, env)

    assert updated == ["PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"]
    assert "PYTORCH_CUDA_ALLOC_CONF" not in env
    assert env["PYTORCH_ALLOC_CONF"] == "backend:cudaMallocAsync"


def test_missing_allocator_configuration_is_a_noop():
    env = {"NCCL_DEBUG": "INFO"}

    updated = configure_vllm_allocator_env(True, env)

    assert updated == []
    assert env == {"NCCL_DEBUG": "INFO"}


def test_sleep_mode_disabled_preserves_expandable_segments():
    env = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

    updated = configure_vllm_allocator_env(False, env)

    assert updated == []
    assert env == {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
