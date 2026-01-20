#!/bin/bash
set -e

# Run All Backend Tests
# This script runs all backend tests and compares the results

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Configuration
export NUM_GPUS=${NUM_GPUS:-8}
export MAX_SAMPLES=${MAX_SAMPLES:-500}
export LOG_DIR="${SCRIPT_DIR}/logs"
export MODEL=${MODEL:-"meta-llama/Meta-Llama-3-8B"}
export DATASET=${DATASET:-"Open-Orca/OpenOrca"}

mkdir -p "${LOG_DIR}"

echo "======================================"
echo "OpenRLHF Backend Test Suite"
echo "======================================"
echo "GPUs: ${NUM_GPUS}"
echo "Model: ${MODEL}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Log directory: ${LOG_DIR}"
echo "======================================"
echo ""

# Function to run a test with error handling
run_test() {
    local test_name=$1
    local test_script=$2
    
    echo "------------------------------------"
    echo "Running: ${test_name}"
    echo "------------------------------------"
    
    if bash "${test_script}"; then
        echo "[PASSED] ${test_name}"
        return 0
    else
        echo "[FAILED] ${test_name}"
        return 1
    fi
}

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Run DeepSpeed test
if [[ "${SKIP_DEEPSPEED}" != "1" ]]; then
    if run_test "DeepSpeed SFT" "./test_sft_deepspeed.sh"; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
fi

# Run FSDP2 test
if [[ "${SKIP_FSDP2}" != "1" ]]; then
    if run_test "FSDP2 SFT" "./test_sft_fsdp2.sh"; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
fi

# Run FSDP2 + AutoTP test
if [[ "${SKIP_FSDP2_TP}" != "1" ]]; then
    if run_test "FSDP2 + AutoTP SFT" "./test_sft_fsdp2_tp.sh"; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
fi

# Run FSDP2 + Ring Attention test
if [[ "${SKIP_RING}" != "1" ]]; then
    if run_test "FSDP2 + Ring Attention SFT" "./test_sft_fsdp2_ring.sh"; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
fi

# Run FSDP2 + AutoTP + Ring Attention combined test
if [[ "${SKIP_COMBINED}" != "1" ]]; then
    if run_test "FSDP2 + AutoTP + Ring Attention SFT" "./test_sft_fsdp2_tp_ring.sh"; then
        ((TESTS_PASSED++))
    else
        ((TESTS_FAILED++))
    fi
fi

echo ""
echo "======================================"
echo "Test Results Summary"
echo "======================================"
echo "Passed: ${TESTS_PASSED}"
echo "Failed: ${TESTS_FAILED}"
echo "======================================"

# Compare DeepSpeed vs FSDP2 if both tests were run
if [[ -f "${LOG_DIR}/deepspeed_test.log" ]] && [[ -f "${LOG_DIR}/fsdp2_test.log" ]]; then
    echo ""
    echo "======================================"
    echo "Comparing DeepSpeed vs FSDP2"
    echo "======================================"
    python compare_backends.py \
        --deepspeed_log "${LOG_DIR}/deepspeed_test.log" \
        --fsdp2_log "${LOG_DIR}/fsdp2_test.log" \
        --threshold 0.05
fi

# Final status
if [[ ${TESTS_FAILED} -eq 0 ]]; then
    echo ""
    echo "All tests passed!"
    exit 0
else
    echo ""
    echo "Some tests failed!"
    exit 1
fi
