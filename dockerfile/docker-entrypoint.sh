#!/bin/bash

USER=${LOCAL_USER:-"root"}

if [[ "${USER}" != "root" ]]; then
    USER_ID=${LOCAL_USER_ID:-9001}
    echo ${USER}
    echo ${USER_ID}

    chown ${USER_ID} /home/${USER}
    useradd --shell /bin/bash -u ${USER_ID} -o -c "" -m ${USER}
    usermod -a -G root ${USER}
    adduser ${USER} sudo

    # user:password
    echo "${USER}:123" | chpasswd

    export HOME=/home/${USER}
    export PATH=/home/${USER}/.local/bin/:$PATH
else
    export PATH=/root/.local/bin/:$PATH
fi

# CUDA forward compatibility: the stack is CUDA 13, which a host NVIDIA driver older than 580
# cannot run natively. On data center GPUs the cuda-compat libcuda bridges to the older driver.
# Put it on the loader path only then, so hosts with a newer driver keep using their own.
# OPENRLHF_CUDA_COMPAT=1 forces it on, =0 off.
COMPAT_DIR=/usr/local/cuda/compat
if [[ -d "${COMPAT_DIR}" ]]; then
    # nvidia-smi reads the host driver; /proc is the fallback when it is not mounted.
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [[ -z "${DRIVER}" ]] && DRIVER=$(grep -oE 'Kernel Module +[0-9]+' /proc/driver/nvidia/version 2>/dev/null | grep -oE '[0-9]+$')
    case "${OPENRLHF_CUDA_COMPAT:-auto}" in
        1) USE_COMPAT=1 ;;
        0) USE_COMPAT= ;;
        *) if [[ -n "${DRIVER}" && "${DRIVER}" -lt 580 ]]; then USE_COMPAT=1; else USE_COMPAT=; fi ;;
    esac
    if [[ -n "${USE_COMPAT}" ]]; then
        export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH}"
        echo "[openrlhf] CUDA forward compatibility on (host driver ${DRIVER:-unknown}): ${COMPAT_DIR}"
    fi
fi

cd $HOME
exec gosu ${USER} "$@"