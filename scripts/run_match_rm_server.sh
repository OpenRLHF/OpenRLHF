#!/bin/bash
ROOT_DIR=/apdcephfs/share_300000800/user/lfsong/exp.tencent_chat/OpenRLHF  # "$(dirname "$0")"

uvicorn --app-dir "${ROOT_DIR}" openrlhf.remote_rm.match_rm_server:app --host 0.0.0.0 --port 1234 >& match_rm.log &