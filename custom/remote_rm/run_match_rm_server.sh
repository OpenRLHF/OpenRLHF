#!/bin/bash
SERVER_DIR=./

uvicorn --app-dir "${SERVER_DIR}" match_rm_server:app --host 0.0.0.0 --port 1234 >& match_rm.log &