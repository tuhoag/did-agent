#!/usr/bin/env bash

LOG_FILE="app.log"
PID_FILE="app.pid"

# Command to run
COMMAND="uv run kg_construction.py"

# Run in background, redirect stdout & stderr to log
nohup bash -c "$COMMAND" >> "$LOG_FILE" 2>&1 &

# Save PID of background process
echo $! > "$PID_FILE"

echo "Started process with PID $(cat "$PID_FILE")"
