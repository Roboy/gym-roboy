#!/usr/bin/env bash
python3 -m pytest --run-integration -v -x $@  # -x to stop on the first failed test
