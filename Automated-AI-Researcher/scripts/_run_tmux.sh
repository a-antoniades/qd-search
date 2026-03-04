#!/bin/bash
export GEMINI_API_KEY=AIzaSyD-DXFBXn4eCltOe7uduB5OgIosmud6GSA
cd /share/edc/home/antonis/qd-search/Automated-AI-Researcher
conda run --live-stream -n aira-dojo bash scripts/run_evo_experiment.sh 4,5,6,7 2>&1 | tee evo_experiment.log
