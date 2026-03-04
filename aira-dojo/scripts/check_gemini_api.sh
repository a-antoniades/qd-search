#!/bin/bash
# Check if gemini-3-flash-preview API is available

cd /share/edc/home/antonis/qd-search/aira-dojo
source /opt/conda/etc/profile.d/conda.sh
conda activate aira-dojo

JOURNAL_PATH="logs/aira-dojo/user_antonis_issue_QD_STUDY_greedy_gdm/user_antonis_issue_QD_STUDY_greedy_gdm_seed_1_id_7b4249ca381e54335d60f55774c6a3f4de5eee41252e3d0bfef1dc6c/user_antonis_issue_QD_STUDY_greedy_gdm_seed_1_id_7b4249ca381e54335d60f55774c6a3f4de5eee41252e3d0bfef1dc6c_20260208151539314596_Greedy_search_data.json"

timeout 90 python -c "
import os, json, logging
logging.basicConfig(level=logging.WARNING)
from dojo.core.solvers.selection import RLMNodeSelector
from dojo.core.solvers.utils.journal import Journal
from dojo.config_dataclasses.selector.rlm import RLMSelectorConfig

with open('$JOURNAL_PATH') as f:
    journal = Journal.from_export_data({'nodes': json.load(f)['nodes']})

cfg = RLMSelectorConfig(backend='gemini', model_name='gemini-3-flash-preview', max_iterations=5, verbose=False)
result = RLMNodeSelector(cfg=cfg, lower_is_better=False).select(journal, {'num_samples': {'improve': 2}})

if result.selected_nodes:
    print(f'SUCCESS: Selected nodes {[n.step for n in result.selected_nodes]}, Op: {result.operator}')
    exit(0)
else:
    print(f'FAILED: {result.reasoning}')
    exit(1)
" 2>&1
