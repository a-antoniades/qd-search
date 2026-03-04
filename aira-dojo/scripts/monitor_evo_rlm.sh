#!/bin/bash
# Monitor EVO+RLM runs
# Usage: ./monitor_evo_rlm.sh

cd /share/edc/home/antonis/qd-search/aira-dojo

echo "=== EVO+RLM Monitor $(date) ==="
echo ""

# Process count
proc_count=$(ps aux | grep "dojo.main_run" | grep -v grep | wc -l)
echo "Processes: $proc_count"

# Check if any processes died
if [ "$proc_count" -lt 5 ]; then
    echo "WARNING: Expected 5+ processes, got $proc_count"
fi
echo ""

# Log sizes and progress
echo "Run Status:"
echo "----------------------------------------"
for f in logs/qd_study_evo_rlm*.log; do
    if [ -f "$f" ]; then
        task=$(basename $f .log | sed 's/qd_study_evo_rlm_//' | sed 's/_seed1//')
        size=$(du -h "$f" | cut -f1)
        gen=$(grep -o "Generation [0-9]*" "$f" | tail -1 | grep -o "[0-9]*" || echo "0")
        nodes=$(grep -c "Creating node" "$f" 2>/dev/null || echo 0)
        rlm_errors=$(grep -c "RLM selection failed" "$f" 2>/dev/null | tr -d '\n' || echo 0)
        exec_errors=$(grep -c "Execution failed" "$f" 2>/dev/null | tr -d '\n' || echo 0)
        timeouts=$(grep -c "Execution exceeded timeout" "$f" 2>/dev/null | tr -d '\n' || echo 0)

        status="OK"
        [ "$rlm_errors" -gt 5 ] 2>/dev/null && status="RLM_ISSUES"
        [ "$timeouts" -gt 100 ] 2>/dev/null && status="TIMEOUT_LOOP"

        printf "%-35s Gen:%s Nodes:%-3s Size:%-5s Errs:%s/%s/%s [%s]\n" \
            "$task" "$gen" "$nodes" "$size" "$rlm_errors" "$exec_errors" "$timeouts" "$status"
    fi
done
echo ""

# Check for critical errors
critical=$(grep -l "TIMEOUT_LOOP\|Traceback.*Error\|CRITICAL" logs/qd_study_evo_rlm*.log 2>/dev/null | wc -l)
if [ "$critical" -gt 0 ]; then
    echo "ALERT: Critical errors found in $critical log(s)"
fi

echo "----------------------------------------"
echo "Legend: Errs = RLM_fails/Exec_fails/Timeouts"
