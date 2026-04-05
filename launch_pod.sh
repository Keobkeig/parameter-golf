#!/bin/bash
# Launch a RunPod pod for parameter-golf training.
# Official template: https://console.runpod.io/hub/template/parameter-golf?id=y5cejece4j
#
# Usage:
#   bash launch_pod.sh              # 1xH100 SXM for testing ($~4/hr)
#   bash launch_pod.sh 8            # 8xH100 SXM for submission (~$20/hr)
#   bash launch_pod.sh 1 h100-nvl   # H100 NVL variant

set -e

GPU_COUNT=${1:-1}
GPU_TYPE=${2:-"NVIDIA H100 80GB HBM3"}  # H100 SXM

TEMPLATE_ID="y5cejece4j"  # Official parameter-golf template

echo "=== Launching RunPod pod ==="
echo "GPUs: $GPU_COUNT x $GPU_TYPE"
echo "Template: $TEMPLATE_ID (parameter-golf official)"
echo ""

POD_JSON=$(runpodctl pod create \
    --template-id "$TEMPLATE_ID" \
    --gpu-id "$GPU_TYPE" \
    --gpu-count "$GPU_COUNT" \
    --name "param-golf-$(date +%m%d-%H%M)" \
    --volume-in-gb 50 \
    --container-disk-in-gb 50 \
    --ssh \
    2>&1)

echo "$POD_JSON"

POD_ID=$(echo "$POD_JSON" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('id',''))" 2>/dev/null || echo "")

if [ -z "$POD_ID" ]; then
    echo ""
    echo "Could not parse pod ID. Check output above."
    exit 1
fi

echo ""
echo "=== Pod created: $POD_ID ==="
echo ""
echo "Waiting for pod to be ready..."

# Poll until running
for i in $(seq 1 30); do
    STATUS=$(runpodctl pod get "$POD_ID" 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('desiredStatus',''))" 2>/dev/null || echo "")
    echo "  Status: $STATUS (${i}/30)"
    if [ "$STATUS" = "RUNNING" ]; then
        break
    fi
    sleep 10
done

echo ""
echo "=== Pod $POD_ID is up ==="
echo ""
echo "--- Next steps ---"
echo ""
echo "1. SSH into the pod:"
echo "   runpodctl ssh connect $POD_ID"
echo ""
echo "2. On the pod, run setup (clones repo + downloads FineWeb):"
echo "   curl -sL https://raw.githubusercontent.com/Keobkeig/parameter-golf/copy-pr549-2026-03-29/runpod_setup.sh | bash"
echo "   # quick subset (1 shard instead of 80):"
echo "   curl -sL https://raw.githubusercontent.com/Keobkeig/parameter-golf/copy-pr549-2026-03-29/runpod_setup.sh | bash -s 1"
echo ""
echo "3. Smoke test (1 GPU, 200 steps — cheap verify before burning 8xH100):"
echo "   cd /workspace/parameter-golf && bash run_smoke_test.sh"
echo ""
echo "4. Full 3-seed submission run:"
echo "   cd /workspace/parameter-golf && bash run_submission.sh"
echo "   cd /workspace/parameter-golf && SEED=42 bash run_submission.sh"
echo "   cd /workspace/parameter-golf && SEED=2025 bash run_submission.sh"
echo ""
echo "5. Pull logs back locally (run on your Mac):"
echo "   # On pod: runpodctl send /workspace/parameter-golf/logs"
echo "   # On Mac: runpodctl receive <code>"
echo ""
echo "6. Stop pod when done:"
echo "   runpodctl pod stop $POD_ID"
echo ""
echo "Pod ID: $POD_ID"
