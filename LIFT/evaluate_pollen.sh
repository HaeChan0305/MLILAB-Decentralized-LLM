STEP=200
for i in {1..15}
do
    echo "checkpoint-$((STEP * i))"
    python evaluate_pollen.py -c /workspace/output_qwen_4_1/checkpoint-$((STEP * i)) -o /workspace/output_qwen_4_1/checkpoint-$((STEP * i))/result.json -b 64
done
