echo "checkpoint-0"
# mkdir /workspace/output_qwen_4_0/checkpoint-0
python evaluate_pollen.py -o /workspace/output_qwen_4_0/checkpoint-0/result.json -b 128

STEP=400
for i in {1..15}
do
    echo "checkpoint-$((STEP * i))"
    python evaluate_pollen.py -c /workspace/output_qwen_4_0/checkpoint-$((STEP * i)) -o /workspace/output_qwen_4_0/checkpoint-$((STEP * i))/result.json -b 128
done