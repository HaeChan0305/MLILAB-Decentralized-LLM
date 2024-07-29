STEP=200
for i in {1..15}
do
    echo "checkpoint-$((STEP * i))"
    python evaluate_pollen.py -c output_qwen_3_2/checkpoint-$((STEP * i)) -o output_qwen_3_2/checkpoint-$((STEP * i))/result.json -b 128
done
