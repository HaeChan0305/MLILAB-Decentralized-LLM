STEP=150
for i in {1..3}
do
    echo "checkpoint-$((STEP * i))"
    python gsm8k/evaluate_gsm8k.py -c gsm8k/output_qwen_1_3/checkpoint-$((STEP * i)) -o gsm8k/output_qwen_1_3/checkpoint-$((STEP * i))/result.json -b 32
done

