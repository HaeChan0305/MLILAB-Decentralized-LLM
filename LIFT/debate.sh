STEP=200
for i in {1..15}
do
    echo "checkpoint-$((STEP * i))"
    # python processing_test_debate.py -c $((STEP * i)) -c1 1 -c2 2 -r 1
    # python processing_test_debate.py -c $((STEP * i)) -c1 2 -c2 1 -r 1
    python debate.py -c /workspace/output_qwen_4_1/checkpoint-$((STEP * i)) -o ./results/output_qwen_4_1/checkpoint-$((STEP * i)) -b 8 -r 1
    # python debate.py -c /workspace/output_qwen_4_2/checkpoint-$((STEP * i)) -b 8 -r 1
    # python compute_metric_pollen.py -c1 ./results/output_qwen_4_2/checkpoint-$((STEP * i))/result.json
    # cp /workspace/output_qwen_4_2/checkpoint-$((STEP * i))/result.json ./results/output_qwen_4_2/checkpoint-$((STEP * i))/result.json
done

