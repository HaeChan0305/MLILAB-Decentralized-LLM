STEP=200
for i in {1..15}
do
    # echo "checkpoint-$((STEP * i))"
    # python processing_test_debate.py -c $((STEP * i)) -c1 1 -c2 2 -r 3
    # python processing_test_debate.py -c $((STEP * i)) -c1 2 -c2 1 -r 3
    # python debate.py -c output_qwen_3_1/checkpoint-$((STEP * i)) -b 8 -r 3
    # python debate.py -c output_qwen_3_2/checkpoint-$((STEP * i)) -b 8 -r 3
    python compute_metric_pollen.py -c1 output_qwen_3_1/checkpoint-$((STEP * i))/result_round_3.json -c2 output_qwen_3_2/checkpoint-$((STEP * i))/result_round_3.json
done

