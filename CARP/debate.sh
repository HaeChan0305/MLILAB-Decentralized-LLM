data=mr
STEP=141
client1=1
client2=2
round=1

python pre_processing.py -d $data -c1 $((client1)) -c2 $((client2)) -r $((round))
python pre_processing.py -d $data -c1 $((client2)) -c2 $((client1)) -r $((round))
python test.py -d ./output_5_2_$((client1))/${data}/${data}_test_round_$((round)).jsonl \
            -c ./ft_model_5_1_$((client1))/${data}/checkpoint-705 \
            -o ./output_5_2_$((client1))/${data}/${data}_result_round_$((round)).json \
            -b 17

python test.py -d ./output_5_2_$((client2))/${data}/${data}_test_round_$((round)).jsonl \
            -c ./ft_model_5_1_$((client2))/${data}/checkpoint-423 \
            -o ./output_5_2_$((client2))/${data}/${data}_result_round_$((round)).json \
            -b 17
# python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result.json


# data=r8
# STEP=85
# client1=1
# client2=2

# for round in {3..8}
# do
#     for i in {1..10}
#     do
#         python pre_processing.py -d $data -c $((STEP * i)) -c1 $((client1)) -c2 $((client2)) -r $((round))
#         python pre_processing.py -d $data -c $((STEP * i)) -c1 $((client2)) -c2 $((client1)) -r $((round))
#         python test.py -d ./output_5_1_$((client1))/${data}/checkpoint-$((STEP * i))/${data}_test_round_$((round)).jsonl \
#                     -c ./ft_model_5_1_$((client1))/${data}/checkpoint-$((STEP * i)) \
#                     -o ./output_5_1_$((client1))/${data}/checkpoint-$((STEP * i))/${data}_result_round_$((round)).json \
#                     -b 3
        
#         python test.py -d ./output_5_1_$((client2))/${data}/checkpoint-$((STEP * i))/${data}_test_round_$((round)).jsonl \
#                     -c ./ft_model_5_1_$((client2))/${data}/checkpoint-$((STEP * i)) \
#                     -o ./output_5_1_$((client2))/${data}/checkpoint-$((STEP * i))/${data}_result_round_$((round)).json \
#                     -b 3
#         # python compute_metric.py -d $data -r ./output_5_1_${client}/${data}/checkpoint-$((STEP * i))/${data}_result.json
#     done
# done