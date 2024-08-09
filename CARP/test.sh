data=mr
STEP=283
# for i in {1..10}
# do
#     echo ${data}
#     echo "checkpoint-$((STEP * i))"
    
#     mkdir ./output_5_0_3/${data}/checkpoint-$((STEP * i))
#     python test.py -d $data -c ./ft_model_5_0_3/${data}/checkpoint-$((STEP * i)) -o ./output_5_0_3/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
# done

mkdir ./output_5_0_3/${data}/checkpoint-0
python test.py -d $data -o ./output_5_0_3/${data}/checkpoint-0/${data}_result.json -b 17


data=r8
STEP=171
# for i in {1..10}
# do
#     echo ${data}
#     echo "checkpoint-$((STEP * i))"
    
#     mkdir ./output_5_0_3/${data}/checkpoint-$((STEP * i))
#     python test.py -d $data -c ./ft_model_5_0_3/${data}/checkpoint-$((STEP * i)) -o ./output_5_0_3/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
#     # python compute_metric.py -d $data -r ./output_5_0_2/checkpoint-0/${data}_result.json -e False
# done

mkdir ./output_5_0_3/${data}/checkpoint-0
python test.py -d $data -o ./output_5_0_3/${data}/checkpoint-0/${data}_result.json -b 17