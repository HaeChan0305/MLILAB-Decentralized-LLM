#for data in agnews mr r8 sst2
# data=mr
# STEP=283
# for i in {1..3}
# do
#     echo ${data}
#     echo "checkpoint-$((STEP * i))"
    
#     mkdir ./output_test/${data}/checkpoint-$((STEP * i))
#     python test.py -d $data -c ./ft_model_test/${data}/checkpoint-$((STEP * i)) -o ./output_test/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17 -e True
#     # python compute_metric.py -d $data -r ./output_5_0_2/checkpoint-0/${data}_result.json -e False
# done

data=r8
STEP=171
for i in {3..10}
do
    echo ${data}
    echo "checkpoint-$((STEP * i))"
    
    mkdir ./output_5_0_2/${data}/checkpoint-$((STEP * i))
    python test.py -d $data -c ./ft_model_5_0_1/${data}/checkpoint-$((STEP * i)) -o ./output_5_0_2/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17 -e True
    # python compute_metric.py -d $data -r ./output_5_0_2/checkpoint-0/${data}_result.json -e False
done