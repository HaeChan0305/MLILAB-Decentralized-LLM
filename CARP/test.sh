# for data in sst2 mr r8 agnews
for data in sst2 mr agnews r8
do
    echo $data
    python test.py -d $data -o ./output_5_0_1/checkpoint-0/${data}_result_2.json -b 31 -e False
    # python compute_metric.py -d $data -r ./output_5_0_1/checkpoint-0/${data}_result_2.json 
done

# python compute_metric.py -d sst2 -r ./output_5_0_2/checkpoint-0/sst2_result.json 