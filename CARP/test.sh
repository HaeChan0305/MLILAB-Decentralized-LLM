# for data in agnews mr r8 sst2
for data in sst2
do
    echo $data
    python test.py -d $data -o ./output_5_0_2/checkpoint-0/${data}_result.json -b 31 -e True
    # python compute_metric.py -d $data -r ./output_5_0_2/checkpoint-0/${data}_result.json 
done

# python compute_metric.py -d sst2 -r ./output_5_0_2/checkpoint-0/sst2_result.json 