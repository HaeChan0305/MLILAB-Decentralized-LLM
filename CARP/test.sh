data=mr
STEP=141
for i in {1..10}
do
    echo ${data}
    echo "checkpoint-$((STEP * i))"
    
    mkdir ./output_5_1_1/${data}/checkpoint-$((STEP * i))
    python test.py -d $data -c ./ft_model_5_1_1/${data}/checkpoint-$((STEP * i)) -o ./output_5_1_1/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
done

data=mr
STEP=141
for i in {1..10}
do
    echo ${data}
    echo "checkpoint-$((STEP * i))"
    
    mkdir ./output_5_1_2/${data}/checkpoint-$((STEP * i))
    python test.py -d $data -c ./ft_model_5_1_2/${data}/checkpoint-$((STEP * i)) -o ./output_5_1_2/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
done

data=r8
STEP=85
for i in {1..10}
do
    echo ${data}
    echo "checkpoint-$((STEP * i))"
    
    mkdir ./output_5_1_1/${data}/checkpoint-$((STEP * i))
    python test.py -d $data -c ./ft_model_5_1_1/${data}/checkpoint-$((STEP * i)) -o ./output_5_1_1/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
done


data=r8
STEP=85
for i in {1..10}
do
    echo ${data}
    echo "checkpoint-$((STEP * i))"
    
    mkdir ./output_5_1_2/${data}/checkpoint-$((STEP * i))
    python test.py -d $data -c ./ft_model_5_1_2/${data}/checkpoint-$((STEP * i)) -o ./output_5_1_2/${data}/checkpoint-$((STEP * i))/${data}_result.json -b 17
done