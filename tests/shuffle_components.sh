home=$HOME
fully=$home/dataspace/graph/fully-synthetic/


for n in 1000 2000 5000 10000 20000
do
    for k in 20 60 100 200 350
    do
        DIR=$fully/small-world-n$n-k$k-p5-seed123/random-d01
        python utils/shuffle_graph.py --input_dir ${DIR} --out_dir $DIR--1
        rm -r $DIR
        mv $DIR--1 $DIR
    done
done
