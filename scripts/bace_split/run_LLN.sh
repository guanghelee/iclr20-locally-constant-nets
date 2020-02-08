DATA=bace_split/
BACK_N=0
P=0.25
DROP=node_dropconnect
DEPTH=5
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    echo "python run_net_training.py --dataset ${DATA} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type ${DROP} --p ${P} --net_type locally_linear --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64"
    python run_net_training.py --dataset ${DATA} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type ${DROP} --p ${P} --net_type locally_linear --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64
    echo ""
done
