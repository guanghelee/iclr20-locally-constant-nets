DATA=sider_split/
TASK=$1
BACK_N=0
DROP=node_dropconnect
for SEED in 1 2 3 4 5 6 7 8 9 10
do
    for DEPTH in 12 11 10 9 8 7 6 5 4 3 2
    do
        for P in 0.25 0.5 0.75
        do
                echo "python run_net_training.py --dataset ${DATA} --sub-task ${TASK} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type ${DROP} --p ${P} --net_type locally_constant --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64"
                python run_net_training.py --dataset ${DATA} --sub-task ${TASK} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type ${DROP} --p ${P} --net_type locally_constant --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64
                echo ""
        done
        echo "python run_net_training.py --dataset ${DATA} --sub-task ${TASK} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type none --p 0 --net_type locally_constant --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64"
        python run_net_training.py --dataset ${DATA} --sub-task ${TASK} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type none --p 0 --net_type locally_constant --epochs 30 --lr 0.1 --lr_step_size 10 --gamma 0.1 --batch-size 64
        echo ""
    done
done


