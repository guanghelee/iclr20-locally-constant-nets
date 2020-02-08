DATA=PDBbind
BACK_N=1
EN=512
SH=0.1
DEPTH=12

for SEED in 1 2 3 4 5 6 7 8 9 10
do
    echo "python run_elcn_training.py --dataset ${DATA} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type node_dropconnect --p 0.25 --net_type locally_constant --epochs 30 --lr 0.0001 --lr_step_size 300 --gamma 0.1 --batch-size 256 --anneal approx --ensemble_n ${EN} --shrinkage ${SH} --optimizer AMSGrad"
    python run_elcn_training.py --dataset ${DATA} --seed ${SEED} --depth ${DEPTH} --back_n ${BACK_N} --drop_type node_dropconnect --p 0.25 --net_type locally_constant --epochs 30 --lr 0.0001 --lr_step_size 300 --gamma 0.1 --batch-size 256 --anneal approx --ensemble_n ${EN} --shrinkage ${SH} --optimizer AMSGrad
    echo ""
done
