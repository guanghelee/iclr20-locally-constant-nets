# Locally Constant Networks

This repository is for the paper

 * "[Oblique Decision Trees from Derivatives of ReLU Networks](https://guanghelee.github.io/pub/Lee_etal_iclr20.pdf)" by [Guang-He Lee](https://people.csail.mit.edu/guanghe/) and [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/) in ICLR 2020.
 * The old title for this paper is "Locally Constant Networks"

<div style="text-align:center"><img src="http://people.csail.mit.edu/guanghe/iclr20_lcn_files/LCN_teaser.jpg" width="85%" height="85%"></div>  

## Package version

* linux (we only tested the codes on Ubuntu)
* python3.6
* pytorch1.1.0
* scikit-learn (for computing AUC)

## Repo structure

* [scripts/](scripts/): scripts for reproducing the experiments
* [data/](data/): datasets used in the paper. Training/validation/testing splits are provided. 
* [log/](log/): the logs of the training results will be stored here.
* [checkpoint/](checkpoint/): the checkpoint of the learned model will be stored here. 

## Main files

* [run_net_training.py](run_net_training.py): main python file for running LCNs/ALCNs/LLNs
* [run_elcn_training.py](run_elcn_training.py): main python file for running ELCN

## Quick start and reproducing experiments

Example scripts are available in [scripts/](scripts/). For example, 

* [scripts/bace_split/](scripts/bace_split/), [scripts/HIV_split/](scripts/HIV_split/), and [scripts/PDBbind/](scripts/PDBbind/) contain the scripts for reproducing the Bace, HIV, and PDBbind experiments with the *tuned* parameters. For example, you may run
	```
    sh scripts/bace_split/run_LCN.sh
    ```
which will run the LCN model with the Bace dataset. 

* [scripts/tox21_split/](scripts/tox21_split/) and [scripts/sider_split/](scripts/sider_split/) contain the scripts for reproducing the Tox21 and Sider experiments with the *parameter tuning procedure*. Since the two datasets are multi-label, you have to specify the task number when you run the script. For example, you may run
	```
    sh scripts/sider_split/run_LCN.sh 1
    ```
which will run the LCN model with the the 1st label in the Sider dataset. The Tox21 contains 12 labels, and the Sider dataset contains 27 labels. 

* The results will be stored in the [log/](log/) directory. The last 3 columns record the training, validation, and testing performance, respectively (from left to right). 

## How to set hyper-parameters

### Switching among LCN, ALCN, and LLN

The default model in [run_net_training.py](run_net_training.py) is LCN. 

* Switching from LCN to ALCN: setting `--anneal` to `approx`.
* Switching from LCN to LLN: setting `--net_type` to `locally_linear`.

### Some suggestions for hyper-parameter tuning

If you would like to apply the codes for other datasets, we suggest to tune the following hyper-parameters. You can see the complete list of hyper-parameters in [arg_utils.py](arg_utils.py). 

* Depth of the network (`--depth`).
* We suggest to use DropConnect (set `--drop_type` to `node_dropconnect`) and mildly tune the dropping probability (e.g., try `--p` in `{0.25, 0.5, 0.75}`).
* You can start trying the model by setting `--back_n` (the depth of the network *g<sub>&phi;</sub>*) to `0`. If it doesn't work, please try to increase it. In our experiments, we found that we need to increase it for regression tasks, and we can simply keep it to `0` for classification tasks. 
* You may want to tune the learning iterations (`--epochs`), learning rate (`--lr`), and optimizer (`--optimizer`) for your tasks. If you change the learning iterations (`--epochs`), you probably should also change the `--lr_step_size` and `--gamma` (see their meaning in the `help` descriptions in [arg_utils.py](arg_utils.py)).
* You may enlarge the `--batch-size` to accelerate training. 

### ELCN

If you would like to try the ensemble version (ELCN), you can just specify the maximum ensemble number as the `--ensemble_n`, and the codes will automatically stores all the logs for each ensemble iteration. For example, if you would like to tune the ensemble number in `{1,2,4,8}`, you can just run `--ensemble_n 8` once, and check the logs for the results with ensemble numbers `{1,2,4,8}`. 

You may also tune the shrinkage parameter. We suggest to try `--shrinkage` in `{1, 0.1}`. 

## Contact

Guang-He Lee (guanghe@csail.mit.edu)

## Citation:

If you find this repo useful for your research, please cite the paper

```
@inproceedings{lee2020oblique,
	title={Oblique Decision Trees from Derivatives of ReLU Networks},
	author={Guang-He Lee and Tommi S. Jaakkola},
	booktitle={International Conference on Learning Representations},
	year={2020},
	url={https://openreview.net/forum?id=Bke8UR4FPB}
}
```
