# DisCoRL: Continual Reinforcement Learning via Policy Distillation

Our environment `Omnirobot` includes three different tasks: target reaching, target circling and target escaping. 
One shall use the branch [escaping_task](https://github.com/anonymous-authors-2018/CoRL/tree/escaping_task) for the escaping task.
For the baseline EWC method, refer to the branch [ewc](https://github.com/anonymous-authors-2018/CoRL/tree/ewc).
For the baseline Progress & Compress, please refer to the branch [ProgressCompress](https://github.com/anonymous-authors-2018/CoRL/tree/ProgressCompress).


| **Omnidirectional robot environment**       |
| ------------------------------------------- |
| <img src="imgs/three_tasks_reallife_sim_update.png " width="600">  |

#  A user guide for Policy Distillation

These are steps to reproduce our experiments and results. We recommend launching a `visdom` server to monitor the training process of Reinforcement learning by running: 
```
python -m visdom.server
```
The guidelines below are for distilling three tasks. A simplified two tasks distillation instruction is in folder [rl_baselines/supervised_rl/](https://github.com/anonymous-authors-2018/CoRL/tree/escaping_task/rl_baselines/supervised_rl)



# Step 0 - Generate dataset for SRL (random policy)

```
cd robotics-rl-srl
# Dataset 1 (Target Reaching task)
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_random_simple --env OmnirobotEnv-v0 --simple-continual --num-episode 250 -f
# Dataset 2 (Target Circling task)
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_circular --env OmnirobotEnv-v0 --circular-continual --num-episode 250 -f
# Dataset 3 (Target Escaping task)
python -m environments.dataset_generator --num-cpu 6 --name Omnibot_escape --env OmnirobotEnv-v0 -esc --num-episode 250 -f
```

# Step 1 - Train Baselines


### 1.1) Train SRL

```
cd srl_zoo
# Dataset 1 (Target Reaching task)
python train.py --data-folder data/Omnibot_random_simple  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
# Dataset 2 (Target Circling task)
python train.py --data-folder data/Omnibot_circular  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
# Dataset 3 (Target Escaping task)
python train.py --data-folder data/Omnibot_escape  -bs 32 --epochs 20 --state-dim 200 --training-set-size 20000 --losses autoencoder inverse
```


### 1.2) Train policy

Train

```
cd ..

# save config file
cp config/srl_models.yaml config/srl_models_temp.yaml

# Dataset 1 (Target Reaching task)
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --srl-config-file config/srl_models_simple.yaml --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/simple/  --num-cpu 8 --simple-continual  --latest

# Dataset 2 (Target Circling task)
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --srl-config-file config/srl_models_circular.yaml --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/circular/  --num-cpu 6 --circular-continual  --latest

# Dataset 3 (Target Escaping task)
python -m rl_baselines.train --algo ppo2 --srl-model srl_combination --srl-config-file config/srl_models_escape.yaml --num-timesteps 1000000 --env OmnirobotEnv-v0 --log-dir logs/escape/  --num-cpu 6 -esc  --latest

# restore config file
cp config/srl_models_temp.yaml config/srl_models.yaml
```

Visualizing the trained policy and plotting the RL learning curve

```
python -m replay.enjoy_baselines --log-dir *file* --num-timesteps 10000 --render --action-proba
example : python -m replay.enjoy_baselines --log-dir logs/simple/OmnirobotEnv-v0/srl_combination/ppo2/19-04-25_10h19_42/ --num-timesteps 10000 --render --action-proba



# plot results
python -m replay.plots --log-dir /logs/simple/OmnirobotEnv-v0/srl_combination/ppo/ --latest

python -m replay.plots --log-dir /logs/circular/OmnirobotEnv-v0/srl_combination/ppo/ --latest

python -m replay.plots --log-dir /logs/escape/OmnirobotEnv-v0/srl_combination/ppo/ --latest
```

# Step 2 - Distillation Training procedure


### 2.1) Generating a dataset on-policy

```
# Dataset 1 (Target Reaching task)
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --short-episodes --save-path data/ --name reaching_on_policy -sc

# Dataset 2 (Target Circling task)
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --short-episodes --save-path data/ --name circular_on_policy -cc

# Dataset 3 (Target Escaping task)
python -m environments.dataset_generator --env OmnirobotEnv-v0 --num-episode 100 --run-policy custom --log-custom-policy logs/*path2policy* --short-episodes --save-path data/ --name escape_on_policy -esc

# Merging Datasets

(/ ! \ it removes the generated dataset for dataset 1 and 2)
# Merge the dataset of dataset 1(TR) and dataset 2(TC)
python -m environments.dataset_merger --merge data/circular_on_policy/ data/reaching_on_policy/ data/merge_CC_SC
# Merge the dataset of merged_dataset (TR&TC) and dataset 2(TE)
python -m environments.dataset_merger --merge data/merge_CC_SC data/escape_on_policy/ data/merge_CC_SC_ESC


# Copy the merged Dataset to srl_zoo repository
cp -r data/merge_CC_SC_ESC srl_zoo/data/merge_CC_SC_ESC

```


### 2.2) Running Policy Distillation

```
# make a new log folder
mkdir logs/CL_SC_CC

# Policy distillation on merged dataset
python -m rl_baselines.train --algo distillation --srl-model raw_pixels --env OmnirobotEnv-v0 --log-dir logs/CL_SC_CC --teacher-data-folder srl_zoo/data/merge_CC_SC_ESC  --distillation-training-set-size 40000 --epochs-distillation 5
```

# Step 3 -  Evaluating the trained policy

```
python -m replay.enjoy_baselines --log-dir logs/CL_SC_CC/*path_to_policy_model* --num-timesteps 10000 --render --action-proba --*task-name* [--simple-continual, --circular-continual, --escape-continual]
```
