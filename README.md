
# DisCoRL: Continual Reinforcement Learning via Policy Distillation

## Branch for Elastic Weight Consolidation method test:
<p>
<img src="imgs/ewc_lam=400_p800k.png " width="300" alt> 
</p>
<p>
<em>Test on EWC method, y-axis is the normalized reward. X-axis consists of the number of episodes of training. This figure is based on a continual learning from Task Target Reaching to Circling task.
EWC penalization is added at the beginning of 3200 episode. We could see that EWC doesn't succeed in improve rewards for both tasks at the same time.
</em>
</p>

The test is available for proximal policy optimization algorithm. Instruction for using this software:

## 1) Train a reinforcement learning agent with PPO2 algorithm:

Enter the root directory:

```
# Train a agent for Circling Task

python -m rl_baselines.train --algo ppo2 --srl-model ground_truth --num-timestep 2000000  --log-dir logs/pretrained_policy/ --env OmnirobotEnv-v0 --circular-continual
```

## 2) Use EWC to transfer and continual learning


```
# The ewc weight: lambda is set to 0.001 here

python -m rl_baselines.train --algo ewc --ewc-weight 0.001 --load-rl-model-path  *path_to_pretrained_model* --log-dir logs/EWC_CC2SC/ --srl-model ground_truth --env OmnirobotEnv-v0 --simple-continual

```


## 3) Evaluate the policy on two environment:

```
python -m rl_baselines.cross_eval --log-dir *path_to_EWC_model* --num-iteration 5

# Example commands:
# python -m rl_baselines.cross_eval --log-dir logs/CC2SC/OmnirobotEnv-v0/srl_combination/ppo2/19-05-03_11h35_10/ --num-iteration 5
```

## 4) Plot the EWC evaluation 

```
# Example command:
# python -m replay.cross_eval_plot -i /*path_to_EWC_model*/eval.pkl
```
