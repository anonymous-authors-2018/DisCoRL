from stable_baselines.common.policies import FeedForwardPolicy


class ProgressiveFeedForwardPolicy(FeedForwardPolicy):
    """
    A progressive neural network implementation for feedForward Policy

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """
    def __init__(self,sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(ProgressiveFeedForwardPolicy,self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        feature_extraction="mlp", **_kwargs)

        
