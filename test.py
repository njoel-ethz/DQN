import gym
import atari_py
import numpy as np

from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN

from tqdm import tqdm
import os
import tensorflow as tf


POLICY = CnnPolicy
GAME = 'Pong-v0'
TIMESTEPS = 100000

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    zipname = GAME.split('-')[0] + "_model.zip"

    #test_if_gpu()

    #print(atari_py.list_games())
    env = gym.make(GAME)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    #env = DummyVecEnv([lambda: env])


    model = DQN(POLICY, env, verbose=1) #, learning_rate=1e-3, buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02)

    #model = DQN.load(zipname, env)

    print('--- TRAINING PHASE ---')
    print(GAME.split('-')[0].upper())
    model.learn(total_timesteps=TIMESTEPS, callback=callback) #callback=callback

    print("Saving model to " + zipname)
    model.save(zipname)

    obs = env.reset()

    print('--- TEST PHASE ---')
    for i in tqdm(range(3000)):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

    print('--- DONE ---')


def callback(lcl, _glb):
    """
    The callback function for logging and saving

    :param lcl: (dict) the local variables
    :param _glb: (dict) the global variables
    :return: (bool) is solved
    """
    # stop training if reward exceeds 199
    if len(lcl['episode_rewards'][-101:-1]) == 0:
        mean_100ep_reward = -np.inf
    else:
        mean_100ep_reward = round(float(np.mean(lcl['episode_rewards'][-101:-1])), 1)
    is_solved = lcl['self'].num_timesteps > 100 and mean_100ep_reward >= 199
    return not is_solved

def test_if_gpu():
    with tf.device('/gpu:0'):
        a = tf.constant(3.0)
    i = 0
    with tf.Session() as sess:
        while i < 100:
            print(sess.run(a))
            i += 1
        return

if __name__ == '__main__':
    main()