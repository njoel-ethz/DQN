import cv2
import gym
import atari_py
import time
import numpy as np
from tqdm import tqdm
import os
from scipy.ndimage.filters import gaussian_filter
from PIL import Image, ImageDraw, ImageFont
import random

import tensorflow as tf
from stable_baselines.deepq.policies import MlpPolicy, LnMlpPolicy, CnnPolicy, LnCnnPolicy
#from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import DQN
import torch
from sal_model import TASED_v2

TIMESTEPS = 500000
TRAIN_MODEL = True
USE_SALIENCY = True
#if True, change Tensor Shape in \stable_baselines\common\base_class.py
#LOAD_MODEL = False

POLICY = CnnPolicy
GAME = 'Enduro-v0'
SALIENCY_WEIGHTS = 'enduro_weights_1000.pt'
EVERY_N_ITERATIONS = 2

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if USE_SALIENCY:
        zipname = GAME.split('-')[0] + '_' + str(EVERY_N_ITERATIONS) + "_sal_model"
    else:
        zipname = GAME.split('-')[0] + "_model"

    #test_if_gpu()
    #print(atari_py.list_games())

    sal_model = build_sal_model(SALIENCY_WEIGHTS, USE_SALIENCY)
    env = gym.make(GAME)
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])

    if TRAIN_MODEL:
        if not os.path.isdir('trained_models'):
            os.makedirs('trained_models')

        seed = random.randint(0, 99999)
        model = DQN(POLICY, env, verbose=1, learning_rate=1e-5, seed=seed) # ,buffer_size=50000, exploration_fraction=0.1, exploration_final_eps=0.02)

        print('--- TRAINING PHASE ---')
        print(GAME.split('-')[0].upper())

        model.learn(total_timesteps=TIMESTEPS, use_saliency=USE_SALIENCY, sal_model=sal_model, n=EVERY_N_ITERATIONS, zipname=zipname) #, callback=callback)

        print("Saving model to " + zipname + '_final.zip')
        model.save(os.path.join('trained_models', zipname + '_final.zip'))
    else:
        model = DQN.load(os.path.join('trained_models', zipname + '_final.zip'), env)
        print('RL model loaded: ' + zipname + '_final.zip')

    snippet = []
    len_temporal = 32
    obs_for_video, smap_for_video = [], []

    obs = env.reset()
    if USE_SALIENCY:
        snippet, obs = initial_sal(obs, snippet, len_temporal, sal_model)

    print('--- TEST PHASE ---')
    for i in tqdm(range(5000)):

        """ Save observation and saliency map for qualitative analysis """
        if i < 300 and USE_SALIENCY:
            obs_for_video.append(obs[:, :, :3])
            smap_for_video.append(obs[:, :, 3:])
        else:
            time.sleep(0.01)
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

        """ SALIENCY PREDICTION ON OBSERVATION """
        if USE_SALIENCY:
            if i%EVERY_N_ITERATIONS==0:
                snippet, smap = produce_saliency_maps(snippet, obs, len_temporal, sal_model)
            else:
                img = cv2.resize(obs, (384, 224))
                img = img[..., ::-1]
                snippet.append(img)
                del snippet[0]
            o1, o2, o3 = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
            s1, s2, s3 = smap[:, :, 0], smap[:, :, 1], smap[:, :, 2]
            obs = np.dstack((o1, o2, o3, s1, s2, s3))

    produce_video(obs_for_video, smap_for_video , GAME, EVERY_N_ITERATIONS)
    os.system('python run_reward_plot.py')

    print('--- DONE ---')

def initial_sal(obs, snippet, len_temporal, sal_model):
    snippet, smap = produce_saliency_maps(snippet, obs, len_temporal, sal_model)
    o1, o2, o3 = obs[:, :, 0], obs[:, :, 1], obs[:, :, 2]
    s1, s2, s3 = smap[:, :, 0], smap[:, :, 1], smap[:, :, 2]
    obs = np.dstack((o1, o2, o3, s1, s2, s3))
    return snippet, obs

def produce_video(obs_list, smap_list, game, n):
    if len(obs_list)==0:
        print('No data for video available')
        return

    dir = 'qualitative_videos'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    name = game.split('-')[0] + '_' + str(n)

    height, width, _ = obs_list[0].shape
    video_array = []
    for i in range(len(obs_list)):
        temp_frame = Image.fromarray(obs_list[i])
        temp_map = cv2.cvtColor(smap_list[i], cv2.COLOR_RGB2GRAY)
        temp_map = Image.fromarray(temp_map)
        red_img = Image.new('RGB', (160, 210), (0, 0, 255))
        temp_frame.paste(red_img, mask=temp_map)
        temp_frame = np.array(temp_frame)
        video_array.append(temp_frame)

    video = cv2.VideoWriter(os.path.join(dir, name + '.mp4'), cv2.VideoWriter_fourcc(*'DIVX'), 24, (width, height))
    print('Writing video to ' + os.path.join(dir, name + '.mp4'))
    for i in range(len(video_array)):
        video.write(video_array[i])
    video.release()

def produce_saliency_maps(snippet, obs, len_temporal, sal_model):
    img = cv2.resize(obs, (384, 224))
    img = img[..., ::-1]
    snippet.append(img)
    count = 0
    # not enough previous frames to produce correct saliency prediction on first 31 frames
    while len(snippet) < len_temporal:
        count += 1
        snippet.append(img)

    clip = transform(snippet)
    smap = process(sal_model, clip)
    # shape of obs & smap each: (210, 160, 3) np array
    smap = cv2.cvtColor(smap, cv2.COLOR_GRAY2RGB)

    del snippet[0]

    if count>0:
        print('imputed ' + str(count) + ' frames')

    return snippet, smap


def transform(snippet):
    ''' stack & noralization '''
    snippet = np.concatenate(snippet, axis=-1)
    snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
    snippet = snippet.mul_(2.).sub_(255).div(255)

    return snippet.view(1,-1,3,snippet.size(1),snippet.size(2)).permute(0,2,1,3,4)

def process(model, clip):
    ''' process one clip and save the predicted saliency map '''
    with torch.no_grad():
        smap = model(clip.cuda()).cpu().data[0]

    smap = (smap.numpy()*255.).astype(np.int)/255.
    smap = gaussian_filter(smap, sigma=7)
    smap = cv2.resize(smap, (160, 210))

    return (smap/np.max(smap)*255.).astype(np.uint8)

def build_sal_model(file_weight, sal_flag):
    if not sal_flag:
        return None
    model = TASED_v2()
    file_weight = os.path.join('saliency_weights', file_weight)

    # load the weight file and copy the parameters
    if os.path.isfile(file_weight):
        print('loading weight file')
        weight_dict = torch.load(file_weight)
        model_dict = model.state_dict()
        for name, param in weight_dict.items():
            if 'module' in name:
                name = '.'.join(name.split('.')[1:])
            if name in model_dict:
                if param.size() == model_dict[name].size():
                    model_dict[name].copy_(param)
                else:
                    print(' size? ' + name, param.size(), model_dict[name].size())
            else:
                print(' name? ' + name)

        print(' loaded')
    else:
        print('weight file?')

    model = model.cuda()
    torch.backends.cudnn.benchmark = False
    model.eval()

    return model

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
    is_solved = lcl['self'].num_timesteps > 100 and mean_100ep_reward >= 29900000 #199
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