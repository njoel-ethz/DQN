import sys
import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt


def main():
    plot_mean_and_stdev = True
    plot_methods_together = True
    path_to_dir = os.path.join('measurements')
    averager = 5
    full_data = {}

    for dir in os.listdir(path_to_dir):
        path_indata = os.path.join(path_to_dir, dir)
        data = {}
        length = {}
        min_length = {} #joint min length over all 5 runs

        for input_file in os.listdir(path_indata):
            if input_file.endswith(".csv"):
                game_name = input_file.split('_')[0]
                method_name = " ".join(input_file.split('_')[1:-1])
                data[game_name] = []  #e.g. data[Enduro]
                length[game_name] = []
                min_length[game_name] = 0
                if not game_name in full_data:
                    full_data[game_name] = {}
                if not method_name in full_data[game_name]:
                    full_data[game_name][method_name] = []  #e.g. full_data[Enduro][5 sal model reward]

        for input_file in os.listdir(path_indata):
            # unzip
            if input_file.endswith(".csv"):
                reward_file = os.path.join(path_indata, input_file.split('.')[0] + '.png')
                averaged_reward_file = os.path.join(path_indata, input_file.split('.')[0] + '_averaged.png')

                rewards = [row[0] for row in csv.reader(open(os.path.join(path_indata, input_file)))]

                if 'SEED' in rewards[0]:
                    seed = rewards[0].split('_')[1]
                    rewards = rewards[1:]
                    #print('found seed: ' + seed)


                rewards = [float(item) for item in rewards]
                rewards = rewards[:-1] #discard last unfinished episode

                averaged_rewards = []
                temp_sum = 0
                index_stats = []
                for i in range(1,len(rewards)+1):
                    temp_sum += rewards[i-1]
                    if i%averager==0:
                        index_stats.append(i)
                        averaged_rewards.append(temp_sum/averager)
                        temp_sum = 0

                game_name = input_file.split('_')[0]
                method_name = " ".join(input_file.split('_')[1:-1])
                data[game_name].append(averaged_rewards)
                length[game_name].append(len(averaged_rewards))
                full_data[game_name][method_name].append(averaged_rewards)

                plt.ylabel('Reward')
                plt.xlabel('Episode')
                plt.plot(rewards)
                plt.savefig(reward_file)
                plt.clf()

                plt.ylabel('Averaged Reward')
                plt.xlabel('Episode')
                plt.plot(index_stats, averaged_rewards)
                plt.savefig(averaged_reward_file)
                plt.clf()

        if plot_mean_and_stdev:
            for method in data:
                min_length = np.min(length[method])
                for idx, item in enumerate(data[method]):
                    data[method][idx] = item[:min_length]
                    #print(len(data[game][idx]))
                reward_lists = data[method] #2D array of same length reward logs
                x = range(averager, averager*(min_length+1), averager)
                #print(reward_lists)
                stdev = np.round(np.std(reward_lists, axis=0), 2)
                mean = np.mean(reward_lists, axis=0)

                plt.ylabel('Averaged Reward')
                plt.xlabel('Episode')
                #plt.errorbar(x, mean, yerr=stdev, fmt='-o')
                plt.errorbar(x, mean, yerr=stdev, fmt='o', color='black',
                             ecolor='lightgray', elinewidth=3, capsize=0);
                plt.savefig(os.path.join(path_indata, method+'_reward_plot.png'))
                plt.clf()

    if plot_methods_together:
        """for key in full_data:
            print(key)
            for key2 in full_data[key]:
                print(key2)"""
        for game in full_data:
            averaged_per_method = {}
            stdev_per_method = {}
            x_per_method = {}
            for method in full_data[game]:
                print(game + ": " + method)
                reward_lists = full_data[game][method] #2D array of same length reward logs
                min_length = np.inf
                for i in range(len(reward_lists)):
                    min_length = min(min_length, len(reward_lists[i]))
                for i in range(len(reward_lists)):
                    temp = reward_lists[i][:min_length]
                    reward_lists[i] = temp
                x_per_method[method] = range(averager, averager*(min_length+1), averager)
                stdev = np.round(np.std(reward_lists, axis=0), 2)
                stdev_per_method[method] = stdev
                averaged_per_method[method] = np.mean(reward_lists, axis=0)

            colours = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
            methods = ['model reward', '5 sal overlaid model reward', '5 sal model reward', '2 sal model reward']
            method_names = ['Original DQN', 'Non-salient regions blurred', 'Saliency on every 5th frame', 'Saliency on every 2nd frame']

            _, ax = plt.subplots(1)
            for i, method in enumerate(methods):
                mean_temp = averaged_per_method[method]
                stdev_temp = stdev_per_method[method]
                ax.plot(x_per_method[method], mean_temp, lw=2, label=method_names[i], color=colours[i])
                ax.fill_between(x_per_method[method], mean_temp + stdev_temp, mean_temp - stdev_temp, facecolor=colours[i], alpha=0.3)
            ax.set_title(game + r' reward: Empirical $\mu$ and $\pm \sigma$ interval')
            ax.legend(loc='upper left')
            ax.set_xlabel('episode')
            ax.set_ylabel('reward')
            ax.grid()
            plt.savefig(os.path.join(path_to_dir, game + '_reward.png'))
            plt.clf()


    return 0


if __name__ == '__main__':
    main()