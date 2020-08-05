import sys
import os
import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt


def main():
    plot_mean_and_stdev = True
    path_to_dir = os.path.join('measurements')
    averager = 5

    for dir in os.listdir(path_to_dir):
        path_indata = os.path.join(path_to_dir, dir)
        data = {}
        length = {}
        min_length = {}

        for input_file in os.listdir(path_indata):
            # unzip
            if input_file.endswith(".csv"):
                game_name = input_file.split('_')[0]
                data[game_name] = []  #e.g. data[Enduro]
                length[game_name] = []
                min_length[game_name] = 0

        for input_file in os.listdir(path_indata):
            # unzip
            if input_file.endswith(".csv"):
                reward_file = os.path.join(path_indata, input_file.split('.')[0] + '.png')
                averaged_reward_file = os.path.join(path_indata, input_file.split('.')[0] + '_averaged.png')

                rewards = [float(row[0]) for row in csv.reader(open(os.path.join(path_indata, input_file)))]
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
                data[game_name].append(averaged_rewards)
                length[game_name].append(len(averaged_rewards))

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
            for game in data:
                min_length = np.min(length[game])
                for idx, item in enumerate(data[game]):
                    data[game][idx] = item[:min_length]
                    #print(len(data[game][idx]))
                reward_lists = data[game] #2D array of same length reward logs
                x = range(averager, averager*(min_length+1), averager)
                #print(reward_lists)
                stdev = np.round(np.std(reward_lists, axis=0), 2)
                mean = np.mean(reward_lists, axis=0)

                plt.ylabel('Averaged Reward')
                plt.xlabel('Episode')
                #plt.errorbar(x, mean, yerr=stdev, fmt='-o')
                plt.errorbar(x, mean, yerr=stdev, fmt='o', color='black',
                             ecolor='lightgray', elinewidth=3, capsize=0);
                plt.savefig(os.path.join(path_indata, game+'_reward_plot.png'))
                plt.clf()


    return 0


if __name__ == '__main__':
    main()