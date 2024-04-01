#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:07:21 2023

@author: milica
"""

import minesweeper
import numpy as np
from tqdm import tqdm 
from dqn_agent import DQNAgent
from dqn_agent import MEM_SIZE_MIN
from minesweeper import MinesweeperEnv 


BOARD_SIZE = 4
NUM_MINES = 3
NUM_LIVES = 2
NUM_TELEPORTERS = 2

EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01

def main():
    game_environment = MinesweeperEnv(board_size=BOARD_SIZE, num_mines=NUM_MINES, num_lives=NUM_LIVES, num_teleporters=NUM_TELEPORTERS)
    game_agent = DQNAgent(env=game_environment)

    episode_rewards = {}
    total_wins, total_games, score = 0, 0, 0

    for episode in tqdm(range(1000)):   
        current_state = game_environment.reset()
        episode_reward = 0
        done = False
        
        while not done:        
            action = game_agent.get_action(current_state)
            new_state, reward, done  = game_environment.step(action)
            
            game_environment.render()
            
            episode_reward += reward

            game_agent.update_replay_memory((current_state, action, reward, new_state, done))

            game_agent.train(done)

            current_state = new_state

            if done: 
                score += np.count_nonzero((current_state != minesweeper.CLOSED) & (current_state != minesweeper.MINE) & (current_state != minesweeper.LIFE) & (current_state != minesweeper.TELEPORTER)) / (BOARD_SIZE * BOARD_SIZE - NUM_MINES - NUM_LIVES - NUM_TELEPORTERS)
                if reward == 1: total_wins += 1       
        total_games += 1
        
        episode_rewards[episode] = episode_reward
        
        game_agent.epsilon = max(EPSILON_MIN, game_agent.epsilon * EPSILON_DECAY)
        
        if episode % 100 == 0:
            win_ratio = round(total_wins / total_games, 3)
            epsilon = round(game_agent.epsilon, 3)
            episode_reward = round(episode_reward, 3)

            print('\n' + f'Episode {episode}: reward: {episode_reward}, win ratio : {win_ratio}, exploration rate: {epsilon}')  
      
              
        if len(game_agent.replay_memory) < MEM_SIZE_MIN:
            continue
        
    sorted_rewards = dict(sorted(episode_rewards.items(), key=lambda item: item[1], reverse=True))
    top_ten_episode_rewards = list(sorted_rewards.items())[:10]
    print('\n' +'Top ten episode rewards:')
    for episode, reward in top_ten_episode_rewards:
        print(f'Episode {episode}: Reward {reward}')
        
    game_environment.close()   
    
    

if __name__ == "__main__":
    main()