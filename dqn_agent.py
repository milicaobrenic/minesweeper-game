#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:06:58 2023

@author: milica
"""

import random
import numpy as np
from collections import deque
from dqn import create_dqn
import minesweeper
from minesweeper import MinesweeperEnv

#memory
MEM_SIZE = 50000
MEM_SIZE_MIN = 1000 

#learning
BATCH_SIZE = 64
LEARN_RATE = 0.01
LEARN_DECAY = 0.999
LEARN_MIN = 0.001
DISCOUNT = 0.1 

#exploration
EPSILON = 0.95
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01


class DQNAgent(object):
    
    
    def __init__(self, env, conv_units=64, dense_units=256): 
        self.env = env
        self.discount = DISCOUNT
        self.learn_rate = LEARN_RATE
        self.epsilon = EPSILON
        self.model = create_dqn(self.learn_rate, self.env.bs, self.env.board_size ** 2, conv_units, dense_units)

        self.target_model = create_dqn(self.learn_rate, self.env.bs, self.env.board_size ** 2, conv_units, dense_units)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=MEM_SIZE)
        self.target_update_counter = 0

    def get_action(self, state):
        board = state.reshape(1, self.env.board_size ** 2)

        rand = np.random.random() 

        if rand < self.epsilon:
            action = 0
            while state[int(action / self.env.board_size)][action % self.env.board_size] != minesweeper.CLOSED:
                action = self.env.action_space.sample()
            move = action
        else:
            moves = self.model.predict(np.reshape(state, (1, self.env.board_size, self.env.board_size, 1)))
            moves[board != minesweeper.CLOSED] = np.min(moves) 
            move = np.argmax(moves)

        return move

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def train(self, done):
 
        if len(self.replay_memory) < MEM_SIZE_MIN:
            return

        batch = random.sample(self.replay_memory, BATCH_SIZE)

        current_states = np.array([transition[0] for transition in batch])
        current_states = current_states.reshape(-1, 4, 4, 1)
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in batch])
        new_current_states = new_current_states.reshape(-1, 4, 4, 1)
        future_qs_list = self.target_model.predict(new_current_states)

        x,y = [], []

        for i, (current_state, action, reward, new_current_state, done) in enumerate(batch):
            if not done:
                max_future_q = np.max(future_qs_list[i])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[i]
            current_qs[action] = new_q

            x.append(current_state)
            y.append(current_qs)
            
        x = np.array(x)
        y = np.array(y)

        x = x.reshape(-1, 4, 4, 1)    
                 
        self.model.fit(x, y, batch_size=BATCH_SIZE, shuffle=False, verbose=0)

        if done:
            self.target_update_counter += 1

        if self.target_update_counter > 5:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

        self.learn_rate = max(LEARN_MIN, self.learn_rate*LEARN_DECAY)

        self.epsilon = max(EPSILON_MIN, self.epsilon*EPSILON_DECAY)



if __name__ == "__main__":
    DQNAgent(MinesweeperEnv(board_size=4, num_mines=3, num_lives=2, num_teleporters=2))