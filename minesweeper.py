

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:49:24 2023

@author: milica
"""
import numpy as np
import random
from enum import Enum
import gym
from gym import spaces

cell_type_values = [-1, 2, 5, 9]

MINE = cell_type_values[0]
LIFE = cell_type_values[1]
TELEPORTER = cell_type_values[2]
CLOSED = cell_type_values[3]

class RewardType(Enum):
    REWARD_WIN = 1
    REWARD_LOSS = -1
    REWARD_SAFE = 0.1
    REWARD_LIFE = 0.8
    REWARD_TELEPORTER = 0.2
    REWARD_VALID_MOVE = 0.3
    REWARD_INVALID_MOVE = -0.3
   
RewardType = Enum('RewardType', ['REWARD_WIN', 'REWARD_LOSS', 'REWARD_SAFE', 'REWARD_LIFE', 'REWARD_TELEPORTER', 'REWARD_VALID_MOVE', 'REWARD_INVALID_MOVE'])


def create_board(board_size, num_mines, num_lives, num_teleporters):
    board = np.zeros((board_size, board_size), dtype=int)
    mine_positions = random.sample(range(board_size * board_size), num_mines)
    life_positions = random.sample(range(board_size * board_size), num_lives)
    teleporter_positions = random.sample(range(board_size * board_size), num_teleporters)
       
    add_cell_value(board, board_size, 'mine' , mine_positions)
    add_cell_value(board, board_size, 'life' , life_positions)
    add_cell_value(board, board_size, 'teleporter', teleporter_positions)
   
    return board


def add_cell_value(board, board_size, type_value, value_positions):
    for position in value_positions:
        x = position // board_size
        y = position % board_size
        if is_valid_cell(board_size, x, y):
            if type_value == 'mine' and board[x][y] not in cell_type_values:
                board[x][y] = MINE
            elif type_value == 'life' and board[x][y] not in cell_type_values:
                board[x][y] = LIFE
            elif type_value == 'teleporter' and board[x][y] not in cell_type_values:
                board[x][y] = TELEPORTER
            else:
                board[x][y] = 0   
               
               
def is_valid_cell(board_size, x, y):
    if 0 <= x < board_size and 0 <= y < board_size:
        return True
    else:
        return False

   
def stringify(board):
    s = ''
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            s += str(board[x][y]) + '\t'
        s += '\n'
    return s[:-len('\n')]



class MinesweeperEnv(gym.Env):
    
    def __init__(self, board_size, num_mines, num_lives, num_teleporters):
        self.board_size = board_size
        self.num_mines = num_mines
        self.num_lives = num_lives
        self.num_teleporters = num_teleporters
        self.board = create_board(board_size, num_mines, num_lives, num_teleporters)
        self.my_board = np.full((board_size, board_size), CLOSED, dtype=int)
        self.action_space = spaces.Discrete(board_size * board_size)
        self.bs = (self.board_size, self.board_size, 1)
        self.num_actions = 0
        self.lives_count = 0


    def count_neighbour_mines(self, x, y):
        mines_count = 0
        for delta_x in [-1, 0, 1]:
            for delta_y in [-1, 0, 1]:
                if delta_x == 0 and delta_y == 0:
                    continue
                neighobur_x, neighobur_y = x + delta_x, y + delta_y
                if (0 <= neighobur_x < self.board_size
                    and 0 <= neighobur_y < self.board_size
                    and self.board[neighobur_x][neighobur_x] == MINE):
                    mines_count += 1          
        return mines_count


    def place_teleporter(self, x, y):
        x1 = random.randint(0, 3)
        y1 = random.randint(0, 3)
        if (is_valid_cell(self.board_size, x1, y1)):
            manhattan_distance = abs(x1 - x) + abs(y1 - y)
            if manhattan_distance >= 3:
                return x1, y1
        return x, y
   

    def open_cell(self, x, y):
        if self.my_board[x][y] == CLOSED: 
            if self.board[x][y] == LIFE:
                self.lives_count += 1
                self.my_board[x][y] = 0
                return RewardType.REWARD_LIFE.value, False
            elif self.board[x][y] == TELEPORTER:
                new_x, new_y = self.place_teleporter(x, y)
                if self.my_board[new_x][new_y] == MINE:
                    return RewardType.REWARD_LOSS.value, True
                self.my_board[x][y] = 0
                return RewardType.REWARD_TELEPORTER.value, False
            elif self.board[x][y] == MINE and self.lives_count > 0:
                self.my_board[x][y] = 0
                return RewardType.REWARD_SAFE.value, False
            elif self.board[x][y] == MINE and self.lives_count == 0:
                return RewardType.REWARD_LOSS.value, True
            else:
                neighobour_mines = self.count_neighbour_mines(x, y)
                self.my_board[x][y] = neighobour_mines
                self.num_actions += 1
                count_actions = self.board_size ** 2 - (self.num_mines + self.num_lives + self.num_teleporters)
                if self.num_actions == count_actions:
                    return RewardType.REWARD_WIN.value, True
                self.my_board[x][y] = 0
                return RewardType.REWARD_VALID_MOVE.value, False
        else:
            self.my_board[x][y] = 0
            return RewardType.REWARD_INVALID_MOVE.value, False


    def step(self, action):
        x = action // self.board_size
        y = action - x * self.board_size
        reward, done = self.open_cell(x, y)
        return self.my_board.copy(), reward, done


    def reset(self):
        self.board = create_board(self.board_size, self.num_mines, self.num_lives, self.num_teleporters)
        self.my_board = np.full((self.board_size, self.board_size), CLOSED, dtype=int)
        self.bs = (self.board_size, self.board_size, 1)
        self.lives_count = 0
        self.num_actions = 0
        return self.my_board.copy()


    def render(self, mode='human'):
        s = stringify(self.my_board)
  #      print('board: ')
  #      print(s)



