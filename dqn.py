#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 11:05:38 2023

@author: milica
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    model = Sequential([
        Conv2D(conv_units, (3, 3), activation='relu', padding='same', input_shape=input_dims),
        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),
        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),
        Conv2D(conv_units, (3, 3), activation='relu', padding='same'),
        Flatten(),
        Dense(dense_units, activation='relu'),
        Dense(dense_units, activation='relu'),
        Dropout(0.2),
        Dense(n_actions, activation='linear') 
    ])

    model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=1e-7), loss='mse')

    return model