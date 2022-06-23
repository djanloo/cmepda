"""Utilities"""
import numpy as np
import pd4ml


def get_arrival_times(data):
    return data[:, :, ::81]
