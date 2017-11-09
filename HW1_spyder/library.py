# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 13:28:05 2017

@author: giaco
"""
import math

def franke(x1, x2):
  return (
    .75 * math.exp(-(9 * x1 - 2) * 2 / 4.0 - (9 * x2 - 2) * 2 / 4.0) +
    .75 * math.exp(-(9 * x1 + 1) ** 2 / 49.0 - (9 * x2 + 1) / 10.0) +
    .5 * math.exp(-(9 * x1 - 7) * 2 / 4.0 - (9 * x2 - 3) * 2 / 4.0) -
    .2 * math.exp(-(9 * x1 - 4) * 2 - (9 * x2 - 7) * 2)
  )