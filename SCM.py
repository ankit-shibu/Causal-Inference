#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 23:55:36 2021

@author: ankit-shibu
"""
import numpy as np
from numpy.random import randn
from collections import OrderedDict

def f1(epsilon, **kwargs):
  return epsilon[0]

def f2(epsilon, x, **kwargs):
  return x + epsilon[1]

model1 = OrderedDict ([
  ('x', f1),
  ('y', f2),
])
    
def sample_from_model(model, epsilon = None):
  if epsilon is None:
     epsilon = randn(len(model))
  sample = {}
  for variable, function in model.items():
    sample[variable] = function(epsilon, **sample)
  return sample

def intervene(model, **interventions):
  new_model = model.copy()
  for variable, value in interventions.items():
    new_model[variable] = lambda epsilon, value=value, **kwargs : value
  return new_model

def sample_counterfactuals(model, epsilon=None, **interventions):
  mutilated_model = intervene(model, **interventions)
  if epsilon is None:
     epsilon = randn(len(model))
  factual_sample = sample_from_model(model, epsilon)
  counterfactual_sample = sample_from_model(mutilated_model, epsilon)
  #renaming variables
  counterfactual_sample = dict((key+'*', value) for key, value in counterfactual_sample.items())
  return {**factual_sample, **counterfactual_sample}