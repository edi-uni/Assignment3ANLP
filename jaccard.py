#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 13:14:55 2018

@author: s1879797
"""

def jaccard_similarity(x, y):
    intersection_cardinality = len(set(x) & set(y))
    union_cardinality = len(set(x) | set(y))
    return intersection_cardinality/float(union_cardinality)


x = [0, 3, 5]
y = [0, 1, 3, 6, 8]

print (jaccard_similarity(x, y))