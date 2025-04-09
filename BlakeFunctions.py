r"""
Code for working with Dyck vectors and q,t-Catalan numbers

The q,t-Catalan numbers are symmetric polynomials in q and t which are an (area,dinv) refinement of the Catalan numbers.
Also known as a q,t-analog, these polynomials are known to be symmetric in q and t from their definition. However, J. Haglund
and M. Haiman proved that the q,t-Catalan numbers can be realized combinatorially via the area and dinv statistics on Dyck paths.
This lead to a natural question: since the q,t-Catalan numbers are symmetric, what is a bijection on Dyck paths that interchnages
the area and dinv statistics? In other words, is there a purely combinatorial proof of the q,t-symmetry of the q,t-Catalan numbers?
There has been partial progress, but the results are limited.

This code explores the q,t-Catalan numbers through ML methods. It will generate data as well as randomized bijections for ML techniques.
Here are a few examples.

EXAMPLES:

# This outputs all Dyck paths of length 8 = 2*4 along with their area and dinv statistics
vecs = dyckVectors(4)
for vec in vecs:
    DyckWord(area_sequence=vec).pretty_print()
    print("Area: ",dyckArea(vec))
    print("Dinv: ",dyckDinv(vec))
    print("Area Vector: ", vec)
    print('-------------------------')

# This code will output the matrix representaion of the q,t-Catalan number for n=6.
A = qtCatMat(6)
print(A)

# This code outputs the number of possible (area,dinv) interchanging bijections for n = 1,...,10
for i in range(1,11):
    print(f"n = {i}: number of bijections = {numPossibleBijections(i)}")    


AUTHORS:

- Blake Jackson (2025-04-09): initial version

"""

# ****************************************************************************
#       Copyright (C) 2025 Blake Jackson blake.jackson@uconn.edu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#                  https://www.gnu.org/licenses/
# ****************************************************************************

import numpy as np
import math as m
import random
#from sage.all import *

# This function generates the set of all Dyck vectors of length 'n' an integer
def dyckVectors(n):
    '''
    This code will generate a list of all the Dyck vectors of length n. 
    These are in bijective correspondence with the Dyck paths from (0,0) to (n,n). 
    The rules for these Dyck vectors are as follows:
        1. v = (v1, v2, ..., vn) is a vector with nonnegative integer entries
        2. v[1] = 0
        3. v[i+1] <= v[i] + 1 for all i < n.
    '''
    if n < 1:
        print("Invalid input: must input a positive integer.")
        return -1
    if n == 1:
        return [[0]]  # Base case corrected to return a list containing [0]
    else:
        vecs = []
        foo = dyckVectors(n-1)
        for subword in foo:
            for i in range(subword[-1] + 2):  # Fixed the range to allow for one larger than the last entry
                bar = subword + [i]  # Appended the new element
                if i <= subword[-1] + 1:  # Condition 3
                    vecs.append(bar)
    return vecs


# This function calculates the area for a Dyck vector 'd'
def dyckArea(d):
    '''
    This code will calculate the area statistic of an area Dyck vector.
    '''
    return sum(d)

# This function calculates the diagonal inversion number for a Dyck vector 'd'
def dyckDinv(d):
    '''
    This code will calculate the dinv statistic of an area Dyck vector.
    '''
    dinv = 0
    d = list(d)
    for i in range(len(d)):
        for j in range(i+1,len(d),1):
            if d[i] - d[j] == 0 or d[i] - d[j] == 1:
                dinv += 1
    return dinv

# This function calculates the deficit of a Dyck vector 'd'
def dyckDefc(d):
    '''
    This code will calculate the deficit statistic of a Dyck vector.
    This is used when building the q,t-Catalan matrix for the
    q,t-Catalan number. It also appears in the results of Lee, Li, and Loehr.
    '''
    n = int(len(d))
    return (fact(n))/(2*fact(n-2)) - dyckArea(d) - dyckDinv(d)

# This function calculates the q,t-Catalan number for 'n' an integer
def qtCat(n):
    '''
    BROKEN RIGHT NOW
    This code will generate a polynomial in two variables
    (q and t) known as the q,t-Catalan number.
    '''
    R = PolynomialRing(QQ,'q,t')
    q = R.gen(0)
    t = R.gen(1)
    poly = 0
    vecs = dyckVectors(n)
    for vec in vecs:
        n = dyckArea(vec)
        m = dyckDinv(vec)
        poly = poly + q^n*t^m  # q^area t^dinv
    return poly

# This function calculates the q,t-Catalan matrix for 'n' an integer
def qtCatMat(n):
    '''
    This code will generate a matrix that encodes the data
    of the q,t-Catalan number. The output is an integer matrix
    where the number in the i,j entry is the coefficient on the
    q^i t^j term of the q,t-Catalan number.
    '''
    if n < 1:
        print("n must be an integer >= 1.")
        return -99
    #need the (n-1) st triangular number plus 1 since the area can range from 0-Tn
    Tn = int((n-1)*(n)/2+1)
    mat = np.zeros((Tn,Tn))
    vecs = dyckVectors(n)
    for vec in vecs:
        i = int(dyckArea(vec))
        j = int(dyckDinv(vec))
        # we want the entry i rows up and j columns over from the bottom-left of the matrix to represent q^i t^j
        mat[Tn-i-1, j] += 1
    return mat

# This function calculates the total possible bijections that interchange area and dinv
def numPossibleBijections(n):
    '''
    This code will calculate the total possible bijections that swap
    area and dinv for a fixed n.
    '''
    mat = qtCatMat(n)
    k = mat.shape[0]-1
    # This creates an array the same size as qtCatMat with True/False entries where it is true for entries
    # on or above the antidiagonal of the qt-Catalan matrix
    mask = np.fromfunction(lambda i, j: i + j <= k, mat.shape, dtype=int)
    # These are the elements we want to take a product over
    elements = mat[mask]
    product = 1
    for i, c in enumerate(elements):
        product *= m.factorial(int(c))
    return product

# This function calculates the percent of Dyck paths which are not determined uniquely by their Area,Dinv vector
def percentNondeterministic(n):
    '''
    This code will generate a matrix which encodes the data of the q,t-Catalan number and a similar one that replaces
    the 1's in the matrix with 0's to see how many Dyck paths are not determined by their (Area, Dinv) vector. 
    The output is the percentage of non-deterministic Dyck paths of length n.
    '''
    mat1 = qtCatMat(n)
    mat2 = mat1.copy() # Make a copy of the qt Catalan matrix
    for i,j in np.ndindex(mat2.shape):
        if mat2[i,j] == 1:
            mat2[i,j] = 0 # Eliminate any Dyck paths which are determined by their Area, Dinv vector
    # Compute sums
    sum_mat1 = np.sum(mat1)
    sum_mat2 = np.sum(mat2)
    # Avoid division by zero
    if sum_mat1 == 0:
        return 0.0 
    # Ensure floating-point division
    return sum_mat2 / sum_mat1

# You need to pass this function a list of Dyck vectors 'vecs'
def dyckListToDict(vecs):
    '''
    This code will generate a list of dictionaries for all of the Dyck vectors of a certain size.
    The elements of the list are dictionaries with keys:
    'id': an integer >= 1
    'dyckVec': the area vector corresponding to a Dyck path
    'area': the area statistic of the Dyck vector
    'dinv': the dinv statistic of the Dyck vector
    '''
    elements = []
    i = 1
    for vec in vecs:
        elements.append({'id':i, 'dyckVec':vec, 'area':dyckArea(vec), 'dinv':dyckDinv(vec)})
        i += 1
    return elements

# This function returns a random bijection of Dyck paths
def randomBijection(n):
    '''
    This code generates a random bijection of the form (dyckPath1, dyckPath2) for all Dyck paths of size 'n'
    with the requirement that dyckArea(dyckPath1) = dyckDinv(dyckPath2) and dyckArea(dyckPath2) = dyckDinv(dyckPath1).
    This makes the dyckPairing(n) function obsolete.

    Parameters:
    n (integer): The size of the square that the Dyck path lives in.

    Returns:
    list of tuples of length Cat(n): A list of paired vectors (vect1, vect2) which are area-Dyck paths.
    '''
    vecs = dyckListToDict(dyckVectors(n))
    # Extract vectors along with their statistics
    vectors = [(entry['dyckVec'], entry['area'], entry['dinv']) for entry in vecs]
    # Shuffle vectors randomly
    # Make a copy
    vectors_copy = vectors.copy()
    # Shuffle the copy
    random.shuffle(vectors_copy)
    # Create a bijection by pairing elements
    bijection = []
    used_i = set()
    used_j = set()
    for i, (v1, area1, dinv1) in enumerate(vectors):
        if i in used_i:
            continue
        for j, (v2, area2, dinv2) in enumerate(vectors_copy):
            if j in used_j:
                continue
            if area1 == dinv2 and area2 == dinv1:
                # Add the pair only if it's not already in the list
                pair1 = [v1, v2]
                pair2 = [v2, v1]
                if pair1 not in bijection:
                    bijection.append(pair1)
                if v1 != v2 and pair2 not in bijection:
                    bijection.append(pair2)
                used_i.add(i)
                used_j.add(j)
                break  # Move to the next available vector
    return bijection

#####################################################################################################
######################  Code Archive, the functions are outdated, use at your own risk ############## 
#####################################################################################################

def dyckPairing(n):
    '''
    This code will generate a list of paired elements for all of the Dyck vectors of a certain size 'n'.
    Pairing occurs based on area-to-dinv matching. There is no randomization, matching occurs based on first-come-first-serve basis.
    Output is currently a list of pairs of Dyck paths given by their area-sequence representation.
    '''
    vecs = dyckVectors(n)
    elements = dyckListToDict(vecs)
    # Step 1: Create a lookup dictionary
    area_dict = {}
    dinv_dict = {}
    for element in elements:
        area = element['area']
        dinv = element['dinv']
        if area not in area_dict:
            area_dict[area] = []
        if dinv not in dinv_dict:
            dinv_dict[dinv] = []
        area_dict[area].append(element)
        dinv_dict[dinv].append(element)
    # Step 2: Find pairs
    paired_elements = []
    unpaired_ids = set(element['id'] for element in elements)
    for element in elements:
        area = element['area']
        dinv = element['dinv']
        # Check for self-pairing
        if area == dinv and element['id'] in unpaired_ids:
            paired_elements.append((element, element))
            unpaired_ids.remove(element['id'])
            continue
        if dinv in area_dict and area in dinv_dict:
            # Find matching pairs
            possible_pairs = [e for e in dinv_dict[area] if e['area'] == dinv and e['id'] in unpaired_ids]
            for pair in possible_pairs:
                if element['id'] in unpaired_ids and pair['id'] in unpaired_ids:
                    paired_elements.append((element, pair))
                    unpaired_ids.remove(element['id'])
                    unpaired_ids.remove(pair['id'])
                    break
    return paired_elements




    
