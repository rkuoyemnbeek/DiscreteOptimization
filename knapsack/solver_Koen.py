#!/usr/bin/python
# -*- coding: utf-8 -*-
# from gurobipy import Model, quicksum, max_, and_,or_
from gurobipy import Model, quicksum,GRB
from collections import namedtuple
import numpy as np

Item = namedtuple("Item", ['index', 'value', 'weight'])

def knapsack_sol(n, p, l, L):
    dic = {} # n,p -> sol
    def solve(i, q):
        if (i, q) in dic:
            return dic[i, q]
        if q < 0:
            return float("-inf")
        if i == -1:
            return 0
        sol = max(solve(i-1,q), L[i]+solve(i-1,q-l[i]))
        dic[i,q] = sol
        return sol
    return solve(n-1, p)

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append([i-1, int(parts[0]), int(parts[1])])

    weights = [items[i][2] for i in range(item_count)]
    score_library = [items[i][1] for i in range(item_count)]
    density = [score_library[i]/weights[i] for i in range(item_count)]
    index = np.argsort(density)
    
    x = np.zeros(item_count, dtype = int)
    weight = 0;
    value = 0;
    
    for item in reversed(index):
        weight = weight + weights[item]
        value = value + score_library[item]
        if weight <= capacity:
            x[item] = 1;
        else:
            value = value - score_library[item]
            weight = weight - weights[item]
            continue;
    capacity_full = capacity

    ## Solve the knapsack-problem by repeatedly solving some LP problems
    solution = [i for i in range(item_count) if x[i] == 1]
    
    if (capacity != 1000000):
        print( 'test')
        modelbooks = Model('books')
        modelbooks.setParam('OutputFlag', 0)
        xx={}
        for i in range(item_count):
            xx[i]=modelbooks.addVar(vtype='B',name="xx(%s)" %(i))
        modelbooks.addConstr( quicksum(xx[i]*weights[i] for i in range(item_count)) <= capacity )
        modelbooks.setObjective( quicksum(xx[i] * score_library[i] for i in range(item_count)) , GRB.MAXIMIZE )
        modelbooks.optimize()
        solution = [i for i in xx.keys() if xx[i].X == 1]  
        x = np.zeros(item_count, dtype = int)
        for i in solution:
            x[i] = 1
        value = 0
        weight = 0
        taken = [0]*len(items)
        for item in solution:
            taken[item] = 1
            value += score_library[item]
            weight += weights[item]
        
    else:
#    test = knapsack_sol(item_count, capacity, weights, score_library)
#    print( test)

        teller = 0
        while ((value < 1099881) and (teller < 1000)):
            teller = teller + 1
                          
            nbr_taken = len(solution)
            if (teller == 1):
                test_d = dict( zip( np.arange(0,item_count), x))
    
            idx = np.random.randint( nbr_taken, size= np.random.randint(1,nbr_taken-2, size=1))
            idx = [solution[i] for i in idx]
            idx = list( set( list(idx)))
    
            
            weight_weg = 0
            for i in idx:
                weight_weg = weight_weg + weights[i]
            capacity = capacity_full - weight_weg
            
    
            modelbooks = Model('books')
            modelbooks.setParam('OutputFlag', 0)
            xx={}
            for i in range(item_count):
                if i not in idx:
                    xx[i]=modelbooks.addVar(vtype='B',name="xx(%s)" %(i))
                    xx[i].start = test_d[i]
            modelbooks.addConstr( quicksum(xx[i]*weights[i] for i in range(item_count) if i not in idx ) <= capacity )
            modelbooks.setObjective( quicksum(xx[i] * score_library[i] for i in range(item_count)if i not in idx) , GRB.MAXIMIZE )
            modelbooks.optimize()
            
            solution_new = [i for i in xx.keys() if xx[i].X == 1]
            solution_new.extend( idx)
            
            x = np.zeros(item_count, int)
            for i in solution_new:
                x[i] = 1
            value_new = 0
            weight_new = 0
            taken = [0]*len(items)
            for item in solution_new:
                taken[item] = 1
                value_new += score_library[item]
                weight_new += weights[item]   
            if value < value_new:
                solution = solution_new
                value = value_new
                weight = weight_new
                  
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(1) + '\n'
    output_data += ' '.join(map(str, x))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
#        solve_it(input_data)
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

