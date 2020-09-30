#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Carleton Coffrin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from collections import namedtuple
from gurobipy import Model, quicksum,GRB

Set = namedtuple("Set", ['index', 'cost', 'items'])

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    parts = lines[0].split()
    item_count = int(parts[0])
    set_count = int(parts[1])
    
    sets = {}
    cost = []
    items = []
    for i in range(1, set_count+1):
        parts = lines[i].split()
#        sets.append(Set(i-1, float(parts[0]), map(int, parts[1:])))
        sets[i-1] = {'cost': int(parts[0]), 'items': [int(parts[i]) for i in range(1, len(parts))]}
        cost.append( int( parts[0]))
        items.append( [int(parts[i]) for i in range(1, len(parts))] )
    
#    print( sets[0])
    item_fire = [ [] for item in range(item_count)]
    item_fire_bool = [ [] for item in range(item_count)]

    for set1 in sets.keys():
        for item in sets[set1]["items"]:           
            item_fire[item].append(set1)
    
    for item in range( item_count):
        for i in range( set_count):
            if i in item_fire[item]:
                item_fire_bool[item].append(1)
            else:
                item_fire_bool[item].append(0)
    

    # Make model
    # Solve the problem as a linear programming problem
    model = Model("set cover")
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', 10*60)
    fire_stat = {}
    
    for i in range( set_count):
        fire_stat[i] = model.addVar( vtype='B', name="fire_stat(%s)" %(i))
        
    model.setObjective( quicksum( fire_stat[i]*cost[i] for i in range(set_count)), GRB.MINIMIZE )
    for i in range( item_count):
        model.addConstr( quicksum( item_fire_bool[i][j]*fire_stat[j] for j in range(set_count)) >= 1 )
    
    model.optimize()
    solution = [ int(fire_stat[i].X) for i in range(set_count)]
    obj = sum( [ cost[i]*solution[i] for i in range(set_count)])
    
    output_data = str(obj) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data


import sys

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')

