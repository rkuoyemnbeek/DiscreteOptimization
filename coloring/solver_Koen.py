#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from ortools.sat.python import cp_model
import networkx as nx
import pickle
import copy

def optimize_solution( solution, edge_m):
    k = max( solution)
    node_count = len( solution)
    C = [ [] for j in range(k+1)]
    for node in range(node_count):
        color = solution[node]
        C[color].append( node)
    
    len_C = [ len( C[color]) for color in range(k+1)]
    sort_c = list( np.argsort( len_C))
    C_test = [ C[color] for color in sort_c[k::-1]]

        
    for j in range(k+1):
        for node in C[j]:
            solution[node] = j  
    teller = 0
    
    for teller in range(50):
        j = 0
        while j in range(k+1):
            if (j >= k):
                break
            all_fit = True
            C_test = copy.deepcopy(C[k])
            for node in C[k]:
                fit = False
                c = k-1
                while (fit == False and c >= 0):
                    if sum( [edge_m[node][node1] for node1 in C[c]]) == 0: 
                        # dwz node kan toegevoegd worden aan klasse c
                        C[c].append( node)
                        C_test.remove( node)
    #                    print(c, node, C[c], C_test)
                        fit = True
                    else:
                        c = c - 1
                if fit == False:
                    all_fit = False
                    break
            C[k] = C_test
                        
            if all_fit == True:
                k = k - 1
                j = 0
            else:
                j = j + 1
                C_test = copy.deepcopy( C[k])           
                C[k] = C[k-j]
                C[k-j] = C_test
            solution = [-1]*node_count
            for i in range(k+1):
                for node in C[i]:
                    solution[node] = i  
        
        solution = [-1]*node_count
        for j in range(k+1):
            for node in C[j]:
                solution[node] = j  
    
    return (C, solution, k)



def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    first_line = lines[0].split()
    node_count = int(first_line[0])
    edge_count = int(first_line[1])
    print( node_count, edge_count)
    edges = []
    for i in range(1, edge_count + 1):
        line = lines[i]
        parts = line.split()
        edges.append((int(parts[0]), int(parts[1])))
    
        
    
    buren_d = {}
    edge_m = [ [False]*node_count for _ in range(node_count)]
    for edge in edges:
        if edge[0] in buren_d.keys():
            buren_d[edge[0]].append( edge[1])
        else:
            buren_d[  edge[0]] = [ edge[1]]
        if edge[1] in buren_d.keys():
            buren_d[edge[1]].append( edge[0])
        else:
            buren_d[ edge[1]] = [ edge[0]]
        
        edge_m[edge[0]][edge[1]] = True
        edge_m[edge[1]][edge[0]] = True
    
    
    if ( (node_count != 250) and (node_count != 1000)):
        # voor al deze volstond het om: 
        # 1) greedy opl maken
        # 2) deze opl verder te optimaliseren met or-tools
        
        # greedy solution    
        nbr_neighbours = [len( buren_d[item]) for item in range(node_count)]
        colors_chosen = [node_count]*node_count
        colors_not_chosen = [ [] for _ in range(node_count) ]
        idxs = np.argsort(nbr_neighbours)
        
        
        for idx in reversed(idxs):
            test = set( list( np.arange(node_count))) - set( colors_not_chosen[idx])
            c_idx = min(test )
            colors_chosen[idx] = c_idx
            for node in buren_d.keys():
                if idx in buren_d[node]:
                    colors_not_chosen[node].append( c_idx)
                    
        nbr_col_greedy = max( colors_chosen) + 1
        
        solution1 = [0]
        
        # Kijken of er complete subgrafen zijn
        # G = nx.Graph()
        # G.add_nodes_from( range(node_count))
        # G.add_edges_from( edges)
        
        # all_subgraphs = list( nx.find_cliques(G))
        
        # print( 'found all subgraphs')
        # with open( 'gc_250_9.data', 'wb') as filehandle:
        #     pickle.dump(all_subgraphs, filehandle)
        
        # with open('listfile.data', 'rb') as filehandle:
        # # read the data as binary data stream
        #     placesList = pickle.load(filehandle)
        
        
        
        ## model definieren            
        model = cp_model.CpModel()
       
        node_l = [model.NewIntVar(0, int( nbr_col_greedy), 'x[%i]' % i) for i in range(node_count)]
        max_color = model.NewIntVar(1, int( nbr_col_greedy)+1, 'max_color')
        model.AddMaxEquality(max_color, node_l);
                
        # adding constraints    
        for edge in edges:
            model.Add( node_l[edge[0]] != node_l[edge[1]] )
        for i in range(nbr_col_greedy):
            model.Add(node_l[i] <= i);
        # for subgraph in all_subgraphs:
        #     model.AddAllDifferent( [node_l[node] for node in subgraph])
        
        
        # objective functie:
        model.Minimize(max_color)
        
        ## oplossen 
        max_minutes = 2
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60*max_minutes
        status = solver.Solve(model)
        print('Status = %s' % solver.StatusName(status))
        
        # for node in range(node_count):
        #     print( node, solver.Value( node_l[node]))
        solution = [ solver.Value( node_l[node]) for node in range(node_count)]
        min_k = int(solver.ObjectiveValue())
    else:
        
        k = node_count
        min_k = k
    
    
    # maken van initiele oplossing
        for i in range(100):
            min_k = min( min_k, k)
            print( "aantal klassen: ", k)
            print( '-----------------------------')
            colors_chosen = list( range(node_count))
            idxs = np.random.permutation(node_count)
            k = node_count
            solution = [colors_chosen[idx] for idx in idxs]
            C, solution, k = optimize_solution( solution, edge_m)  
            if k < min_k:
                min_k = k
                C_best = C
                solution_best = solution
        print( min_k)                
    
        # with open('1000_5_initial_solution.pickle', 'wb') as f:
        #     pickle.dump( (C_best, solution_best, min_k) , f)

#    C_best, solution_best, min_k = pickle.load(open("1000_5_initial_solution.pickle", "rb"))
        obj_funct_best = sum( [ len( C_best[i])**2 for i in range(min_k+1)] )
        k = copy.deepcopy( min_k)
        t0 = node_count/(2*min_k)
        C = copy.deepcopy( C_best)
        solution = copy.deepcopy( solution_best)
        obj_funct = copy.deepcopy(obj_funct_best)
        teller = 0
        
        ## soort van simulated annealing gecombineerd met een stuk eigen code:
        ## de eigen code probeert de kleuren die weinig voorkomen te elemineren
        ## door de node die deze kleur hebben, te verdelen over de andere kleuren
        
        for i in range(700000):
            ## kies node
            
            valid = False
            while valid == False:
                
                node = np.random.randint(0, node_count)
                kleur_node = solution[node]
                kleur = np.random.randint(0, k+1)
                while kleur == kleur_node: #stel dat we toevallig dezelfde kleur hebben gekozen, kiezen we opnieuw
                    kleur = np.random.randint(0, k+1)
                if sum( [edge_m[node][node1] for node1 in C[kleur]]) == 0: 
                    # als we feasible blijven
                    if np.random.uniform(0,1) <= np.exp(2*( len(C[kleur]) - len( C[kleur_node]))/t0):
    #                    print( np.exp(2*( len(C[kleur]) - len( C[kleur_node]))/t0))
                        # Metropolis-Hastings ding
                        valid = True
                        solution[node] = kleur
                        C[kleur].append( node)
                        C[kleur_node].remove( node)
                        obj_funct = obj_funct + 2*( len(C[kleur]) - len( C[kleur_node]))
                    else:
                        teller = teller + 1
                        # print( i, teller)
                        
            k = max( solution)
    #        t0 = node_count/(2*min_k)
    #        t0 = 0.9999*t0
            if (k < min_k) or (k == min_k and obj_funct_best < obj_funct):
                C_best = copy.deepcopy( C)
                solution_best = copy.deepcopy( solution)
                
                if (k < min_k):
                    print( 'new best', k+1, i)
                    min_k = copy.deepcopy( k)
                # else:
                #     print( 'new obj', obj_funct)
                
            if (i % (node_count)) == 0:
                if i % 10000 == 0:
                    print( i)
                C_best, solution_best, k = optimize_solution( solution, edge_m)
                if k < min_k:
                    print( 'new best:', k+1)
                min_k = copy.deepcopy( k)
                obj_funct_best = sum( [ len( C_best[i])**2 for i in range(min_k+1)] )
                C = copy.deepcopy( C_best)
                solution = copy.deepcopy( solution_best)
                obj_func = copy.deepcopy( obj_funct_best)
            if (node_count == 1000 and k + 1 == 100) or (node_count == 250 and k + 1 == 78): 
                print( 'hiep hiep hoera, gevonden!')
                break
                
    output_data = str(min_k+1) + ' ' + str(0) + '\n'
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
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/gc_4_1)')

