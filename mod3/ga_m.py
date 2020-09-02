import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import math
import random

def _function(x):
    if x>-63 and x< 63:
        y =(x**3)+9
        return round(y,6)
    else:
        return 0
function = np.vectorize(_function)


def individual_generation(num_of_genes):
    return [random.randint(0,1) for x in range(num_of_genes)]

def fitness_cal(individual):
    m = len(individual)
    f = 0
    for x in range(m):
        f += individual[x]*(2**(m-1-x))
    f = function(f)
    return f

def pop_gen(size_of_population, num_of_genes):
    population = [individual_generation(num_of_genes) for x in range(size_of_population)]
    return population

def fitnesss(population,size_of_population):
    return [fitness_cal(population[i]) for i in range(size_of_population)]

def probability_cal(fitness):
    total = float(sum(fitness))
    ind_prob = [f/total for f in fitness]
    probabilities = [sum(ind_prob[:i+1]) for i in range(len(ind_prob))]
    return probabilities

def single_point_crossover(ind1,ind2):
    m = len(ind1)
    t1 = ind1.copy()
    t2 = ind2.copy()
    for i in range(m//2):
        t1[i] = t2[i]
    for i in range(m//2,m):
        t2[i] = t1[i]
    return t1,t2

def rou_selection(population,number,probabilities):
    chosen = []
    for n in range(number):
        r = random.random()
        for i in range(len(population)):
            if(r <= probabilities[i]):
                chosen.append(population[i])
                break
    return chosen

def mutation(individual,p):
    for i in range(len(individual)):
        r = random.random()
        if(r <= p):
            if(individual[i] == 0):
                individual[i] = 1 
            else:
                individual[i] = 0
    return individual

def max_2(parents_fit):
    l = len(parents_fit)
    i1 = parents_fit.index(max(parents_fit))
    t = -99999
    i2 = -1
    for i in range(l):
        if((parents_fit[i] > t) and (i != i1)):
            t = parents_fit[i]
            i2 = i
    return i1,i2

def min_2(offspring):
    l = len(offspring_fit)
    i1 = offspring_fit.index(min(offspring_fit))
    t = 9999999
    i2 = -1
    for i in range(l):
        if((offspring_fit[i] < t) and (i != i1)):
            t = offspring_fit[i]
            i2 = i
    return i1,i2


number_of_itr = 12
mut_prob = 0.01
size_of_pop = 10
population = pop_gen(size_of_pop ,6)
for x in range(number_of_itr):
    fit = fitnesss(population,size_of_pop)
    probabilities = probability_cal(fit)
    parents = rou_selection(population,10,probabilities)
    parents_fit = fitnesss(parents,10)
    offspring = []
    i = 0
    while(i < 10):
        t1,t2 = single_point_crossover(parents[i],parents[i+1])
        offspring.append(t1)
        offspring.append(t2)
        i += 2
    i = 0
    while(i < 10):
        offspring[i] = mutation(offspring[i],mut_prob)
        i += 1
    offspring_fit = fitnesss(offspring,10)
    k1,k2 = max_2(parents_fit)
    k3,k4 = min_2(offspring)
    offspring[k3] = parents[k1]
    offspring[k4] = parents[k2]
    #print(i1,i2,i3,i4)
    print(offspring)
    print(population)
    print(parents)
    print(fit)
    #print(parents_fit)
    #print(offspring_fit)
    #print(probabilities)
    print('After Generation ',x,':',max(fit))
    population = offspring
##############################
def _fitness(x):
    if x>-63 and x< 63:
        y =(x**3)+9
        return round(y,6)
    else:
        return 0
fitness = np.vectorize(_fitness)

x = np.linspace(start= -63,stop=63 ,num=200)
plt.plot(x,fitness(x))
#################################

def mutate(parents, fitness_function):#takes a list of parents and fittness function
    n=int(len(parents))
    scores= fitness_function(parents)#calculating fitness for every individual
    idx = scores > 0 #filter for only positives values only
    scores=scores[idx]
    parents = np.array(parents)[idx]
    #resample parents with probabilities proportional to fitness
    #then, add some noise for 'random' mutation
    children=np.random.choice(parents,size=n,p=scores/scores.sum())#resample based on fittness(Roulette Wheel Selection)
    children=children+np.random.uniform(-0.51,0.51,size=n)#add some noise to mutate
    return children.tolist()#convert to array to list
    
############################################
def GA(parents,fitness_function,popsize=10,max_iter=100):
    History=[]    
    #initial parents;gen zero
    best_parent, best_fitness= _get_fitness_parent(parents,fitness)#extract fitness individual
    
    #first plot initial parents
    x = np.linspace(start= -20,stop=20 ,num=200)#population range
    plt.plot(x,fitness_function(x))
    plt.scatter(parents,fitness_function(parents),marker='x')
    
    #for each next generation
    for i in range(1,max_iter):
       # parents=crossover(parents, offspring_size=parents.shape[0])
        parents= mutate(parents,fitness_function=fitness_function)
        
        curr_parent,curr_fitness=_get_fitness_parent(parents,fitness_function)
        
        #update best fitness values
        if curr_fitness>best_fitness:
            best_fitness = curr_fitness
            best_parent = curr_parent
            
        curr_parent, curr_fitness = _get_fitness_parent(parents,fitness_function)
        
            
        History.append((i,np.max(fitness_function(parents)))) #save generation MAX fitness
            
        plt.scatter(parents,fitness_function(parents))
        plt.scatter(best_parent, fitness_function(best_parent), marker='.',c='b',s=200)
        plt.pause(0.09)
        plt.ioff()
        #return best parents
        return best_parent,best_fitness,History
#################################################################
        
def _get_fitness_parent(parents,fitness):
    _fitness =fitness(parents)
    PFitness= list(zip(parents,_fitness))
    PFitness.sort(key= lambda x:x[1],reverse=True)
    best_parent,best_fitness=PFitness[0]
    return round(best_parent,4), round(best_fitness,4)

####################################################################
    
x = np.linspace(start= -63,stop=63,num=200)#population range
init_pop=np.random.uniform(low=-63,high=63,size=100)

#################################################################

parent_,fitness_,history_ =GA(init_pop,fitness)


