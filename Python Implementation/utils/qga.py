import numpy as np
import jax.numpy as jnp
from jax import jit
from jax import random



 


def Measure_Eval1(q,qubo):  # direct without XGboost data
    fitness, probs = [], []
    born_prob = q**2
    m = q.shape[1]
    k = m**1
   
    for _ in range(k):
        sol_prob = 1
        bitstring = np.zeros(m)
        ran = np.random.random(m)
        for j in range(m):
            #if (born_prob[0,j]) < (born_prob[1,j]):
            if ran[j] < (born_prob[1,j]):
                bitstring[j]=1
    
        X = np.array(bitstring)
        X_T = np.transpose(X)
        fit = np.matmul(X_T,np.matmul(qubo,X))
        fitness.append(fit)

        for i in range(len(bitstring)):
            if X[i] == 0:
                sol_prob = sol_prob * born_prob[0][i]
            elif X[i] == 1:
                sol_prob = sol_prob * born_prob[1][i] 
        probs.append(sol_prob) 
        #print(X,fit,sol_prob) 
    
    return np.dot(fitness,probs)/np.sum(probs)    



def Measure_Eval2(q,qubo):  # direct without XGboost data
    fitness, probs = [], []
    born_prob = q**2
    born_beta = born_prob[1,:]
    m = q.shape[1]
    k = m**1
   
    for _ in range(k):
        ran = np.random.random(m)
        X = (born_prob[1,:] > ran).astype(int)
        X_T = np.transpose(X)
        fit = np.matmul(X_T,np.matmul(qubo,X))
        fitness.append(fit)
        sol_prob = np.prod(np.array([born_prob[1][i] if X[i]==1 else born_prob[0][i] for i in range(X.shape[0])]))    
        probs.append(sol_prob) 
        #print(X,fit,sol_prob) 
    
    return np.dot(fitness,probs)/np.sum(probs)    




def Rotate(temperature,state):
    new_state = []
    for i in range(len(state[0,:])): 
        if np.random.random() > 0.5:
            t = temperature/100
        else:
            t = -temperature/100

        U = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        new = np.matmul(U,state[:,i])        
        new_state.append(new)
    new_state = np.array(new_state)
    return np.transpose(new_state)





def NOT_gate(state):
    new_state = []
    randoms = np.random.random(state.shape[1])
    for i in range(len(state[0,:])): 
        if randoms[i] < 0.1:
            t = np.pi
            U = np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
            new = np.matmul(U,state[:,i]) 
            new_state.append(new)
           
        else:
            new_state.append(state[:,i])
        
    new_state = np.array(new_state)
    return np.transpose(new_state)






def Final_Measure(quantum_system):
    solution_state = []
    born_prob = quantum_system**2
    solution_state = np.rint(born_prob[1,:])
    prob = np.prod(np.array([born_prob[1][i] if solution_state[i]==1 else born_prob[0][i] for i in range(len(solution_state))]))    
    return solution_state,round(prob*100,2)


 ##------------------------------- 0 index means without T


def Replacement0(current_systems,rotated_systems,fit_current,fit_rotated,T,replacement_count): # replaced_systems
    delta_fit = np.array(fit_current) - np.array(fit_rotated)
    probability = 1/(1 + np.exp(-delta_fit))
    randoms = np.random.random(len(current_systems))
    for i in range(len(probability)):
        if probability[i] > 0.5 and probability[i] > randoms[i]:
            current_systems[i] = rotated_systems[i]
            fit_current[i] = fit_rotated[i]
            print('{}th system has been replaced'.format(i))
            replacement_count +=1
    return current_systems, fit_current, replacement_count, delta_fit, randoms, probability



def Mutation0(replaced_systems,mutated_systems,fit_replaced,fit_mutated,T,mutation_count):
    delta_fit = np.array(fit_replaced) - np.array(fit_mutated)
    probability = 1/(1 + np.exp(-delta_fit))
    randoms = np.random.random(len(replaced_systems))
    for i in range(len(probability)):
        if probability[i] > 0.5 and probability[i] > randoms[i]:
            replaced_systems[i] = mutated_systems[i]
            fit_replaced[i] = fit_mutated[i]
            print('{}th system has been mutated'.format(i))
            mutation_count +=1
    return replaced_systems, fit_replaced, mutation_count, delta_fit, randoms, probability



def Migration0(systems,fits,T,migration_count):
    best_fit = min(fits)             
    max_fit_index = fits.index(best_fit)
    diff_fit = np.array(fits) - best_fit
    best_system = systems[max_fit_index]
    # diff_fit = []   
    # for value in fits:
    #     diff_fit.append(value - best_fit)
    # diff_fit = np.array(diff_fit)
    probability = 1/(1 + np.exp(-diff_fit))
    randoms = np.random.random(len(systems))
    for i in range(len(probability)):
        if probability[i] > 0.5 and probability[i] > randoms[i]:
            systems[i] = best_system
            fits[i] = best_fit
            print('{}th system has migrated to {}th system'.format(max_fit_index,i))
            migration_count +=1
    return systems, fits, migration_count, diff_fit, randoms, probability




 ##------------------------------- 1 index means with T



def Replacement1(current_systems,rotated_systems,fit_current,fit_rotated,T,replacement_count):
    delta_fit = np.array(fit_rotated) - np.array(fit_current)
    probability = 1/(1 + np.exp(delta_fit/T))
    randoms = np.random.random(len(current_systems))
    for i in range(len(probability)):
        if probability[i] > 0.5:
           if randoms[i] < probability[i]:
                current_systems[i] = rotated_systems[i]
                print('{}th system has been replaced'.format(i))
                replacement_count +=1
    return current_systems, replacement_count, delta_fit, randoms, probability



def Mutation1(replaced_systems,mutated_systems,fit_replaced,fit_mutated,T,mutation_count):
    delta_fit = np.array(fit_mutated) - np.array(fit_replaced)
    probability = 1/(1 + np.exp(delta_fit))
    randoms = np.random.random(len(replaced_systems))
    for i in range(len(probability)):
        if probability[i] > 0.5 and probability[i] > randoms[i]:
            replaced_systems[i] = mutated_systems[i]
            fit_replaced[i] = fit_mutated[i]
            print('{}th system has been mutated'.format(i))
            mutation_count +=1
    return replaced_systems, fit_replaced, mutation_count, delta_fit, randoms, probability




def Migration1(systems,fits,T,migration_count):              # 1 means with T
    max_fit_index = fits.index(min(fits))
    best_system = systems[max_fit_index]
    diff_fit = []            
    for value in fits:
        diff_fit.append(min(fits) - value)
    diff_fit = np.array(diff_fit)
    probability = 1/(1 + np.exp(diff_fit/T))
    randoms = np.random.random(len(systems))
    for i in range(len(probability)):
        if probability[i] > 0.5:
            if randoms[i] < probability[i]:
                systems[i] = best_system
                print('{}th system has migrated to {}th system'.format(max_fit_index,i))
                migration_count +=1
    return systems, migration_count, diff_fit, randoms, probability











