import numpy as np
import matplotlib.pyplot as plt
from utils.qubo import *
from utils.utility import *
from utils.qga import *
import time
import math



filename = 'iris'
data = pd.read_csv('datasets/'+ str(filename) + '.csv')
qubo = pd.read_csv('qubos/'+ str(filename) + '_qubo.csv')
qubo = qubo.values[:,1:(qubo.shape[0]+1)]


n = qubo.shape[1]     # number of qubits and quantum systems
q = (1/np.sqrt(2))*np.ones((2,n))  # Initialization of Quantum system with equal superposition
print('We have {} number of Quantum systems'.format(n))
current_systems = n*[q]
iteration,T = 1,100
iteration_count = []
Fits, temps = [], []


#np.random.seed(20)  ###########  random seed


migration_count,replacement_count, mutation_count = 0,0,0
beta_squared, detail, fitness_detail = [], [], []
replacement_records, mutation_records, migration_records = [], [], []

sys_prob_log = np.zeros(n)    # systems's probability log report
all_sys_prob_log = []



start_time = time.time()
while sys_prob_log.min() < 99.5 and T>1:
    iteration_count.append(iteration)
    print('############################################################# Iteration No: {}'.format(iteration))
    print('Current temperature is---------------------------------------------------------------------',T)
    fit_current = [Measure_Eval2(system,qubo) for system in current_systems]
    rotated_systems = [Rotate(T,system) for system in current_systems]
    fit_rotated = [Measure_Eval2(system,qubo) for system in rotated_systems]    
    replaced_systems,fit_replaced,replacement_count,del_fit_rep,ran_rep,prob_rep = Replacement0(current_systems,rotated_systems,fit_current,fit_rotated,T,replacement_count)
    replacement_records.append(replacement_count)   
    # fit_replaced = [Measure_Eval3(system,qubo) for system in replaced_systems] # not required

    mutated_systems = [NOT_gate(system) for system in replaced_systems]
    fit_mutated = [Measure_Eval2(system,qubo) for system in mutated_systems]    
    adapted_systems,fit_adapted,mutation_count,del_fit_rep,ran_rep,prob_rep = Mutation0(replaced_systems,mutated_systems,fit_replaced,fit_mutated,T,mutation_count)
    mutation_records.append(mutation_count)
    # fit_adapted = [Measure_Eval3(system,qubo) for system in adapted_systems]  # not required  
        



    migrated_systems,fit_migrated,migration_count,diff_fit_mig,ran_mig,prob_mig = Migration0(adapted_systems,fit_adapted,T,migration_count)
    migration_records.append(migration_count)
    #migrated_systems = replaced_systems  #------un-comment this for no migrations
    # fit_migrated = [Measure_Eval3(system,qubo) for system in migrated_systems] # not required

    sys_prob_log = np.array([Final_Measure(system)[1] for system in migrated_systems])
    all_sys_prob_log.append(sys_prob_log)
    print('\n\nSystem probabilities {} #######\n'.format(sys_prob_log))




    #################################################################
    print('fit Current:',fit_current)
    print('fit  Rotate:',fit_rotated)
    print('fit Replace:',fit_replaced)
    print('fit Mutated:',fit_mutated)
    print('fit Migrate:',fit_migrated)

    fitness_detail.append(fit_current)
    fitness_detail.append(fit_rotated)
    fitness_detail.append(fit_replaced)
    fitness_detail.append(fit_mutated)
    fitness_detail.append(fit_migrated)
    fitness_detail.append([iteration,'$','$','$','$'])

    #################################################################
    Fits.append(fit_migrated)
    current_systems = migrated_systems
    beta_squared.append([system[1,:]**2 for system in current_systems])
    ################################################################
    
    detail.append(del_fit_rep) 
    detail.append(ran_rep)
    detail.append(prob_rep)
    detail.append(diff_fit_mig)
    detail.append(ran_mig)
    detail.append(prob_mig)   
    detail.append([iteration,'$','$','$','$'])  


    print('\n\n\n')
    temps.append(T)
    T = T*0.99
    iteration += 1

end_time = time.time()
results = [Final_Measure(system) for system in current_systems]
best_result,E = best_system(results,qubo)


##################  Result of Measurement on all systems  #################
print('\n')
print('Best system is:',best_result)
print('Energy of the best system is:',round(E,4))
print('Number of selected features is:',sum(best_result))
print('Total number of iteration is:',iteration)
print('Execution time is (seconds):',end_time - start_time)
print('Total number of Replacement is:',replacement_count)
print('Total number of  Migrations is:',migration_count)
print('Total number of  Mutations is:',mutation_count)
print('Prediction accuracy with complete data is:',round(prediction(data),2),'%')
print('Prediction accuracy with new data is:',round(prediction(New_data(best_result,data)),2),'%')
print('\n\nLog of indivisual Quantum systems')
for i in range(len(results)):
    print('quantum system {} is: {} with probability {}%'.format(i+1,results[i][0],results[i][1]))



########## fitness vs temperature plot of quantum systems  ##########
Fits = np.array(Fits)
all_sys_prob_log = np.array(all_sys_prob_log)


for i in range(Fits.shape[1]):
    plt.plot(iteration_count,Fits[:,i])

plt.xlabel('Iteration count')
plt.ylabel('fitness')
plt.title('Iteration vs Fitness')
plt.xlim(0,iteration+5)
plt.grid(True)
plt.show()

############# Iteration count vs system probability  ##########
for i in range(all_sys_prob_log.shape[1]):
    plt.plot(iteration_count,all_sys_prob_log[:,i])

plt.xlabel('Iteration count')
plt.ylabel('system probability')
plt.title('Iteration vs system probability')
plt.xlim(0,iteration+5)
plt.grid(True)
plt.show()