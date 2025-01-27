import random
import numpy as np
import matplotlib.pyplot as plt


def russian_roulette(n):
    """
    Run a game of RR with 2 players
    The function returns 1 if player 1 loses
                         2 if player 2 loses
    """
    players = [1, 2]
    while True:
        for player in players:
            if random.randint(1, n) == 1:  # 1/n chance of losing
                return player
            
plays = [10,50,100,500,1000,5000,10000,50000,100000,500000,1000000]

#plays = [100,500,1000]

N = 6

total_run = 10 # number of times to cycle through plays list

theory_lost = N / (2*N-1)

data = []

for z in range(total_run):
    lost_by_A = []
    
    lost_by_B = []
    
    for i in plays:
        A = 0
        B = 0
        for j in range(i):
            result = russian_roulette(N)
            if result == 1:
                A = A +1
            elif result == 2:
                B = B +1
        lost_by_A.append(A)
        lost_by_B.append(B)
        
    probabilities = []
        
    for k in range(len(plays)):
        probabilities.append(lost_by_A[k]/plays[k])
    data.append(probabilities)
    
probabilities_average = []
probabilities_error = []

for i in range(len(data[0])):
    column = [sublist[i] for sublist in data]
    probabilities_average.append(np.mean(column))
    probabilities_error.append(np.std(column))
    

        

plt.figure(figsize=(8, 4), dpi=300)
plt.errorbar(plays, probabilities, yerr=probabilities_error, fmt='o', 
             color='black', 
             ecolor='red', 
             elinewidth=2, 
             capsize=5, 
             capthick=2, 
             linestyle='None')

#plt.ylim(0.0, 1)
plt.xscale('log')
plt.xlabel('Number of games')
plt.ylabel('Observed games lost by A')
plt.title('Average over 10 simulations of x games with 6 chambers')
#plt.legend()
plt.grid(True, linestyle='--', alpha=0.2)

plt.axhline(y=theory_lost, color='green', linestyle='--', linewidth=2,label='Theoretical value')

plt.savefig("6-chambers.png", dpi=300, bbox_inches='tight')
plt.show()


        
    