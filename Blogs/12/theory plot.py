import numpy as np
import matplotlib.pyplot as plt

def discrete_plot(f, domain, filename="theory-plot.png", dpi=300):
    x_vals = np.array(domain)
    y_vals = np.array([f(x) for x in x_vals])
    
    plt.figure(figsize=(8, 4), dpi=dpi)
    plt.scatter(x_vals, y_vals, color='b')
    plt.vlines(x_vals, 0, y_vals, colors='gray', linestyles='dotted')
    
    plt.ylim(0.5, 0.7)
    plt.xlabel('Number of chambers in the barrel')
    plt.ylabel('Probability of A losing')
    #plt.title('Discrete Plot')
    #plt.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.show()






discrete_plot(lambda x: x/(2*x-1), range(2, 16))