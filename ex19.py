import ex18
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import timedelta

def main():
    
    As = []
    bs = []
    cs = []
    bases = []

    for n in range(4,11):
        #generate Matrix
        l = [[2.**(j-i) for i in range(n)] for j in range(n)]
        A = np.tril(l)
        np.fill_diagonal(A,1.)
        A = np.concatenate((A,np.eye(n)),axis=1)
        As.append(A)

        #generate b
        bs.append(np.array([5.**(i+1) for i in range(n)]))

        #generate c
        cs.append(np.array([-(2.**(n - 1 -i)) if (n-i >0) else 0.for i in range(2*n)]))

        #generate basis
        bases.append([i for i in range(n,2*n)])





    #ex 14
    #c = np.array([3.0,1.0,0.0,0.0])
    #A = np.array([[1.0,-1.0,1.0,0.0], [1.0,-3.0,0.0,1.0]])
    #b = np.array([3.0,1.0])
    #basis = [0 , 1]
#
    #ex18.simplex_agorithm(c,A,b,basis,True)

    #list of basis solutions with basis and steps
    returns = []
    times = []
    for i in range(7):
        print("n = {i}")
        s = time.perf_counter()
        temp = ex18.simplex_agorithm(cs[i],As[i],bs[i],bases[i],False)
        del_t = time.perf_counter() - s
        print(" ")
        returns.append(temp)
        times.append(del_t)

    #get list of steps
    steps = []
    for x in returns:
        steps.append(x[2])

    #plot
    fig1, ax1 = plt.subplots()
    ax1.plot(range(4,11), steps)
    ax1.set(xlabel="n",ylabel = "steps", title="How many steps does it take to calc an opt solution?")
    plt.savefig("myplt_steps.png")
    plt.show()

    fig2, ax2 = plt.subplots()
    ax2.plot(range(4,11), times)
    ax2.set(xlabel="n",ylabel = "times[s]", title="How much time does it take to calc an opt solution?")
    plt.savefig("myplt_time.png")
    plt.show()


if __name__ == "__main__":
    main()