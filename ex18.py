import numpy as np
from numpy.linalg import inv
import sys



def simplex_agorithm(c, A, b, basis, pricing = False):
    
    #define temp_basis, not temp basis and supposedly feasible temp_basis vector
    temp_basis = basis
    x = np.zeros(c.size)
    x[temp_basis] = np.dot(inv(A[:,temp_basis]), b.T)
    ntemp_basis = [k for k in range(c.size) if k not in basis]

    if any(x[temp_basis]<0):
        print(f"{basis} is not a feasible basis")

    i = 0
    while(i < 100):
   

        #calculate reduced cost vector
        c_reduced = c[ntemp_basis] - np.dot(A[:,ntemp_basis].T, np.dot(inv(A[:,temp_basis]).T, c[temp_basis]))
        #check if c_reduced has  an element < 0 , if not => found opt
        if not any(c_reduced < 0.0):
            print("\nthe following is an optimal solution:\n")
            print(f"x = {x}")
            print(f"f(x) = {np.dot(c,x)}")
            break
        elif(pricing == False):

            #choose r as given by task (first min of c_reduced)
            
            r = ntemp_basis[np.where(c_reduced == c_reduced.min())[0][0]]

            d = np.dot(inv(A[:,temp_basis]), A[:,r])
            #check if d <= 0 everywhere -> porblem unbounded
            if all(d<=0):
                print("problem is unbounded")
                break

            d_min_plus = d[np.where(d>0)].min()
            x_div_d = (x[temp_basis]/d)

            t = (x[temp_basis][np.where(d == d_min_plus)[0]]/d[np.where(d == d_min_plus)[0]])[0]
           

            #choose s as given by task 
            s = min([temp_basis[i] for i in np.intersect1d(np.where(x_div_d == t)[0], np.where(d>0)[0])])


        elif(pricing == True):

            #choose r as given by task (Bland)
            r = min([ntemp_basis[i] for i in np.where(c_reduced < 0.0)[0]])

            d = np.dot(inv(A[:,temp_basis]), A[:,r])
            #check if d <= 0 everywhere -> porblem unbounded
            if all(d<=0):
                print("problem is unbounded")
                break
            
            d_min_plus = d[np.where(d>0)].min()
            x_div_d = (x[temp_basis]/d)

            t = (x[temp_basis][np.where(d == d_min_plus)[0]]/d[np.where(d == d_min_plus)[0]])[0]
            #choose s as given by task (Bland)
            s = min([temp_basis[i] for i in np.intersect1d(np.where(x_div_d == t)[0], np.where(d>0)[0])])


        temp_basis.remove(s)
        temp_basis.append(r)
        temp_basis.sort()
        ntemp_basis.remove(r)
        ntemp_basis.append(s)
        ntemp_basis.sort()

        x[temp_basis] = x[temp_basis] - t*d
        x[ntemp_basis] = 0.0
        x[r] = t
        i+=1
    print(f"steps: {i}")
    return([x[temp_basis], temp_basis,i])
        






def main():
    
    #ex 14
    c = np.array([3.0,1.0,0.0,0.0])
    A = np.array([[1.0,-1.0,1.0,0.0], [1.0,-3.0,0.0,1.0]])
    b = np.array([3.0,1.0])
    basis = [0 , 1]

    #ex20b
    #c = np.array([-1.0,-1.0,0.,0.,0.])
    #A = np.array([[2.0,1.0,-1.0,1.0,0.0],[1.0,1.0,-1.0,0.0,1.0]])
    #b = np.array([8.0,6.0])
    #basis = [0,1]
    

    #ex20a
    #c = np.array([-1.,-1.,0.,0.])
    #A = np.array([[-1.,2.,1.,0.], [1.,0.,0.,1.]])
    #b = np.array([2.,2.])
    #basis=[0,1]

    #ex17
    #c = np.array([-2.,-3.,1.,12.,0.,0.])
    #A = np.array([[-2.,-9.,1.,9.,1.,0.],[1.,3.,-1.,-6.,0.,3.]])
    #b = np.array([0.,0.])
    #basis = [4,5]

    #ex18)d
    c = np.array([-3./4., 150., -1./50., 6., 0.,0.,0.])
    A = (np.array([[1./4., -60., -1./25., 9.,1., 0.,0.], [1./2., -90., -1./50., 3., 0.,1.,0.], [0.,0.,1.,0.,0.,0.,1.]]))
    b = np.array([0.,0.,1.])
    basis = [4,5,6]

    [x_basis, my_basis, i] = simplex_agorithm(c,A,b,basis,True)

if __name__ == "__main__":
    main()