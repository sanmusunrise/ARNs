import numpy as np

def argsort(x,reverse = False):
    order = np.argsort(x).tolist()
    if reverse:
        order.reverse()
    return order


if __name__ =="__main__":
    a = [1,4,3,5,2]
    order = argsort(a,reverse = True)
    print "Original:",a
    print "Order:",order
    b = [a[i] for i in order]
    print "Sorted:",b
    c = argsort(order,reverse = False)
    d = [b[i] for i in c]
    print "Recovered:",d

