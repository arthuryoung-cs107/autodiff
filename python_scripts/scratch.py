import numpy as np
from CMGradobject import CMGobject
def sin(x):
    try:
        val = np.sin(x.val)
        grad = np.cos(x.val)*x.grad
        y = CMGobject(val,grad)
        return y
    except AttributeError:
        return np.sin(x) # if x is a constant

import numpy as np
class CMGobject():
    #Constructor sets value and derivative
    def __init__(self, val, grad: np.array = np.array([1.0])):
            self.val = float(val)
            self.grad = grad
    def __add__(self, other):
        try:
            return CMGobject(self.val+other.val, self.grad+other.grad)
        except AttributeError:
            other = CMGobject(other, np.zeros(np.shape(self.grad)) )  #derivative of a constant is zero
            return CMGobject(self.val+other.val, self.grad+other.grad)
    def __mul__(self, other):
        try:
            return CMGobject(self.val*other.val, np.matmul(np.vstack((self.grad, other.grad)).T, np.array([other.val, self.val] )).reshape(np.shape(self.grad)))
        except AttributeError:
            return CMGobject(self.val*other, self.grad*other)



    def __radd__(self, other): #ensure commutativity of addition
        return self.__add__(other)
    def __repr__(self):
        return f'CMGobject(val = {self.val}, grad = {self.grad})'

    # compares two CMGobjects to determine if they're equal based on their value AND derivative
    # overloading __ne__ unecessary because it just inverts __eq__ by default
    # https://docs.python.org/3/reference/datamodel.html#object.__ne__

    def __eq__(self, other):
        if isinstance(other, CMGobject):
            return self.val == other.val and np.array_equal(self.grad,other.grad)
        return False


    # overload methods to allow for addition of non-class values
    def __mul__(self, other):
        try:
            return CMGobject(self.val*other.val, np.matmul(np.vstack((self.grad, other.grad)).T, np.array([other.val, self.val] )).reshape(np.shape(self.grad)))
        except AttributeError:
            return CMGobject(self.val*other, self.grad*other)

    def __rmul__(self, other):
        return self.__mul__(other)


    def __sub__(self,other):
        try:
            return CMGobject(self.val-other.val, self.grad-other.grad)
        except AttributeError:
            other = CMGobject(other, np.zeros(np.shape(self.grad)) )  #derivative of a constant is zero
            return CMGobject(self.val-other.val, self.grad-other.grad)

    def __rsub__(self,other):
        try:
            return CMGobject(other.val-self.val, other.grad-self.grad)
        except AttributeError:
            other = CMGobject(other, np.zeros(np.shape(self.grad)) )  #derivative of a constant is zero
            return CMGobject(other.val-self.val, other.grad-self.grad)

    # Quotient rule ((v*du/dx - u*dv/dx) / v^2)
    def __truediv__(self,other):
        return self*(other)**(-1)
    def __rtruediv__(self,other):
        return other*(self)**(-1)
    def __pow__(self, other):
        if isinstance(other, CMGobject):
            return_val = (self.val)**(other.val)
            return_grad = return_val*(other.grad*(np.log(self.val)) + self.val**(-1)*(self.grad)*other.val )
            return CMGobject(return_val, return_grad)
        else:
            return CMGobject(self.val**other, other*(self.val)**(other-1)*self.grad)

    def __rpow__(self, other):
        if isinstance(other, CMGobject):
            return other.__pow__(self)
        else:
            return_val = (other)**(self.val)
            return_grad = np.log(other)*(other)**(self.val)*self.grad
            return CMGobject(return_val, return_grad)

    def __neg__(self):
        return CMGobject(-self.val, -self.grad)
