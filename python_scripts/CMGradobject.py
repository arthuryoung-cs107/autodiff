import numpy as np

class CMGobject():
    """Creates a forward automatic differentiation class:
        CMGobject(val, grad=np.array([1.0]))

    INPUTS
    ======
    val : the value of the object
    grad : the gradient of the object, default seed = np.array([1.0])

    RETURNS
    =======
    CMGobject for forward automatic differentiation

    EXAMPLES
    ========
    >>> x = CMGobject(4, np.array([2]))
    >>> x.val
    4
    >>> x.grad
    [2]
    """
    #Constructor sets value and derivative
    def __init__(self, val, grad: np.array = np.array([1.0])):
        try:
            self.val = float(val)
            self.grad = grad
        except:
            raise ValueError('ValueError: val and grad must be real numbers.')

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

    def __add__(self, other):
        try:
            return CMGobject(self.val+other.val, self.grad+other.grad)
        except AttributeError:
            other = CMGobject(other, np.zeros(np.shape(self.grad)) )  #derivative of a constant is zero
            return CMGobject(self.val+other.val, self.grad+other.grad)

    def __radd__(self, other): #ensure commutativity of addition
        return self.__add__(other)

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


class CMvector():
    """Creates a forward automatic differentiation class for vector valued functions:
        CMGobject(f_list)

    INPUTS
    ======
    f_list : a list of CMGradobjects that have attributes of self.val and self.grad

    RETURNS
    =======
    CMvector : a vector valued function for forward automatic differentiation
    which has attributes of self.val and self.jac (jacobian)
    
    EXAMPLES
    ========
    >>> F_list = [2*x3 + CMfunc.cos(x1 - 2*x2), 3*x4 - x3, x2**x4, 1/(x3 - x4)  ]
    >>>     (where x1 thru x4 are of class CMGradobject)
    >>> F = CMvector(F_list)
    >>> F.val
    array([ 5.0100075 , 9. , 16. , -1. ])

    >>> F.jac
    array([[ 0.14112001, -0.28224002,  2.        ,  0.        ],
           [ 0.        ,  0.        , -1.        ,  3.        ],
           [ 0.        , 32.        ,  0.        , 11.09035489],
           [-0.        , -0.        , -1.        ,  1.        ]])
    """
    def __init__(self, f_list):
        self.val = np.array([f_list[0].val])
        self.jac = np.array([ f_list[0].grad ])
        for func in f_list[1:]:
            self.val = np.append(self.val, func.val)
            self.jac = np.vstack((self.jac, [func.grad]))
    def __add__(self, other):
        if isinstance(other, CMvector):
            assert other.val.shape[0] == self.val.shape[0]
            assert other.jac.shape[1] == self.jac.shape[1]
            val_out = self.val + other.val
            jac_out = self.jac + other.jac
            return_list = []
            for i, val in enumerate(val_out):
                return_list.append(CMGobject(val, jac_out[i]))
            return CMvector(return_list)
        elif isinstance(other, CMGobject):
            assert other.grad.shape[0] == self.jac.shape[1]
            val_out = self.val + other.val
            jac_out = np.add(self.jac, other.grad)
            return_list = []
            for i, val in enumerate(val_out):
                return_list.append(CMGobject(val, jac_out[i]))
            return CMvector(return_list)
        else:
            #print("get it together mate")
            raise ValueError('Must be added with CMGobject or CMvector')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, CMvector):
            assert other.val.shape[0] == self.val.shape[0]
            assert other.jac.shape[1] == self.jac.shape[1]
            val_out = self.val - other.val
            jac_out = self.jac - other.jac
            return_list = []
            for i, val in enumerate(val_out):
                return_list.append(CMGobject(val, jac_out[i]))
            return CMvector(return_list)
        elif isinstance(other, CMGobject):
            assert other.grad.shape[0] == self.jac.shape[1]
            val_out = self.val + other.val
            jac_out = np.add(self.jac, -1*other.grad)
            return_list = []
            for i, val in enumerate(val_out):
                return_list.append(CMGobject(val, jac_out[i]))
            return CMvector(return_list)
        else:
            #print("get it together mate")
            raise ValueError('Must be added with CMGobject or CMvector')

    def __rsub__(self, other):
        return self.__sub__(other)

    def __repr__(self):
        return f'CMvector(val = {self.val}, \n jacobian = {self.jac})'
