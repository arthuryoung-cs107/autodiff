import numpy as np
from CMGradobject import CMGobject as cmg
import CMfunc as cm

def cart2pol(cart_vec): # converts position coordinates
    pol_vec = np.zeros(np.shape(cart_vec) )
    for i, vec in enumerate(cart_vec):
        r = np.linalg.norm(vec)
        if vec[0] == 0:
            theta = np.pi/2 + np.pi*(0**( 1 + np.sign(vec[1])))
        else:
            theta = np.arctan(vec[1]/vec[0]) + np.pi*(0**( 1 + np.sign(vec[0])))

        pol_vec[i][0] = r
        pol_vec[i][1] = theta
    return pol_vec

def pol2cart(pol_vec): # converts position coordinates
    cart_vec = np.zeros(np.shape(pol_vec) )
    for i, vec in enumerate(pol_vec):
        cart_vec[i][0] = vec[0]*np.cos(vec[1])
        cart_vec[i][1] = vec[0]*np.sin(vec[1])
    return cart_vec

def pol2cart_grad(grad_vec_pol, pos_vec_pol):
    grad_vec_cart = np.zeros(np.shape(grad_vec_pol))
    pos_vec_cart = pol2cart(pos_vec_pol)
    for i, vec in enumerate(grad_vec_pol):
        grad_vec_cart[i][0] = vec[0]*np.cos(pos_vec_pol[i][1]) + vec[1]*np.cos((np.pi/2) + pos_vec_pol[i][1])/pos_vec_pol[i][0]
        grad_vec_cart[i][1] = vec[0]*np.sin(pos_vec_pol[i][1]) + vec[1]*np.sin((np.pi/2) + pos_vec_pol[i][1])/pos_vec_pol[i][0]
    return grad_vec_cart

class Flow_it():
    def __init__(self, input_list):
        self.flow = input_list
        self.index = 0
        self.max_it = len(input_list)
    def __iter__(self):
        return self
    def __next__(self):
        if self.index < self.max_it:
            next_val = self.flow[self.index]
            self.index += 1
            return next_val
        else:
            raise StopIteration
    def __add__(self, other):
        return_list = self.flow
        for i, f in enumerate(other):
            return_list[i] += f
        return Flow_it(return_list)

class Flow():
    def __init__(self, key, inputs):
        self._strength = inputs[0]
        self._pos = np.array([0, 0])
        self._key = key
        self.CMGs = []
        self._points = np.array([])
    def rule_out_points(self, cart_coords: np.array):
        return cart_coords
    def compute_points(self, cart_coords: np.array):
        self._points = cart2pol(cart_coords)
    def compute_flow(self):
        raise NotImplementedError
    def __iter__(self):
        return Flow_it(self.CMGs)

class uniform(Flow):

    def rule_out_points(self, cart_coords: np.array):
        return_points = np.array([0, 0])
        for pos in cart_coords:
            if not (pos[0] == self._pos[0] and pos[1] == self._pos[1]):
                return_points = np.vstack((return_points, [pos]))
        try:
            return return_points[1:]
        except ValueError:
            print("Somehow, you have only given points that lie in singularities. Please reconsider your decisions.")
    def compute_flow(self):
        for pos in self._points:
            r = cmg(pos[0], np.array([1., 0.]))
            theta = cmg(pos[1], np.array([0., 1.]))
            self.CMGs.append(self._strength*r*cm.cos(theta))
        return Flow_it(self.CMGs)

    def __repr__(self):
        return repr(self._key + ': a uniform flow of strength {}'.format(self._strength) )

class source(Flow):
    def __init__(self, key, inputs):
        self._strength = inputs[0]
        self.b = inputs[1]
        self._pos = np.array([inputs[2], inputs[3]])
        self._key = key
        self.CMGs = []
        self._points = np.array([])

    def rule_out_points(self, cart_coords: np.array):
        return_points = np.array([0, 0])
        for pos in cart_coords:
            if not (pos[0] == self._pos[0] and pos[1] == self._pos[1]):
                return_points = np.vstack((return_points, [pos]))
        try:
            return return_points[1:]
        except ValueError:
            print("Somehow, you have only given points that lie in singularities. Please reconsider your decisions.")

    def compute_points(self, cart_coords: np.array):
        self._points = cart2pol(np.subtract(cart_coords, self._pos))
    def compute_flow(self): ## assumes, for now, unitary b
        for pos in self._points:
            r = cmg(pos[0], np.array([1., 0.]))
            theta = cmg(pos[1], np.array([0., 1.]))
            self.CMGs.append((self._strength/(2*np.pi*self.b))*cm.log(r))
        return Flow_it(self.CMGs)

    def __repr__(self):
        return repr(self._key + ': a source of strength {} at (x, y) = {}'.format(self._strength, self._pos ) )

class sink(source):
    def __init__(self, key, inputs):
        self._strength = -inputs[0]
        self.b = inputs[1]
        self._pos = np.array([inputs[2], inputs[3]])
        self._key = key
        self.CMGs = []
        ##### Changed Here because I did not find self._points ########
        self._points = np.array([])
    def __repr__(self):
        return repr(self._key + ': a sink of strength {} at (x, y) = {}'.format(self._strength, self._pos ) )

class vortex(Flow):
    def __init__(self, key, inputs):
        self._strength = inputs[0]
        self._pos = np.array([inputs[1], inputs[2]])
        self._key = key
        self.CMGs = []
        self._points = np.array([])


    def rule_out_points(self, cart_coords: np.array):
        return_points = np.array([0, 0])
        for pos in cart_coords:
            if not (pos[0] == self._pos[0] and pos[1] == self._pos[1]):
                return_points = np.vstack((return_points, [pos]))
        try:
            return return_points[1:]
        except ValueError:
            print("Somehow, you have only given points that lie in singularities. Please reconsider your decisions.")

    def compute_points(self, cart_coords: np.array):
        self._points = cart2pol(np.subtract(cart_coords, self._pos))
    def compute_flow(self): ## assumes, for now, unitary b
        for pos in self._points:
            theta = cmg(pos[1], np.array([0., 1.]))
            self.CMGs.append(self._strength*theta)
        return Flow_it(self.CMGs)

    def __repr__(self):
        return repr(self._key + ': a vortex of strength {} at (x, y) = {}'.format(self._strength, self._pos ) )

class doublet(source):
    def __init__(self, key, inputs):
        self._strength = inputs[0]
        self._pos = np.array([inputs[1], inputs[2]])
        self._key = key
        self.CMGs = []
        ##### Changed Here because I did not find self._points ########
        self._points = np.array([])
    def compute_flow(self): ## assumes, for now, unitary b
        for pos in self._points:
            r = cmg(pos[0], np.array([1., 0.]))
            theta = cmg(pos[1], np.array([0., 1.]))
            self.CMGs.append(self._strength*cm.cos(theta)/r)

        return Flow_it(self.CMGs)

    def __repr__(self):
        return repr(self._key + ': a doublet of strength {} at (x, y) = {}'.format(self._strength, self._pos ) )

class tornado(source):
    def __init__(self, key, inputs):
        self._strength = inputs[0]
        self._vorticity = inputs[1]
        self._pos = np.array([inputs[2], inputs[3]])
        self._key = key
        self.CMGs = []
        self._points = np.array([])

    def compute_flow(self): ## assumes, for now, unitary b
        for pos in self._points:
            r = cmg(pos[0], np.array([1., 0.]))
            theta = cmg(pos[1], np.array([0., 1.]))
            self.CMGs.append(self._strength*cm.log(r) + self._vorticity*theta)
        return Flow_it(self.CMGs)

    def __repr__(self):
        return repr(self._key + ': a tornado of strength {}, vorticity {} at (x, y) = {}'.format(self._strength, self._vorticity, self._pos ) )

class whirlpool(tornado):
    def __init__(self, key, inputs):
        self._strength = -inputs[0]
        self._vorticity = inputs[1]
        self._pos = np.array([inputs[2], inputs[3]])
        self._key = key
        self.CMGs = []
        ##### Changed Here because I did not find self._points ########
        self._points = np.array([])

    def __repr__(self):
        return repr(self._key + ': a whirlpool of strength {}, vorticity {} at (x, y) = {}'.format(self._strength, self._vorticity, self._pos ) )

def identify_flow(key_in, inputs):
    library = {
        "uniform": uniform,
        "source": source,
        "sink": sink,
        "vortex": vortex,
        "doublet": doublet,
        "tornado": tornado,
        "whirlpool": whirlpool
    }
    for key in library:
        if key in key_in:
            return (library[key])(key_in, inputs)

def generate_cart_gradients(F, positions_cart):
    gradients_polar = np.array([0, 0])
    phi = np.array([])
    for points in F:
        phi = np.append(phi, points.val)
        gradients_polar = np.vstack((gradients_polar, points.grad))

    gradients_cart = pol2cart_grad(gradients_polar[1:], cart2pol(positions_cart))
    return gradients_cart, phi
