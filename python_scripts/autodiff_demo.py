#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt

def my_house(Z):
    ''' This is your own Householder triangularization for QR factorization.'''
    # Initialize matrices
    m, n = Z.shape
    R = Z.copy().astype(float)
    Q = np.eye(m)

    for k in range(n):
        # Step 1: Select the column vector to zero out below diagonal
        x = R[k:,k]

        # Step 2: Computer Householder vector v
        e1 = np.zeros_like(x)
        e1[0] = 1
        v = np.sign(x[0]) * np.linalg.norm(x) * e1 + x
        v /= np.linalg.norm(v)

        # Step 3: Update R matrix iteratively
        R[k:,k:] -= 2 * np.outer(v, v.T @ R[k:,k:])

        # Step 4: Update Q matrix
        Q[:, k:] -= 2 * np.outer(Q[:, k:] @ v, v)

    # Convert full QR to reduced QR
    # Note: Here we implicitly assume that m > n
    # so that the rank of Z is at most n.
    r = np.linalg.matrix_rank(Z)
    Qhat = Q[:, :r]
    Rhat = R[:r, :]

    return Qhat, Rhat



### Part (a) ###
Z = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 7],
              [4, 2, 3],
              [4, 2, 2]])

Qhat, Rhat = my_house(Z)
print('my_house QR factorization:')
print('Qhat:\n', Qhat)
print('Rhat:\n', Rhat)



### Part (b) ###
Qhat, Rhat = np.linalg.qr(Z)
print('NumPy QR factorization:')
print('Qhat:\n', Qhat)
print('Rhat:\n', Rhat)



### Part (d) ###
xs = [-2, -1, 2, 4, 4, 5]
ys = [3, -2, 1, 5, 6, 7]
# Build the Vandermonde matrix A
A = np.vander(xs, N=3,increasing=True)
b = np.array(ys)
print('Vandermonde matrix A:\n', A)
print('Vector b:\n', b)
print('Condition number of A:', np.linalg.cond(A))



### Part (e) ###
# Step i: Compute the reduced QR factorization of A
Qhat, Rhat = my_house(A)
# Step ii: Compute y = Qhat^T b
y = Qhat.T @ b
# Step iii: Solve Rhat x = y for x
x = np.linalg.solve(Rhat, y)
print('Coefficients of the fitted polynomial:', x)

# Plot the data points and the fitted polynomial
fig, ax = plt.subplots(1, 1, figsize=(3, 2), dpi=300)
ax.scatter(xs, ys, color='red', label='Data points')
x_fit = np.linspace(min(xs)-1, max(xs)+1, 100)
y_fit = np.polyval(x[::-1], x_fit)  # evaluate the fitted polynomial
ax.plot(x_fit, y_fit, color='blue', label='Fitted polynomial')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.legend()
plt.show()
