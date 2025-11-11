import numpy as np

def compute_margins(X, y, w):
    '''
    Compute the (signed) margins vector m where m_i = y_i * <w, x_i>.
    Inputs:
        X: design matrix of shape (m, d), rows are x_i^T.
        y: labels in {-1, +1} of shape (m,).
        w: weight vector of shape (d,).
    Output:
        margins: vector of shape (m,).
    '''
    
    # element wise multiplication of y with dot product of X and w
    return y * (X @ w)

def compute_s(margins):
    '''
    Compute the indicator vector s where s_i = 1 if margins_i < 1, else 0.
    Input:
        margins: vector of shape (m,).
    Output:
        s: vector in {0,1}^m of shape (m,).
    '''

    return np.array([1 if m < 1 else 0 for m in margins])

def compute_J(X, y, w, C=1.0):
    '''
    Compute the soft-SVM objective:
        J(w) = (1/2)||w||^2 + C * sum_i max(0, 1 - y_i <w, x_i>).
    Inputs:
        X: (m, d)
        y: (m,)
        w: (d,)
        C: positive scalar
    Output:
        J: float scalar
    '''

    return float(0.5 * (w @ w) + C * np.sum(np.maximum(0, 1 - compute_margins(X, y, w))))

def compute_dJ_dw(X, y, w, C=1.0):
    '''
    Compute the gradient of J(w) w.r.t. w using the matrix expression:
        ∇_w J(w) = w - C X^T (Y s),
    where s_i = 1{ y_i <w, x_i> < 1 }.
    Inputs:
        X: (m, d)
        y: (m,)
        w: (d,)
        C: positive scalar
    Output:
        dJ_dw: (d,)
    '''

    return w - C * (X.T @ (y * compute_s(compute_margins(X, y, w))))

def update_w(w, dJ_dw, alpha=0.001):
    '''
    Given the gradient dJ_dw, perform a gradient descent step:
        w <- w - α * dJ_dw.
    Inputs:
        w: current weights, shape (d,)
        dJ_dw: gradient, shape (d,)
        alpha: learning rate, float scalar
    Output:
        w: updated weights, shape (d,)
    Hint: you could solve this problem in 1 line of code.
    '''
    
    return w - alpha * dJ_dw

def train(X, Y, C=1.0, alpha=0.001, n_epoch=100):
    '''
    Train a homogeneous soft-margin linear SVM using full-batch gradient descent.
    We perform n_epoch passes over the training data.
    Inputs:
        X: (m, d) matrix
        Y: (m,) labels in {-1, +1}
        C: positive scalar
        alpha: step size
        n_epoch: number of epochs
    Output:
        w: learned weight vector of shape (d,).
    '''
    
    w = np.zeros(X.shape[1], dtype=float)

    for _ in range(n_epoch):
        dJ_dw = compute_dJ_dw(X, Y, w, C)
        w = update_w(w, dJ_dw, alpha)

    return w

def predict(X, w):
    '''
    Predict labels in {-1, +1} using sign(<w, x>).
    Input:
        X: (m, d)
        w: (d,)
    Output:
        yhat: (m,) in {-1, +1}
    '''

    return np.sign(X @ w)
