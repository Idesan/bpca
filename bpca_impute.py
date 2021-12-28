# -*- coding: utf-8 -*-
"""
Data Imputation using Bayesian Principal Component Analysis (BPCA). 

- Simple model assuming an isotropic latent space
    - `em_pca`:
        Finds principal components for data matrices with no missing entries 
        based on the noiseless probabilistic PCA model proposed by Sam Roweis, 
        "EM algorithms for PCA and SPCA." Advances in neural information 
        processing systems (1998): 626-632. 
    - `impute_em_pca`: 
        Incorporates a functionality of estimating the value of missing entry 
        values into `em_pca`. The missing entries are treated as extra latent 
        variables and are point-estimated. 
- Advanced model that can handle anisotropy over latent dimensions.
    - `impute_bpca`: 
        Learns different variances for the latent dimensions. The variance 
        can be zero if it is irrelevant, thereby the dimensionality of the 
        latent space can be chosen automatically. In such a case, there is no 
        straightforward way of computing log-likelihood. Convergence of the 
        projection matrix $\mathsf{W}$ and imputed values are monitored instead. 
    - `impute_bpca_ard`:
        Uses the same model as above, but this automatically removes 
        irrelevant latent dimensions so the log-likelihood can be monitored 
        (However, there are numerically subtle issues here. The value of the 
         likelihood may not be very accurate in some cases. 
- Other functions
    - `impute_transfer`: 
        Used in a transfer learning setting, where you want to impute a test
        data set based on the PCA model learned on a training data set. 
    - `recover_components_em_pca`:
        Recovers principal components as standard unit orthogonal vectors 
        from $\mathsf{W}$ and $\bm{\mu}$. While the column space of 
        $\mathsf{W}$ spans the principal subspace, the column vectors of it 
        are not the same as the eigenvectors in general. 

Created on Sat Dec 25 17:41:18 2021

@author: Tsuyoshi Ide (tide@us.ibm.com)
"""
def impute_bpca_ard(Xtest_samples_in_columns, n_PCs=None, eps=1.e-4,a_min=1e-4,
                   itr_max = 500, err_L_th = 1.e-4, err_x_th=1.e-4,                   
                   reporting_interval='auto',verbose=True):
    '''
    Data imputation using noiseless Bayesian PCA model with automatic relevance
    determination of the latent dimension.

    Parameters
    ----------
    Xtest_samples_in_columns : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    n_PCs : int, optional
        Dimensionality of the latent space. The default is None.
    eps : float, optional
        The variance of the observation model. The default is 1.e-4.
    a_min : float, optional
        Thereshold blow which alpha_l is thought of as 0. The default is 1e-4.
    itr_max : int, optional
        Max number of iteration. The default is 500.
    err_L_th : float, optional
        Error threshold of log likelihood. The default is 1.e-4.
    err_x_th : float, optional
        Error threshold of missing entries. The default is 1.e-4.
    reporting_interval : int, optional
        If this is 10, summary stats are reporeted every 10 iterations. 
        The default is 'auto'.
    verbose : boolean, optional
        Set False to silence it. The default is True.

    Returns
    -------
    X : 2D numpy array
        Data matrix with filled missing entries.
    pca_param : dict
        - W
            Projection matrix
        - mu
            Mean vector
        - Adiag
            Variance of each latent dimension
        - Z
            Latent variable for each sample in the columns
        - loglik
            Log likelihood for each iteration rounds
        - err_L
            Relative error from the previous round for loglik
        - err_x
            Relative error from the previous round for the missing entries

    '''
    import numpy as np
    
    X = Xtest_samples_in_columns.copy()
    M,N = X.shape # N is the number of samples, M is dimensionality
    
    indices_nan = np.where(np.isnan(X))
    if np.isnan(X).any():
        has_missing_entries = True
        x_filled_old = X[indices_nan].flatten()
    else:
        has_missing_entries = False
        x_filled_old = 0
    
    digits_L = 1 + int(np.abs(np.log10(err_L_th))) # for showing progress
    digits_x = 1 + int(np.abs(np.log10(err_x_th))) # for showing progress

    # Initialization
    if (n_PCs is None) and (N > M):
        n_PCs = M - 1
    elif (n_PCs is None) and (N <= M):
        n_PCs = N - 2
    elif n_PCs is not None: 
        n_PCs = verify_input_empca(Xtest_samples_in_columns,n_PCs)       
        
    W,mu,X = initialize_impute_em_pca(X,n_PCs)
    Adiag = np.diag(W*W)/M
    
    loglik = []
    err_x_list =[]
    err_L_list =[]
    L_old = - np.Inf
    
    if reporting_interval =='auto':
        reporting_interval = int(itr_max/10)

    idx_diag = (np.arange(n_PCs),np.arange(n_PCs))
    print('#samples={}, obs.dim={}, initial latent dim={}'.format(N,M,n_PCs))
    
    for itr in range(itr_max):

        # Solving (W^T W +eps*I) Z = W^T(X - mu1^T)
        WtW = W.T.dot(W)
        WtW[idx_diag] = WtW[idx_diag] + eps
        Z,_,_,_ = np.linalg.lstsq(WtW, W.T.dot(X - mu),rcond=None)
        
        # Solving (AZZ^T + eps*I)W^T = Z (X - mu1^T)^T
        Phi = X - mu 
        ZZt = np.diag(Adiag).dot(Z).dot(Z.T)
        ZZt[idx_diag] = ZZt[idx_diag] + eps
        AZPt = np.diag(Adiag).dot(Z).dot(Phi.T)
        W,_,_,_ = np.linalg.lstsq(ZZt,AZPt,rcond=None)
        W = W.T
        
        # Adjusting mu
        mu = (X - W.dot(Z)).sum(axis=1)/N
        mu = mu.reshape(-1,1)
        
        # Updating Adiag
        Adiag = (W**2).sum(axis=0)/M
        
        # Removing irrelevant latent dimension
        idxs = np.where(Adiag < a_min)[0]
        if len(idxs)>0:
            Adiag = np.delete(Adiag,idxs)
            W = np.delete(W,idxs,axis=1)
            Z = np.delete(Z,idxs,axis=0)
            print('\titr={}:latent space got shrunk from {} '.format(itr,n_PCs),end='')
            n_PCs = n_PCs - len(idxs)
            idx_diag = (np.arange(n_PCs),np.arange(n_PCs))
            print('to {}'.format(n_PCs))
        
        # Re-filling the missing entries
        X[indices_nan] =(W.dot(Z)+ mu)[indices_nan]
        
        # Computing noiseless loglikelihood 
        L = loglik_bpca(X,W,mu,Adiag,Z,eps)          
        loglik.append(L)
        err_L = (L-L_old)/np.abs(L)
        err_L_list.append(err_L)
        L_old = L
        
        # Checking convergence
        if has_missing_entries:
            x_filled = X[indices_nan].flatten()
            x_filled_norm = np.sqrt((x_filled**2).sum())
            err_x = np.sqrt(((x_filled- x_filled_old)**2).sum())/x_filled_norm
            x_filled_old[:] = x_filled[:]
        else:
            err_x = 0 
        err_x_list.append(err_x)

        if (err_L <= err_L_th) and (err_x <= err_x_th):
            break
        elif ((itr+1)%reporting_interval ==0) and verbose:
            print('{:4d}: '.format(itr+1),end='')
            print('err_L={:{dd}.{digits_L}f}, '.\
                  format(err_L, dd=digits_L+2, digits_L=digits_L),end='')
            print('err_x={:{dd}.{digits_x}f}'.\
                  format(err_x, dd=digits_x+2, digits_x=digits_x))
    
    print('Finished ARD_BPCA:itr={}, err_L={}, err_x={}'.format(itr+1,err_L,err_x))
    
    pca_param = {'W':W,'mu':mu,'Adiag':Adiag,'Z':Z,'loglik':np.array(loglik),
                 'err_L':np.array(err_L_list),
                 'err_x':np.array(err_x_list)}
    return X, pca_param


def impute_transfer(Xtest_samples_in_columns,W,mu=None,
                  eps=1.e-4, itr_max=500, err_L_th=1.e-4, 
                  err_x_th=1.e-4, reporting_interval='auto',verbose=True):
    '''
    Data imputation in transfer learning setting

    Parameters
    ----------
    Xtest_samples_in_columns : 2D numpy array
        Data matrix. Samples are treated as a column vectors..
    W : 2D numpy array
        Projection matrix.
    mu : 2D numpy array, optional
        Estimated mean vector. The default is None.
    eps : float, optional
        DESCRIPTION. The default is 1.e-4.
    itr_max : int, optional
        Max number of iteration. The default is 500.
    err_L_th : float, optional
        Error threshold of log likelihood. The default is 1.e-4.
    err_x_th : float, optional
        Error threshold of missing entries. The default is 1.e-4.
    reporting_interval : int, optional
        If this is 10, summary stats are reporeted every 10 iterations. 
        The default is 'auto'.
    verbose : boolean, optional
        Set False to silence it. The default is True.

    Returns
    -------
    X : 2D numpy array
        DESCRIPTION.
    param : dict
        - mu
            Mean vector
        - Z
            Latent variable for each sample in the columns
        - loglik
            Log likelihood for each iteration rounds
        - err_L
            Relative error from the previous round for loglik
        - err_x
            Relative error from the previous round for the missing entries

    '''
    import numpy as np
    
    digits_L = 1 + int(np.abs(np.log10(err_L_th))) # for showing progress
    digits_x = 1 + int(np.abs(np.log10(err_x_th))) # for showing progress
       
    X = Xtest_samples_in_columns.copy()
    M,N = X.shape # N is the number of samples, M is dimensionality
    n_PCs = W.shape[1]
    if mu is None: # We will estimate mu using test data as well
        estimate_mu = True
        mu = np.nanmean(X,axis=1).reshape(-1,1)
    else:
        estimate_mu = False
    
    verify_input_transfer_impute(Xtest_samples_in_columns,W,mu)
    
    indices_nan = np.where(np.isnan(X))
    if np.isnan(X).any():
        has_missing_entries = True
        x_filled_old = X[indices_nan].flatten()
        
        # Fill nan with mu
        X[indices_nan] = np.take(mu, indices_nan[0])    
    else:
        has_missing_entries = False
        x_filled_old = 0
    
    loglik = []
    err_x_list =[]
    err_L_list =[]
    L_old = - np.Inf
    
    if reporting_interval =='auto':
        reporting_interval = int(itr_max/10)

    idx_diag = (np.arange(n_PCs),np.arange(n_PCs))

    for itr in range(itr_max):

        # Solving (W^T W) Z = W^T Phi
        WtW = W.T.dot(W)
        WtW[idx_diag] = WtW[idx_diag] + eps
        Z,_,_,_ = np.linalg.lstsq(WtW, W.T.dot(X - mu),rcond=None)
        
        # Updating mu if estimate_mu is true (i.e., user didn't give mu)
        if estimate_mu:
            mu = (X - W.dot(Z)).sum(axis=1)/N
            mu = mu.reshape(-1,1)
        
        # Re-filling the missing entries
        X[indices_nan] =(W.dot(Z)+ mu)[indices_nan]
        
        # Computing noiseless loglikelihood (actually computes eps*L)
        L = loglik_transfer_impute(X,W,mu,Z,eps)
        loglik.append(L)
        err_L = (L-L_old)/np.abs(L)
        err_L_list.append(err_L)
        L_old = L
        
        # Checking convergence
        if has_missing_entries:
            x_filled = X[indices_nan].flatten()
            x_filled_norm = np.sqrt((x_filled**2).sum())
            err_x = np.sqrt(((x_filled- x_filled_old)**2).sum())/x_filled_norm
            x_filled_old = x_filled
        else:
            err_x = 0        
        err_x_list.append(err_x)

        if (err_L <= err_L_th) and (err_x <= err_x_th):
            break
        elif ((itr+1)%reporting_interval ==0) and verbose:
            print('{:4d}: '.format(itr+1),end='')
            print('err_L={:{dd}.{digits_L}f}, '.\
                  format(err_L, dd=digits_L+2, digits_L=digits_L),end='')
            print('err_x={:{dd}.{digits_x}f}'.\
                  format(err_x, dd=digits_x+2, digits_x=digits_x))
    
    print('Finished EM_PCA transfer itr={},err_L={},err_x={}'.\
          format(itr+1,err_L,err_x))

    param = {'mu':mu,'Z':Z,'loglik':np.array(loglik),
             'err_x':np.array(err_x_list),'err_L':np.array(err_L_list)}
    return X, param
    
    
def em_pca(X_samples_in_columns,n_PCs,itr_max = 100, err_th = 1.e-4,
           reporting_interval='auto', verbose = True):
    '''
    EM (expectation-maximization) PCA (principal component analysis) proposed in
    Sam Roweis, "EM algorithms for PCA and SPCA." Advances in neural 
    information processing systems (1998): 626-632.

    Parameters
    ----------
    X_samples_in_columns : TYPE
        Data matrix. Samples are treated as a column vectors..
    n_PCs : int
        Dimensionality of the latent space. 
    itr_max : int, optional
        Max number of iteration. The default is 500.
    err_th : float, optional
        Error threshold of log likelihood. The default is 1.e-4.
    reporting_interval : int, optional
        If this is 10, summary stats are reporeted every 10 iterations. 
        The default is 'auto'.
    verbose : boolean, optional
        Set False to silence it. The default is True.

    Returns
    -------
    W : 2D numpy array
        Projection matrix.
    mu : 2D numpy array (column vector)
        Mean vector.
    Z : 2D numpy array
        Latent variable for each sample in the columns.
    loglik : 1D numpy array
        Log likelihood for each iteration rounds.

    '''
    import numpy as np
    n_PCs = verify_input_empca(X_samples_in_columns,n_PCs)
    X = X_samples_in_columns.copy()
    M,N = X.shape # N is the number of samples, M is dimensionality

    # Initialization
    mu = X.mean(axis=1).reshape(-1,1)
    Phi = X - mu
    if M < N:
        L = np.linalg.cholesky(Phi.dot(Phi.T)/N)
        W = L[:,0:n_PCs]
    else:
        Q,R= np.linalg.qr(Phi.T)
        W = R.T/np.sqrt(N)
        W = W[:,0:n_PCs]        
    
    loglik = []
    L_old = - np.Inf
    if reporting_interval =='auto':
        reporting_interval = int(itr_max/10)
        
    for itr in range(itr_max):

        # Solving (W^T W) Z = W^T Phi
        Z,_,_,_ = np.linalg.lstsq(W.T.dot(W), W.T.dot(X - mu),rcond=None)
        
        # Solving (ZZ^T)W^T = Z Phi^T
        Phi = X - mu 
        W,_,_,_ = np.linalg.lstsq(Z.dot(Z.T), Z.dot(Phi.T),rcond=None)
        W = W.T
        
        # Adjusting mu
        mu = (X - W.dot(Z)).sum(axis=1)/N
        mu = mu.reshape(-1,1)
        
        # Computing noiseless loglikelihood (actually computes eps*L)
        L = -0.5*((X - mu - W.dot(Z))**2).sum()
        loglik.append(L)
        err = (L-L_old)/np.abs(L)
        L_old = L
        if err <= err_th:
            break
        elif ((itr+1)%reporting_interval ==0) and verbose:
            print('itr:{}, err={} (L={})'.format(itr+1,err,L))
    
    print('EM_PCA final:err={}@itr={}'.format(err,itr+1))
    return W,mu,Z,loglik



def recover_components_em_pca(W,mu,X):
    '''
    Recovers eigenvectors (principal components)

    Parameters
    ----------
    W : 2D numpy array
        Projection matrix.
    mu : 2D numpy array (column vector)
        Mean vector.
    X : 2D numpy array
        Data matrix used to learn W and mu.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    U : 2D numpy array
        Eigenvectors in the columns.

    '''
    import numpy as np
    if np.isnan(X).any():
        raise ValueError('X must not contain any nan')
    verify_input_transfer_impute(X,W,mu)
    Q,R = np.linalg.qr(W)
    C,d,V = np.linalg.svd(Q.T.dot(X-mu),full_matrices=False)
    U = Q.dot(C)
    return U


def shrink_latent_space(W,Adiag,Z=None,a_min=1e-5):
    '''
    For output of impute_bpca, explicitly removes irrelevant dimsions

    Parameters
    ----------
    W : 2D numpy array
        Projection matrix.
    Adiag : 1D numpy array
        Variance of each latent dimension.
    Z : 2D numpy array, optional
        Latent variable for each sample in the columns. The default is None.
    a_min : float, optional
        Thereshold blow which alpha_l is thought of as 0. . The default is 1e-5.

    Returns
    -------
    W0 : 2D numpy array
        Shrunk version of W.
    Adiag0 : 1D numpy array
        Shrunk version of Adiag.
    Z0 : 2D numpy array
        Shrunk version of Z.

    '''
    import numpy as np
    idx = np.where(Adiag < a_min)
    W0 = np.delete(W,idx,axis=1)
    Adiag0 = np.delete(Adiag,idx)
    if Z is not None:
        Z0 = np.delete(Z,idx,axis=0)
    else:
        Z0 = None
    return W0,Adiag0,Z0
    

def initialize_impute_em_pca(X,n_PCs):
    '''
    Internal function for initialization

    Parameters
    ----------
    X : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    n_PCs : int, optional
        Dimensionality of the latent space. The default is None.

    Returns
    -------
    W : 2D numpy array
        Projection matrix intialized.
    mu : 2D numpy array
        Column vector of the means.
    X : 2D numpy array
        Data matrix without nan.

    '''
    import numpy as np
    M,N = X.shape
    # Fill nan with its mean
    mean_row_wise = np.nanmean(X,axis = 1)
    indices_nan = np.where(np.isnan(X))
    X[indices_nan] = np.take(mean_row_wise,indices_nan[0])     
            
    mu = np.nanmean(X,axis=1).reshape(-1,1)
    Phi = X - mu
    Q,R= np.linalg.qr(Phi.T)
    W = R.T/np.sqrt(N)
    W = W[:,0:n_PCs]        
    return W,mu,X        
   

    
def impute_em_pca(Xtest_samples_in_columns, n_PCs, itr_max = 500,
                  err_L_th = 1.e-4, err_x_th=1.e-4,
                  reporting_interval='auto', verbose=True):
    '''
    EM-PCA-based data imputation. 

    Parameters
    ----------
    Xtest_samples_in_columns : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    n_PCs : int, optional
        Dimensionality of the latent space. The default is None.
    itr_max : int, optional
        Max number of iteration. The default is 500.
    err_L_th : float, optional
        Error threshold of log likelihood. The default is 1.e-4.
    err_x_th : float, optional
        Error threshold of missing entries. The default is 1.e-4.
    reporting_interval : int, optional
        If this is 10, summary stats are reporeted every 10 iterations. 
        The default is 'auto'.
    verbose : boolean, optional
        Set False to silence it. The default is True.
        
    Returns
    -------
    X : 2D numpy array
        Data matrix with filled missing entries.
    param : dict
        - W
            Projection matrix
        - mu
            Mean vector
        - Z
            Latent variable for each sample in the columns
        - loglik
            Log likelihood for each iteration rounds
        - err_L
            Relative error from the previous round for loglik
        - err_x
            Relative error from the previous round for the missing entries

    '''
    import numpy as np
    
    n_PCs = verify_input_empca(Xtest_samples_in_columns,n_PCs)
    
    digits_L = 1 + int(np.abs(np.log10(err_L_th))) # for showing progress
    digits_x = 1 + int(np.abs(np.log10(err_x_th))) # for showing progress
    
    X = Xtest_samples_in_columns.copy()
    M,N = X.shape # N is the number of samples, M is dimensionality
    indices_nan = np.where(np.isnan(X))
    if np.isnan(X).any():
        has_missing_entries = True
        x_filled_old = X[indices_nan].flatten()
    else:
        has_missing_entries = False
        x_filled_old = 0

    # Initialization
    W,mu,X = initialize_impute_em_pca(X,n_PCs)
    if reporting_interval =='auto':
        reporting_interval = int(itr_max/10)
    
    loglik = []
    err_x_list =[]
    err_L_list =[]
    L_old = - np.Inf
    for itr in range(itr_max):

        # Solving (W^T W) Z = W^T Phi
        Z,_,_,_ = np.linalg.lstsq(W.T.dot(W), W.T.dot(X - mu),rcond=None)
        
        # Solving (ZZ^T)W^T = Z Phi^T
        Phi = X - mu 
        W,_,_,_ = np.linalg.lstsq(Z.dot(Z.T), Z.dot(Phi.T),rcond=None)
        W = W.T
        
        # Adjusting mu
        mu = (X - W.dot(Z)).sum(axis=1)/N
        mu = mu.reshape(-1,1)
        
        # Re-filling the missing entries
        X[indices_nan] =(W.dot(Z)+ mu)[indices_nan]
        
        # Computing noiseless loglikelihood (actually computes eps*L)
        L = -0.5*((X - mu - W.dot(Z))**2).sum()
        loglik.append(L)
        err_L = (L-L_old)/np.abs(L)
        err_L_list.append(err_L)
        L_old = L
        
        # Checking convergence
        if has_missing_entries:
            x_filled = X[indices_nan].flatten()
            x_filled_norm = np.sqrt((x_filled**2).sum())
            err_x = np.sqrt(((x_filled- x_filled_old)**2).sum())/x_filled_norm
            x_filled_old = x_filled
        else:
            err_x = 0        
        err_x_list.append(err_x)

        if (err_L <= err_L_th) and (err_x <= err_x_th):
            break
        elif ((itr+1)%reporting_interval ==0) and verbose:
            print('{:4d}: '.format(itr+1),end='')
            print('err_L={:{dd}.{digits_L}f}, '.\
                  format(err_L, dd=digits_L+2, digits_L=digits_L),end='')
            print('err_x={:{dd}.{digits_x}f}'.\
                  format(err_x, dd=digits_x+2, digits_x=digits_x))
    
    print('Finished EM_PCA itr={},err_L={},err_x={}'.format(itr+1,err_L,err_x))
    param = {'W':W,'mu':mu,'Z':Z,'loglik':np.array(loglik),
             'err_x':np.array(err_x_list),'err_L':np.array(err_L_list)}
    return X,param


def loglik_bpca(X,W,mu,Adiag,Z,eps):
    '''
    Internal function to compute log likelihood

    Parameters
    ----------
    X : 2D numpy array
        Data matrix without nan.
    W : 2D numpy array
        Projection matrix intialized.
    mu : 2D numpy array
        Column vector of the means.
    Adiag : 1D numpy array
        Array of alpha's, which is the variance of latent dimensions.
    Z : 2D numpy array
        Latent variable as the column vectors.
    eps : flowat
        Observation variance, which is supposed to be close to 0.

    Returns
    -------
    L : flowat
        Log likelihood value.

    '''    
    import numpy as np
    M,N = X.shape
    d = W.shape[1]
    
    L = -(1/(2*eps))*((X - mu - W.dot(Z))**2).sum() \
        -0.5*M*N*np.log(eps) - 0.5*(Z**2).sum()\
        -0.5*(M*N + N*d + M*d)*np.log(2.*np.pi)

    B,_,_,_ = np.linalg.lstsq(np.diag(Adiag), W.T.dot(W),rcond=None)
    L = L - 0.5*np.diag(B).sum() - 0.5*M*np.log(Adiag).sum()  

    return L



def impute_bpca(Xtest_samples_in_columns, n_PCs=None, eps=1.e-4,
                   itr_max = 500, err_W_th = 1.e-4, err_x_th=1.e-4,
                   reporting_interval='auto',verbose=True):
    '''
    Data imputation using noiseless Bayesian PCA

    Parameters
    ----------
    Xtest_samples_in_columns : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    n_PCs : int, optional
        Dimensionality of the latent space. The default is None.
    eps : float, optional
        The variance of observation model. The default is 1.e-4.
    itr_max : int, optional
        Max number of iteration. The default is 500.
    err_W_th : float, optional
        Error threshold of projection matrix W. The default is 1.e-4.
    err_x_th : float, optional
        Error threshold of missing entries. The default is 1.e-4.
    reporting_interval : int, optional
        If this is 10, summary stats are reporeted every 10 iterations. 
        The default is 'auto'.
    verbose : boolean, optional
        Set False to silence it. The default is True.

    Returns
    -------
    X : 2D numpy array
        Data matrix without nan.
    pca_param : dict
        - W
            Projection matrix
        - mu
            Mean vector
        - Adiag
            Variance of each latent dimension
        - Z
            Latent variable for each sample in the columns
        - err_W
            Relative error from the previous round for W
        - err_x
            Relative error from the previous round for the missing entries

    '''
    import numpy as np
    X = Xtest_samples_in_columns.copy()
    M,N = X.shape # N is the number of samples, M is dimensionality
    
    if reporting_interval =='auto':
        reporting_interval = int(itr_max/10)
    
    indices_nan = np.where(np.isnan(X))
    if np.isnan(X).any():
        has_missing_entries = True
        x_filled_old = X[indices_nan].flatten()
    else:
        has_missing_entries = False
        x_filled_old = 0
    
    digits_W = 1 + int(np.abs(np.log10(err_W_th))) # for showing progress
    digits_x = 1 + int(np.abs(np.log10(err_x_th))) # for showing progress

    # Initialization
    if (n_PCs is None) and (N > M):
        n_PCs = M - 1
    elif (n_PCs is None) and (N <= M):
        n_PCs = M - 2
    elif n_PCs is not None: 
        n_PCs = verify_input_empca(Xtest_samples_in_columns,n_PCs)       
        
    W,mu,X = initialize_impute_em_pca(X,n_PCs)
    Adiag = np.diag(W*W)/M
    print('#samples={}, obs.dim={}, latent dim={}'.format(N,M,n_PCs))
    
    #loglik = []
    err_x_list =[]
    err_W_list =[]
    W_old = np.zeros(W.shape)

    idx_diag = (np.arange(n_PCs),np.arange(n_PCs))
    
    for itr in range(itr_max):

        # Solving (W^T W +eps*I) Z = W^T(X - mu1^T)
        WtW = W.T.dot(W)
        WtW[idx_diag] = WtW[idx_diag] + eps
        Z,_,_,_ = np.linalg.lstsq(WtW, W.T.dot(X - mu),rcond=None)
        
        # Solving (AZZ^T + eps*I)W^T = Z (X - mu1^T)^T
        Phi = X - mu 
        ZZt = np.diag(Adiag).dot(Z).dot(Z.T)
        ZZt[idx_diag] = ZZt[idx_diag] + eps
        AZPt = np.diag(Adiag).dot(Z).dot(Phi.T)
        W,_,_,_ = np.linalg.lstsq(ZZt,AZPt,rcond=None)
        W = W.T
        
        # Adjusting mu
        mu = (X - W.dot(Z)).sum(axis=1)/N
        mu = mu.reshape(-1,1)
        
        # Updating Adiag
        Adiag = (W**2).sum(axis=0)/M
        
        # Re-filling the missing entries
        X[indices_nan] =(W.dot(Z)+ mu)[indices_nan]
        
        '''
        We do not compute the log likelihood due to numerical instability.
        
        # Computing noiseless loglikelihood 
        L = loglik_bpca(X,W,mu,Adiag,Z,eps)          
        loglik.append(L)
        err_L = (L-L_old)/np.abs(L)
        err_L_list.append(err_L)
        L_old = L
        '''
        # We instead monitor the difference between W vs W_old
        err_W = (( W.flatten() - W_old.flatten() )**2).sum()
        err_W = np.sqrt( err_W/ (W.flatten()**2).sum())
        err_W_list.append(err_W)
        W_old[:,:] = W[:,:]
        
        # Checking convergence
        if has_missing_entries:
            x_filled = X[indices_nan].flatten()
            x_filled_norm = np.sqrt((x_filled**2).sum())
            err_x = np.sqrt(((x_filled- x_filled_old)**2).sum())/x_filled_norm
            x_filled_old[:] = x_filled[:]
        else:
            err_x = 0 
        err_x_list.append(err_x)

        if (err_W <= err_W_th) and (err_x <= err_x_th):
            break
        elif ((itr+1)%reporting_interval ==0) and verbose:
            print('{:4d}: '.format(itr+1),end='')
            print('err_W={:{dd}.{digits_W}f}, '.\
                  format(err_W, dd=digits_W+2, digits_W=digits_W),end='')
            print('err_x={:{dd}.{digits_x}f}'.\
                  format(err_x, dd=digits_x+2, digits_x=digits_x))
    
    print('Finished: itr={}, err_W={}, err_x={}'.format(itr+1,err_W,err_x))
    
    pca_param = {'W':W,'mu':mu,'Adiag':Adiag,'Z':Z,#'loglik':np.array(loglik),
                 'err_W':np.array(err_W_list),
                 'err_x':np.array(err_x_list)}
    return X, pca_param


def loglik_transfer_impute(X,W,mu,Z,eps):
    '''
    Internal function to compute log likelihood

    Parameters
    ----------
    X : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    W : 2D numpy array
        Projection matrix intialized.
    mu : 2D numpy array
        Column vector of the means.
    Z : 2D numpy array
        Latent variable as the column vectors.
    eps : flowat
        Observation variance, which is supposed to be close to 0.

    Returns
    -------
    L : flowat
        Log likelihood value.

    '''
    import numpy as np
    M,N = X.shape
    d = W.shape[1]
    
    L = -(1/(2*eps))*((X - mu - W.dot(Z))**2).sum() \
        -0.5*M*N*np.log(eps) - 0.5*(Z**2).sum()\
        -0.5*N*(d+M)*np.log(2.*np.pi)
    return L

def verify_input_empca(X,n_PCs):
    '''
    Internal function to verify the input of empca

    Parameters
    ----------
    X : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    n_PCs : int
        Dimensionality of the latent space.

    Returns
    -------
    n_PCs : int
        The dimensionality of principal subspace.

    '''
    M,N = X.shape
    import warnings
    if (M >= N) and (n_PCs >= min(N,M) - 1):
        n_PCs = min(N,M) -2        
        warnings.warn('You have chosen a too large n_PCs value. '+
                      'When M>=N, only n_PC s<= N-2 is supported, where M,N=X.shape. '+
                      'n_PCs is set to {}.'.format(n_PCs))
    elif (M < N) and (n_PCs >= min(N,M)):
        n_PCs = min(N,M) -1
        warnings.warn('You have chosen a too large n_PCs value. '+
                      'When M<N, only n_PC s<= M-1 is supported, where M,N=X.shape. '+
                      'n_PCs is set to {}.'.format(n_PCs))
    return n_PCs


def verify_input_transfer_impute(Xtest_samples_in_columns,W,mu):
    '''
    

    Parameters
    ----------
    Xtest_samples_in_columns : 2D numpy array
        Data matrix. Samples are treated as a column vectors.
    W : 2D numpy array
        Projection matrix intialized.
    mu : 2D numpy array
        Column vector of the means.

    Raises
    ------
    TypeError
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    import numpy as np
    if not isinstance(Xtest_samples_in_columns,np.ndarray):
        raise TypeError('data must be a 2D Numpy array.')
    
    M,N = Xtest_samples_in_columns.shape
    if (len(mu.shape)!=2):
        raise TypeError('mu must be a ({},1)-sized 2D array'.format(M))
    elif (mu.shape[0] != M) :
        raise TypeError('mu and X have inconsistent dimension')
    
    if W.shape[0] != M:
        raise TypeError('W and X have inconsistent sizes')  