# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 18:42:10 2021

@author: 3G4321897
"""

def show_missing_pattern_binary(df,title_str = 'missing pattern',
                                figsize=(8,8),
                                fontsize_labels=18,fontsize_ticklabels=12):
    '''
    Visualizes the missing pattern of df

    Parameters
    ----------
    df : pandas DataFrame
        Data frame with rows being samples.
    title_str : string, optional
        The default is 'missing pattern'.
    figsize : tuple, optional
        The default is (8,8).
    fontsize_labels : int, optional
        The default is 18.
    fontsize_ticklabels : int, optional
        The default is 12.

    Returns
    -------
    fig : TYPE
        fig object.
    ax : TYPE
        axis object.

    '''
    import matplotlib.pyplot as plt
    import seaborn as sb; sb.set()
    import numpy as np
    
    mask = df.isna()
    tick_location = np.arange(df.shape[1]) + 0.5
    variable_names = df.columns
    
    fig,ax= plt.subplots(1,1,figsize=figsize)        
    ax.pcolor(mask.T,cmap=plt.cm.Blues)
    ax.set_yticks(tick_location, minor=False)
    ax.set_yticklabels(variable_names, minor=False,fontsize=fontsize_ticklabels)
    ax.set_title(title_str,fontsize=fontsize_labels)
    ax.set_xlabel('sample index',fontsize=fontsize_labels)
    fig.tight_layout()
    return fig,ax
    
def compare_missing_patterns(df_raw,df_imputed, figsize=(9,9),
                             fontsize_labels=18,fontsize_ticklabels=12):
    '''
    Visualize how raw data frame has been imputed to be df_imputed.

    Parameters
    ----------
    df_raw : pandas DataFrame
        Data frame with rows being samples. Has missing entries. 
    df_imputed : pandas DataFrame
        Data frame with rows being samples.
    figsize : tuple, optional
        The default is (8,8).
    fontsize_labels : int, optional
        The default is 18.
    fontsize_ticklabels : int, optional
        The default is 12.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        fig object.
    ax : TYPE
        axis object.

    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sb; sb.set()        
    
    df = df_raw.fillna(df_raw.mean(axis=0)).reset_index(drop=True)
    df_imputed = df_imputed.reset_index(drop=True)
    
    if not (df.columns == df_imputed.columns).all():
        raise ValueError('Two data frames have different columns')
            
    Diff = (df - df_imputed).abs()
    
    fig,ax= plt.subplots(1,2,sharey=True,sharex=True,figsize=figsize)
    ax[0].pcolor(df_raw.isna().T,cmap=plt.cm.Blues)
    ax[0].set_yticks(np.arange(df.shape[1]) + 0.5, minor=False)
    ax[0].set_yticklabels(df.columns, minor=False)
    ax[0].set_title("Missing pattern",fontsize=fontsize_labels)
    ax[0].set_xlabel('Sample index',fontsize=fontsize_labels)
    ax[1].pcolor(Diff.T,cmap=plt.cm.Blues)
    ax[1].set_title("|Difference| from mean",fontsize=fontsize_labels)
    ax[1].set_xlabel('Sample index',fontsize=fontsize_labels)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig,ax

def show_missing_pattern(df_filled, mask, figsize=(8,8), 
                         title_string='missing pattern',
                          fontsize_labels=18, fontsize_ticklabels=12):
    '''
    Given missing pattern in the form of boolean mask, show filled values

    Parameters
    ----------
    df_filled : pandas DataFrame
        Imputed data matrix.
    mask : 2D numpy array
        The same dimensionality as df_filled.
    figsize : tuple, optional
        The default is (8,8).
    title_str : string, optional
        The default is 'missing pattern'.
    fontsize_labels : int, optional
        The default is 18.
    fontsize_ticklabels : int, optional
        The default is 12.

    Returns
    -------
    fig : TYPE
        fig object.
    ax : TYPE
        axis object.

    '''
    import matplotlib.pyplot as plt
    import seaborn as sb; sb.set()
    import numpy as np
    
    XX = df_filled.copy()
    XX[mask]=0
    Diff = (df_filled-XX).abs()        
    
    tick_loc = np.arange(df_filled.shape[1]) + 0.5
    varnames = df_filled.columns
    
    fig,ax= plt.subplots(1,1,figsize=figsize)        
    ax.pcolor(Diff.T,cmap=plt.cm.Blues)
    ax.set_yticks(tick_loc, minor=False)
    ax.set_yticklabels(varnames, minor=False,fontsize=fontsize_ticklabels)
    ax.set_title(title_string,fontsize=fontsize_labels)
    ax.set_xlabel('sample index',fontsize=fontsize_labels)
    fig.tight_layout()
    return fig,ax