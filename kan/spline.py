import torch


def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)

    if k == 0:
        value = (x >= grid[:, :-1]) * (x < grid[:, 1:])
    else:
        B_km1 = B_batch(x[:, 0], grid=grid[:, :, 0], k=k - 1, extend=False, device=device)
        value = (x - grid[:, :-(k + 1)]) / (grid[:, k:-1] - grid[:, :-(k + 1)]) * B_km1[:, :-1] + (
                    grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * B_km1[:, 1:]
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>> coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>> coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    # x_eval: (size, batch), grid: (size, grid), coef: (size, coef)
    # coef: (size, coef), B_batch: (size, coef, batch), summer over coef
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    '''
    converting B-spline curves to B-spline coefficients using least squares.
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Example
    -------
    >>> num_spline = 5
    >>> num_sample = 100
    >>> num_grid_interval = 10
    >>> k = 3
    >>> x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar
    mat = B_batch(x_eval, grid, k, device=device).permute(0, 2, 1)
    # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    coef = torch.linalg.lstsq(mat.to(device), y_eval.unsqueeze(dim=2).to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:, :, 0]
    return coef.to(device)
# def lagrange(x, grid, k=0, extend=True):
#     '''
#     evaludate x on B-spline bases
    
#     Args:
#     -----
#         x : 2D torch.tensor
#             inputs, shape (number of splines, number of samples)
#         grid : 2D torch.tensor
#             grids, shape (number of splines, number of grid points)
#         k : int
#             the piecewise polynomial order of splines.
#         extend : bool
#             If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
#         device : str
#             devicde
    
#     Returns:
#     --------
#         spline values : 3D torch.tensor
#             shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.
      
#     Example
#     -------
#     >>> num_spline = 5
#     >>> num_sample = 100
#     >>> num_grid_interval = 10
#     >>> k = 3
#     >>> x = torch.normal(0,1,size=(num_spline, num_sample))
#     >>> grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
#     >>> B_batch(x, grids, k=k).shape
#     torch.Size([5, 13, 100])
#     '''
   
#     # x shape: (size, x); grid shape: (size, grid)
#     def extend_grid(grid, k_extend=0):
#         # pad k to left and right
#         # grid shape: (batch, grid)
#         h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)

#         for i in range(k_extend):
#             grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
#             grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
#         grid = grid
#         return grid

#     if extend == True:
#         grid = extend_grid(grid, k_extend=k)
    
#     indices=torch.arange(0,3).unsqueeze(0)+torch.arange(0,k+1,2).unsqueeze(1)
#     grid=grid[:,indices]
#     # print("x",x.shape,k,grid.shape)
#     x=x.unsqueeze(1)
#     grid=grid.unsqueeze(3) 
#     x_1=grid[:,:,0]
#     x_2=grid[:,:,1]
#     x_3=grid[:,:,2]
#     denom1 = (x_1 - x_2) * (x_1 - x_3)
#     denom2 = (x_2 - x_1) * (x_2 - x_3)
#     denom3 = (x_3 - x_1) * (x_3 - x_2)
#     l1 = (x - x_2) * (x - x_3) / denom1
#     l2 = (x - x_1) * (x - x_3) / denom2
#     l3 = (x - x_1) * (x - x_2) / denom3
#     l=l1+l2+l3
#     mask=(x>=x_1)&(x<=x_3)
#     l=l*mask
 
  
#     return l

# def B_batch(x, grid, k=0, extend=True, device='cpu'):
#     return lagrange(x, grid, k, extend)