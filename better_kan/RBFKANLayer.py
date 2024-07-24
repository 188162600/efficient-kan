from torch import nn
import torch
import math
def linspace_repeative(start, end, steps, k,device=None, dtype=None):
    linspace=torch.linspace(start, end, k,device=device,dtype=dtype)
    linspace=linspace.repeat(steps//k+1)
    return linspace[:steps]
def sin_linspace(min_value, max_value, elements_per_period, num_elements,device=None, dtype=None):
    """
    Generate a tensor with values based on a sine function.

    Args:
        min_value (float): The minimum value of the sine wave.
        max_value (float): The maximum value of the sine wave.
        elements_per_period (int): The number of elements per period of the sine wave.
        num_elements (int): The total number of elements to generate.

    Returns:
        torch.Tensor: A tensor containing the generated values.
    """
    # Calculate the amplitude and offset
    amplitude = (max_value - min_value) / 2
    offset = (max_value + min_value) / 2

    # Generate the linspace for the sine function argument
    x = torch.linspace(0, 2 * math.pi * (num_elements / elements_per_period), num_elements,device=device,dtype=dtype)

    # Apply the sine function and scale/shift the values
    y = amplitude * torch.sin(x) + offset

    return y

class PreactLinear(nn.Module):
   
    def __init__(self, in_features: int, out_features: int, bias: bool = True,full_bias=False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.full_bias=full_bias
        self.weight =torch.nn. Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            if full_bias:
                self.bias =torch.nn. Parameter(torch.empty(out_features,in_features, **factory_kwargs))
            else:
                self.bias =torch.nn. Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn. init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    def forward(self,x):
        assert x.dim()==2
        x=x.unsqueeze(1) # [batch, 1,in_features]
        weight=self.weight.unsqueeze(0) # [1, out_features, in_features]
        if self.bias is not None:
            if self.full_bias:
                bias=self.bias.unsqueeze(0)
            else:
                bias=self.bias.unsqueeze(0).unsqueeze(-1)# [1, out_features, 1]
            return x*weight+bias # [batch, out_features, in_features]
        else:
            return x*weight
    

def pan_algorithm_2_topk(A, k, dim=-1):
    if dim == 0:
        A = A.T
    
    # Step 1: Compute the SVD of A
    # print("svd start",A.shape,"A.shape")
    U, Sigma, Vt = torch.linalg.svd(A, full_matrices=False)
    # print("svd end")
    # Step 2: Select the top k singular values and corresponding singular vectors
    # Uk = U[:, :k]
    # Sigmak = np.diag(Sigma[:k])
    Vk = Vt[:k, :].T
    
    # Step 3: Compute leverage scores
    leverage_scores = torch.sum(Vk**2, axis=1)
    
    # Step 4: Select k columns with the highest leverage scores
    selected_indices = torch.argsort(leverage_scores)[-k:]
    # selected_indices=np.random.a(0,,A.shape[dim])
    # Step 5: Return the selected columns
    C = A[:, selected_indices]
    
    if dim == 0:
        C = C.T
    
    return C, selected_indices
def create_grid(start, end, spacing, dim):
    """
    Create a grid with specified start, end, spacing, and dimension.

    Parameters:
    start (float or list of floats): The starting value(s) for the grid.
    end (float or list of floats): The ending value(s) for the grid.
    spacing (float or list of floats): The spacing between points in the grid.
    dim (int): The dimension of the grid.

    Returns:
    torch.Tensor: The generated grid.
    """
    if isinstance(start, (int, float)):
        start = [start] * dim
    if isinstance(end, (int, float)):
        end = [end] * dim
    if isinstance(spacing, (int, float)):
        spacing = [spacing] * dim

    # Generate ranges for each dimension
    ranges = [torch.arange(start[i], end[i], spacing[i]) for i in range(dim)]

    # Create a meshgrid from the ranges
    grids = torch.meshgrid(*ranges, indexing='ij')

    # Stack the grids to form a grid of points
    grid = torch.stack(grids, dim=-1)
  
    return grid.reshape(-1, dim)
def create_n_grid(start, end, n, dim):
    assert dim>0
    if isinstance(start, (int, float)):
        start = [start] * dim
    if isinstance(end, (int, float)):
        end = [end] * dim
    n_each_dim= math.ceil(n**(1/dim))
    ranges = [torch.linspace(start[i], end[i], n_each_dim) for i in range(dim)]
    grids=torch.meshgrid(*ranges, indexing='ij')
    grid = torch.stack(grids, dim=-1)
    return grid.reshape(-1, dim)[:n]


def rbf_kernel(distance, gamma=1,mode="gaussian"):
    # print(distance.shape)
    # print(gamma)
    # print(distance.shape,"distance2",gamma.shape,"gamma")
    # print(distance.max(),"distance.max()",distance.min(),"distance.min()")
    match mode:
        case "gaussian":
            gamma=torch.clamp(gamma,1e-6)
            return torch.exp(-gamma * distance ** 2)
        case "laplacian":
            return torch.exp(-gamma * distance)
        case "inverse_quadratic":
            return (1/(1+gamma*distance**2))
        case "cauchy":
            
            return 1/(1+gamma*distance)
        case "thin_plate_spline":
            return distance**2*torch.log(distance+1e-6)
        case "multiquadric":
            return torch.sqrt(1+gamma*distance**2)
        
        case "rational_quadratic":
            return 1-1/(1+gamma*distance**2)
        case "inverse_multiquadric":
            return (1/torch.sqrt(1+gamma*distance**2))
        case "linear":
            return gamma*distance
        case "polynomial":
            return (1+gamma*distance)**2
        case "sigmoid":
            return torch.tanh(gamma*distance)
        case "exponential":
            return torch.exp(-gamma*distance)
        case "cosine":
            return torch.cos(gamma*distance)
        case "periodic":
            return torch.exp(-2*gamma*torch.sin(distance/2)**2)
        case "matern":
            return (1+gamma*distance)*torch.exp(-gamma*distance)
        case "compact_support":
            return torch.where(distance<=(1),1-torch.abs(distance),0)
        case "gaussian_derivative":
            return torch.exp(-distance**2)*(1-distance**2)
        case "polyharmonic":
            
            # return bernstein_polynomial(distance,degree=distance.size(1)-1,normalize=True)
            indices=torch.arange(1,distance.size(1)+1,device=distance.device,dtype=distance.dtype)
            log_mask=(indices%2==0)
            log_terms=torch.where(log_mask,torch.log(distance),torch.zeros_like(distance))
            return torch.pow(distance,indices)*log_terms
            
            # results=[]
            # distance=distance.reshape(distance.size(0),-1)
            # for i in range(distance.size(1)):
            #     if i%2==0:
            #         results.append(distance[:,i]**i*torch.log(distance[:,i]))
            #     else:
            #         results.append(distance[:,i]**i)
            # return torch.stack(results,dim=1)
                
        case _:
            raise ValueError("Invalid mode")
        

def normalize(x, target_min, target_max, dim):
    # Find the minimum and maximum values along the specified dimension
    x_min = x.min(dim=dim, keepdim=True)[0]
    x_max = x.max(dim=dim, keepdim=True)[0]
    
    # Prevent division by zero by ensuring that x_max and x_min are not equal
    denominator = x_max - x_min
    denominator[denominator == 0] = 1
    
    # Normalize the tensor to the range [0, 1]
    x_normalized = (x - x_min) / denominator
    
    # Scale and shift the values to the target range [target_min, target_max]
    x_scaled = x_normalized * (target_max - target_min) + target_min
    
    return x_scaled
def rbf_width(distance,k=3):
    width=1/(distance+1e-3)
    width=width*2/k
    width= width**2
    return width
class RBF(nn.Module):
    def __init__(self, in_features, out_features, deg=None,device=None,dtype=None,grid_ep=0.02):
        super(RBF, self).__init__()
        self.grid_ep=grid_ep
        if deg is None:
            deg=2
        factory_kwargs = {'device': device, 'dtype': dtype}
    
        self.in_features = in_features
        self.out_features = out_features
        
        self.deg=deg
        
        self.fc=PreactLinear(in_features, out_features*deg,full_bias=True,**factory_kwargs)
     
        self.centers = nn.Parameter(torch.zeros(in_features,out_features,deg,**factory_kwargs))
        self.centers.data.uniform_(-1, 1)  # Initializing centers uniformly
       
        # self.beta=nn.Parameter(torch.ones(1,**factory_kwargs))
        self.beta = nn.Parameter(torch.ones(in_features, out_features,**factory_kwargs) *100,requires_grad=True) # Initial beta
        self.beta.data=torch.linspace(0.1,10,out_features,**factory_kwargs).unsqueeze(0).repeat(in_features,1)
        # torch.nn.init.uniform_(self.beta,0.1,10)
        # self.beta.data=torch.sort(self.beta,dim=1).values
        
        # self.beta.data=torch.linspace(0.3,3.3,in_features*out_features,**factory_kwargs).reshape(in_features,out_features)
        # self.beta2=nn.Parameter(torch.ones(1,**factory_kwargs) ) # Initial beta
        # torch.nn.init.uniform_(self.beta,0.3,3.3)
        # old_beta=torch.zeros(in_features, out_features,**factory_kwargs)
        # old_beta.data=self.beta.clone()
        # self.register_buffer('old_beta',old_beta)
        
        
    
        

    def forward(self, x):
        # print(self.beta.min(),self.beta.max(),"beta")
        # indices=self.beta<0
        # self.beta.data[indices]=self.old_beta[indices]
        # self.old_beta.data=self.beta.clone()
        # print(self.beta.min(),self.beta.max(),"beta")
        batch=x.size(0)
        # print(self.beta.min(),self.beta.max(),"beta")
        # print(self.in_features,self.out_features,"in out",self.centers.shape,self.fc,"centers,fc")
        x=self.fc(x).reshape(batch,self.out_features,self.deg,self.in_features).permute(0,3,1,2)
        # return x.sum(dim=-1),preact
        
        # x=x*self.weight+self.bais
        # x=self.fc(x).view(-1,self.in_features,self.out_features,self.deg)
        c = self.centers.unsqueeze(0)  # [1, in_features, out_features,deg]
        # print(x.shape,"x.shape",c.shape,"c.shape")
        # print("beta",self.beta)
        distances = ((x  - c)**2).sum(dim=-1).abs()
        # print("beta",self.beta,)
        # print(self.beta.min(),self.beta.max(),"beta")
        # self.beta.data=torch.abs(self.beta)
        # self.beta.data=torch.clamp(self.beta,1e-6)
        return torch.exp(-torch.clamp( self.beta,1e-6)* distances)
    def setup(self, x,proper_min,proper_max,explore_ratio=200,candidate_ratio=20,):
        # return
        # self.centers.data=x[torch.randperm(x.size(0))[:self.out_features*self.deg]].reshape(self.in_features,self.out_features,self.deg)
        # return 
        # print(x.min(),x.max(),"x min,x max")
        centers=create_n_grid(x.min().item(),x.max().item(),self.out_features*self.in_features*explore_ratio,self.deg).to(device=self.centers.device,dtype=self.centers.dtype)
        centers=centers[torch.randperm(centers.size(0))[:self.out_features*self.in_features*candidate_ratio]]
        torch.nn.init.uniform_(centers,-1,1)
        centers=centers.reshape(self.in_features,-1,self.deg)
        centers=torch.cat([centers,self.centers],dim=1)
        centers=centers.unsqueeze(0)
        # print(centers.shape,"centers",x.shape,"x")
        
        distances = ((x.unsqueeze(-1).unsqueeze(-1)  - centers)**2).sum(dim=-1).abs()
        # print(distances.shape,self.beta.shape)
        beta=torch.zeros(self.in_features,centers.size(2),device=self.beta.device,dtype=self.beta.dtype)
        beta_value=(proper_max-proper_min+1e-6)/self.out_features
      
           
        #     torch.nn.init.uniform_( beta[i],10,10)
        beta.fill_(rbf_width(beta_value))
        
        basis=torch.exp(- distances*beta)
       
        basis=basis.reshape(x.size(0)*self.in_features,basis.size(-1))
        
        _,indices=pan_algorithm_2_topk(basis, self.out_features, dim=1)
        # print(indices)
        centers=centers[:,:,indices]
        beta=beta[:,indices]
    
        
        self.beta.data=beta

        
        # centers=torch.gather(centers,1,indices.unsqueeze(-1).repeat(1,1,self.deg))
        self.centers.data=centers.reshape(self.in_features,self.out_features,self.deg)
        # print(self.beta.min(),self.beta.max())
    def setup(self, x,proper_min,proper_max,explore_ratio=200,candidate_ratio=20,):
        
        centers=create_n_grid(x.min().item(),x.max().item(),self.out_features*self.in_features*explore_ratio,self.deg).to(device=self.centers.device,dtype=self.centers.dtype)
        centers=centers[torch.randperm(centers.size(0))[:self.out_features*self.in_features*candidate_ratio]]
        torch.nn.init.uniform_(centers,-1,1)
        centers=centers.reshape(self.in_features,-1,self.deg)
        centers=torch.cat([centers,self.centers],dim=1)
        centers=centers.unsqueeze(0)
        # print(centers.shape,"centers",x.shape,"x")
        
        distances = ((x.unsqueeze(-1).unsqueeze(-1)  - centers)**2).sum(dim=-1).abs()
        # print(distances.shape,self.beta.shape)
        beta=torch.zeros(self.in_features,centers.size(2),device=self.beta.device,dtype=self.beta.dtype)
        beta_value=(proper_max-proper_min+1e-6)/self.out_features
      
           
        #     torch.nn.init.uniform_( beta[i],10,10)
        beta.fill_(rbf_width(beta_value))
        
        basis=torch.exp(- distances*beta)
       
        basis=basis.reshape(x.size(0)*self.in_features,basis.size(-1))
        
        _,indices=pan_algorithm_2_topk(basis, self.out_features, dim=1)
        # print(indices)
        centers=centers[:,:,indices]
        beta=beta[:,indices]
    
        
        self.beta.data=beta

        
        # centers=torch.gather(centers,1,indices.unsqueeze(-1).repeat(1,1,self.deg))
      
        # print(self.beta.min(),self.beta.max())
        
        
        centers_offset=torch.linspace(proper_min.item(),proper_max.item(),self.out_features,device=self.centers.device,dtype=self.centers.dtype)
        centers_offset=centers_offset.unsqueeze(0).unsqueeze(-1).repeat(self.in_features,1,self.deg)
        self.centers.data=centers.reshape(self.in_features,self.out_features,self.deg)+centers_offset
    def setup(self, x,proper_min,proper_max,explore_ratio=200,candidate_ratio=20,):
        
        centers=create_n_grid(proper_min.item(),proper_max.item(),self.out_features*self.in_features*explore_ratio,self.deg).to(device=self.centers.device,dtype=self.centers.dtype)
        centers=centers[torch.randperm(centers.size(0))[:self.out_features*self.in_features*candidate_ratio]]
        # torch.nn.init.uniform_(centers,-1,1)
        centers=centers.reshape(self.in_features,-1,self.deg)
        centers=torch.cat([centers,self.centers],dim=1)
        centers=centers.unsqueeze(0)
        # print(centers.shape,"centers",x.shape,"x")
        
        distances = ((x.unsqueeze(-1).unsqueeze(-1)  - centers)**2).sum(dim=-1).abs()
        # print(distances.shape,self.beta.shape)
        
      
         
        basis=torch.exp(- distances)
       
        basis=basis.reshape(x.size(0)*self.in_features,basis.size(-1))
        
        _,indices=pan_algorithm_2_topk(basis, self.out_features, dim=1)
        # print(indices)
        centers=centers[:,:,indices]
        self.centers.data=centers.reshape(self.in_features,self.out_features,self.deg)
        # self.beta.data=torch.linspace(0.1,proper_max-proper_min+0.1,self.out_features,device=self.beta.device,dtype=self.beta.dtype).unsqueeze(0).repeat(self.in_features,1)
        # return
       
        beta=torch.linspace(0.1,proper_max.item()-proper_min.item()+0.1,candidate_ratio,device=self.beta.device,dtype=self.beta.dtype).reshape(1,1,-1).repeat(1,self.in_features,self.out_features)
        # beta=torch.linspace(1,1,candidate_ratio,device=self.beta.device,dtype=self.beta.dtype).reshape(1,1,-1).repeat(1,self.in_features,self.out_features)
        
        centers=centers.repeat(1,1,candidate_ratio,1)
        distances = ((x.unsqueeze(-1).unsqueeze(-1)  - centers)**2).sum(dim=-1).abs()
        # print(distances.shape,"distrance shape",beta.shape,"beta sha/pe",centers.shape,"centers shape")
        print(beta)
        basis=torch.exp(- distances*beta)
        basis=basis.reshape(x.size(0)*self.in_features,basis.size(-1))
        _,indices=pan_algorithm_2_topk(basis, self.out_features, dim=1)
        # print(indices)
        centers=centers[:,:,indices]
        self.centers.data=centers.reshape(self.in_features,self.out_features,self.deg)
        # print(centers.shape)
        # print(beta.shape)
        # print(beta[:,:,indices].shape,"beta")
        self.beta.data=beta[:,:,indices].reshape(self.in_features,self.out_features)
        
    def setup(self, x,proper_min,proper_max,explore_ratio=200,candidate_ratio=20,):
        
        centers=create_n_grid(proper_min.item(),proper_max.item(),self.out_features*self.in_features*explore_ratio,self.deg).to(device=self.centers.device,dtype=self.centers.dtype)
        centers=centers[torch.randperm(centers.size(0))[:self.out_features*self.in_features*candidate_ratio]]
        # torch.nn.init.uniform_(centers,-1,1)
        centers=centers.reshape(self.in_features,-1,self.deg)
        centers=torch.cat([centers,self.centers],dim=1)
        centers=centers.unsqueeze(0)
        # print(centers.shape,"centers",x.shape,"x")
        
        distances = ((x.unsqueeze(-1).unsqueeze(-1)  - centers)**2).sum(dim=-1).abs()
        # print(distances.shape,self.beta.shape)
        # beta=torch.zeros(self.in_features,centers.size(2),device=self.beta.device,dtype=self.beta.dtype)
        # beta.uniform_(0.1,proper_max-proper_min+0.1)
        
      
         
        basis=torch.exp(- distances)
       
        basis=basis.reshape(x.size(0)*self.in_features,basis.size(-1))
        
        _,indices=pan_algorithm_2_topk(basis, self.out_features, dim=1)
        # print(indices)
        centers=centers[:,:,indices]
        self.centers.data=centers.reshape(self.in_features,self.out_features,self.deg)
       
        # self.beta.data=beta[:,indices].reshape(self.in_features,self.out_features)
        # self.beta.data.uniform_(0.1,proper_max-proper_min+0.1)
        # self.beta.data=torch.linspace(0.1,proper_max-proper_min+0.1,self.out_features,device=self.beta.device,dtype=self.beta.dtype).unsqueeze(0).repeat(self.in_features,1)
        self.beta.data=torch.linspace(0.1,proper_max-proper_min+0.1,self.beta.numel(),device=self.beta.device,dtype=self.beta.dtype).reshape(self.in_features,self.out_features)
    def default_beta(self,proper_min,proper_max):
        beta_value=(proper_max-proper_min+1e-6)/self.out_features
        return rbf_width(beta_value)
    def default_width(self,proper_min,proper_max):
        return (proper_max-proper_min+1e-6)/self.out_features
    @staticmethod
    def get_width_from_sample(x,n):
        x=x.sort(dim=0)[0]
        num_interval=n+1
        ids = [int(x.size(0) / num_interval * i) for i in range(num_interval)] 
        grid=x[ids]
        # print(grid.shape)
        width = (grid[1:] - grid[:-1])
        return width
    def update_grid_from_samples(self, x,proper_min,proper_max):
        # return 
        # x=normalize(x,proper_min,proper_max,dim=0)
        # print(x.min(),x.max(),"x min,x max",proper_min,proper_max,"proper")
        with torch.no_grad():
            y_eval=self.forward(x)
        x=x.sort(dim=0)[0]
        num_interval=self.out_features+1
        ids = [int(x.size(0) / num_interval * i) for i in range(num_interval)] 
        grid=x[ids]
        # print(grid.shape)
        width = (grid[1:] - grid[:-1])
        
        # print(width.shape,self.beta.shape,"width,beta")
        width=width.permute(1,0).contiguous()
        
        self.beta.data=width*(self.grid_ep)+self.default_width(proper_min,proper_max)*(1-self.grid_ep)
        # self.beta.data=rbf_width(width)*(self.grid_ep)+self.beta*(1-self.grid_ep)
        # self.beta.data=rbf_width(width)*(self.grid_ep)+self.default_beta(proper_min,proper_max)*(1-self.grid_ep)
    #     self._fit(x.detach(),y_eval,5,1)
    # def _fit(self,x,y,epochs,lr):
    #     opt=torch.optim.LBFGS([self.centers],lr=lr)
    #     for i in range(epochs):
    #         def closure():
    #             opt.zero_grad()
    #             y_pred=self.forward(x)
    #             loss=torch.nn.functional.mse_loss(y_pred,y)
    #             loss.backward()
    #             return loss
    #         opt.step(closure)
        
        
        
    def get_subset(self,in_id,target):
        target.centers.data = self.centers[in_id].clone()
        target.beta.data = self.beta[in_id].clone()
        target.fc.weight.data = self.fc.weight[:,in_id].clone()
        target.fc.bias.data = self.fc.bias.clone()
        # target.beta2.data = self.beta2.clone()
        return target
    
# Define the RBF Network
class RBFKANLayer(nn.Module):
    def __init__(self, in_features, out_features,num_basis,deg=None,device=None,dtype=None):
        super(RBFKANLayer, self).__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.rbf = RBF(in_features, num_basis, deg=deg,**self.factory_kwargs)
        self.linear = nn.Linear(num_basis, out_features,**self.factory_kwargs,)
        self.deg=deg
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        self.weight=torch.nn.Parameter(torch.ones(in_features,out_features,**self.factory_kwargs))
        self.mask=torch.nn.Parameter(torch.ones(in_features*out_features,**self.factory_kwargs),requires_grad=False)
        self.base_fn=torch.nn.SiLU()
        self.base_weight=torch.nn.Parameter(torch.ones(in_features,out_features,**self.factory_kwargs))
        self.grid_update_num=0
    def proper_mask(self):
        return self.mask.reshape(self.in_features,self.out_features)
    def forward(self, x):
       
        base=x.unsqueeze(-1)*self.base_weight.unsqueeze(0)
        preact=x.unsqueeze(1).repeat(1,self.out_features,1)
        x= self.rbf(x)  # [batch, in_features, num_centers]
        x=x/((x.sum(dim=2,keepdim=True)+1e-6))
        # print(x.shape,"rbf",self.linear,self.linear.weight.shape,self.linear.bias.shape)
        # print(x.shape,"rbf",self.linear(x).shape,"linear",self.weight.shape,"weight",self.proper_mask().shape,"mask")
        x=(self.linear(x))*self.proper_mask()
        postact=x.permute(0,2,1)
        return x.sum(dim=1),preact,postact,postact
    
    def update_grid_from_samples(self, x,proper_min,proper_max):
        
        x=x.detach()
        
        if self.grid_update_num==0:
            batch=x.size(0)
            self.rbf.setup(x,proper_min,proper_max)
            x=create_n_grid(proper_min.item(),proper_max.item(),10000,self.in_features).to(device=x.device,dtype=x.dtype)
            postact=x.unsqueeze(1).repeat(1,self.out_features,1)
            postact=postact.reshape(batch*self.in_features,self.out_features)
            
            basis=self.rbf(x).reshape(batch*self.in_features,self.num_centers)
            # print(basis.shape,postact.shape,"basis,postact")
            # solution=torch.linalg.lstsq(basis,postact)
            # self.linear.weight.data=solution.solution.T
            # self.linear.bias.data=solution.residuals
            # print(solution.solution.shape,"solution",self.linear)
            
            
        else:
            return 
            with torch.no_grad():
                y_eval=self.forward(x)[0]
            
            
            # postact=x.unsqueeze(1).repeat(1,self.out_features,1)
            
            self.rbf.update_grid_from_samples(x,proper_min,proper_max)
            
            # for i in range(3):
            #     opt=torch.optim.LBFGS([*self.linear.parameters()],lr=0.1)
            #     def closure():
            #         opt.zero_grad()
            #         y_pred=self.forward(x)[0]
            #         loss=torch.nn.functional.mse_loss(y_pred,y_eval)
            #         loss.backward()
            #         return loss
            #     opt.step(closure)
        self.grid_update_num+=1
        # with torch.no_grad():
        #     y_eval=self.forward(x)[0]
        # self.rbf.update_grid_from_samples(x)
        # if update_linear:
        #     basis=self.rbf(x).sum(dim=1)
        #     # print(y_eval.shape,basis.shape,"y_eval,basis")
        #     solution=torch.linalg.lstsq(basis,y_eval)
        #     self.linear.weight.data=solution.solution.T
        #     self.linear.bias.data=solution.residuals
      
    
    def get_subset(self, in_id, out_id,target=None):
        '''
        get a smaller KANLayer from a larger KANLayer (used for pruning)
        
        Args:
        -----
            in_id : list
                id of selected input neurons
            out_id : list
                id of selected output neurons
            
        Returns:
        --------
            spb : KANLayer
            
        Example
        -------
        >>> kanlayer_large = KANLayer(in_dim=10, out_dim=10, num=5, k=3)
        >>> kanlayer_small = kanlayer_large.get_subset([0,9],[1,2,3])
        >>> kanlayer_small.in_dim, kanlayer_small.out_dim
        (2, 3)
        '''
        if target is None:
            target= RBFKANLayer(len(in_id), len(out_id),self.num_centers,self.deg,**self.factory_kwargs)
        # print(target.in_features,target.out_features,target.num_centers,"in out num_centers")
        self.rbf.get_subset(in_id,target.rbf)
        # print(target.linear,self.linear.weight[out_id].shape)
        target.linear.weight.data = self.linear.weight[out_id].clone()
        target.linear.bias.data = self.linear.bias[out_id].clone()
        target.weight.data = self.weight[in_id][:,out_id].clone()
        target.proper_mask().data = self.proper_mask()[in_id][:,out_id].clone()
        return target
    def rbf_center_params(self):
        return [self.rbf.centers]
    def log(self):
        print(f"max beta:{self.rbf.beta.max().item()}, min beta:{self.rbf.beta.max().item()},diff{(self.rbf.beta-self.rbf.beta.mean()).abs().mean().item()},{self.rbf.beta}")
        
        
def get_rbf_width(distance,k):
    return (2/k*1/distance)**2
import matplotlib.pyplot as plt
class  RBFKANLayer(nn.Module):
    def __init__(self, in_features, out_features,num_basis,k=3,grid_range=(-1,1),device=None,dtype=None):
        super(RBFKANLayer, self).__init__()
        assert len(grid_range)==2 and grid_range[0]<grid_range[1]
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        self.k=k
        self.grid_range=grid_range
        factory_kwargs = {'device': device, 'dtype': dtype}
        
      
        self.weight_in=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.bias_in=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.widths=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.centers=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.fc_out=nn.Linear(num_basis,out_features)
        self.grid_update_num=0
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_in,a=math.sqrt(5))
        fan_in, _ = torch.nn. init._calculate_fan_in_and_fan_out(self.weight_in)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias_in, -bound, bound)
        
        
        self.widths.data.fill_(get_rbf_width((self.grid_range[1]-self.grid_range[0])/self.num_centers,self.k))
        self.centers=torch.linspace(self.grid_range[0],self.grid_range[1],self.num_centers,device=self.centers.device,dtype=self.centers.dtype).unsqueeze(0).repeat(self.in_features,1)
        self.fc_out.reset_parameters()
    def forward(self, x):
        preact=x.unsqueeze(1).repeat(1,self.out_features,1)
        batch=x.size(0)
        x=x.unsqueeze(1)
        x=x*self.weight_in+self.bias_in
        out=rbf_kernel(x-self.centers,self.widths,mode="gaussian")
        out=out.transpose(1,2)
        assert out.shape==(batch,self.in_features,self.num_centers)
        out=self.fc_out(out)
        out=out.transpose(1,2)
      
        assert preact.shape==out.shape==(batch,self.out_features,self.in_features)
        return out.sum(dim=2),preact,out,out
        
from torch import optim
def normalizer(source_min, source_max, target_min, target_max):
    def normalize(value):
        # Normalize the value to a 0-1 range
        normalized_value = (value - source_min) / (source_max - source_min)
        # Scale to the target range
        return normalized_value * (target_min - target_max) + target_min
    return normalize
def width_of_grid(x, dim):
    # This function assumes x is at least 1D and dim_size in dimension 'dim' is >= 2
    # Get slices for computing width differences centered on each element
    left = x.index_select(dim, torch.arange(0, x.shape[dim] - 1, device=x.device))
    right = x.index_select(dim, torch.arange(1, x.shape[dim], device=x.device))
    
    # Compute half-widths for each pair of adjacent points
    widths = (right - left) 
    
    # Prepare to handle edges by duplicating the boundary widths
    widths_left = torch.cat([widths.narrow(dim, 0, 1), widths], dim=dim)
    widths_right = torch.cat([widths, widths.narrow(dim, -1, 1)], dim=dim)
    
    # Calculate the average width around each point
    width = (widths_left + widths_right) / 2
    
    return width
class  RBFKANLayer(nn.Module):
    def __init__(self, in_features, out_features,num_basis,k=3,grid_range=(-1,1),center_trainable=True,width_trainable=False,grid_ep=0.02,device=None,dtype=None):
        super(RBFKANLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert len(grid_range)==2 and grid_range[0]<grid_range[1]
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        # self.k=torch.nn.Parameter(torch.tensor(float(k),**factory_kwargs),requires_grad=True)
        self.k=k
        self.grid_range=grid_range
        self.grid_ep=grid_ep
      
        
        
      
        self.weight_in=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.bias_in=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs))
        self.widths=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs),requires_grad=width_trainable)
        self.centers=torch.nn.Parameter(torch.zeros(num_basis,in_features,**factory_kwargs),requires_grad=center_trainable)
        self.fc_out=nn.Linear(num_basis,out_features,**factory_kwargs)
        self.mask=torch.nn.Parameter(torch.ones(in_features*out_features,**factory_kwargs),requires_grad=False)
        self.grid_update_num=0
    
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight_in,a=math.sqrt(5))
        fan_in, _ = torch.nn. init._calculate_fan_in_and_fan_out(self.weight_in)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.bias_in, -bound, bound)
        
        
        self.widths.data.fill_((max( (self.grid_range[1]-self.grid_range[0])/self.num_centers,0.1)))
        
        # self.centers.data=torch.linspace(self.grid_range[0],self.grid_range[1],self.num_centers,device=self.centers.device,dtype=self.centers.dtype).unsqueeze(1).repeat(1,self.in_features)
        self.widths.data=width_of_grid(self.centers,dim=0).clamp(0.1)
        # print(width_of_grid(self.centers,dim=0).clamp(0.1))
        # print(self.widths)
        self.fc_out.reset_parameters()
    def default_width(self):
        width= torch.zeros_like(self.widths)
        width.fill_( (max( (self.grid_range[1]-self.grid_range[0])/self.num_centers,0.1)))
        return width
    def default_center(self):
        return torch.linspace(self.grid_range[0],self.grid_range[1],self.num_centers,device=self.centers.device,dtype=self.centers.dtype).unsqueeze(1).repeat(1,self.in_features)
  
        
    # def get_grid
    def rbf_basis(self,x):
        batch=x.size(0)
        assert x.shape==(batch,self.in_features)
        x=x.unsqueeze(1)
        x=x*self.weight_in+self.bias_in
        out=rbf_kernel(x-self.centers,get_rbf_width(self.widths,self.k),mode="gaussian")
        out=out.transpose(1,2)
        
        return out
    def rbf_basis(self,x):
        batch=x.size(0)
        assert x.shape==(batch,self.in_features)
        x=x.unsqueeze(1)
        x=x*self.weight_in+self.bias_in
        out=rbf_kernel(x-self.centers,get_rbf_width(self.widths,self.k),mode="gaussian")
        out=out.transpose(1,2)
        
        return out
        
                
    def forward(self, x):
        batch=x.size(0)
        assert x.shape==(batch,self.in_features)
        preact=x.unsqueeze(1).repeat(1,self.out_features,1)
        out=self.rbf_basis(x)
        
        assert out.shape==(batch,self.in_features,self.num_centers)
      
        out=self.fc_out(out)
        out=out.transpose(1,2)
        
        assert preact.shape==out.shape==(batch,self.out_features,self.in_features)
        return out.sum(dim=2),preact,out,out
    
    def initialize_grid_from_parent(self, parent, x):
        centers=[]
        for i in range(self.in_features):
            centers.append(torch.linspace(x[:,i].min().item(),x[:,i].max().item(),self.num_centers,device=self.centers.device,dtype=self.centers.dtype))
            width=(x[:,i].min().item()- x[:,i].max().item())/self.num_centers
            self.widths.data[:,i]=width
        x=x.detach()
        x=x.sort(dim=0)[0]
        num_interval=self.num_centers
        ids = [int(x.size(0) / num_interval * i) for i in range(num_interval)] 
        grid=x[ids]
        # print(grid.shape)
        width=width_of_grid(grid,dim=0)
        self.widths.data=width
        self.centers.data=grid
        
        return
        with torch.no_grad():
            y_eval=parent.forward(x)[3]
        width_trainable=self.widths.requires_grad
        centers_trainable=self.centers.requires_grad
        self.widths.requires_grad_(True)
        self.centers.requires_grad_(True)
        
        opt=torch.optim.LBFGS(self.parameters(),lr=0.1)
        for i in range(20):
           
            def closure():
                opt.zero_grad()
                y_pred=self.forward(x)[3]
                loss=torch.nn.functional.mse_loss(y_pred,y_eval)
                loss.backward()
                return loss
            opt.step(closure)
        self.widths.requires_grad_(width_trainable)
        self.centers.requires_grad_(centers_trainable)
        
        
        
    def update_grid_from_samples(self, x,range_min,range_max):
        # print("update grid")
        if self.grid_update_num==0:
          
            # self.grid_range=grid_range
            self.reset_parameters()
        else:
            return
            with torch.no_grad():
                y_eval=self.forward(x)[0]
               
            x=x.detach()
            x=x.sort(dim=0)[0]
            num_interval=self.num_centers
            ids = [int(x.size(0) / num_interval * i) for i in range(num_interval)] 
            grid=x[ids]
            
            
            self.centers.data=grid*(self.grid_ep)+self.default_center() *(1-self.grid_ep)
            return 
            # self.widths.data=width_of_grid(grid,dim=0).clamp(0.1)
            rbf_basis=self.rbf_basis(x).sum(dim=1)
        
            solution=torch.linalg. lstsq(rbf_basis,y_eval)
            solution_weight=solution.solution.T
            solution_bias=-solution.residuals
            # print("diff wieght",solution_weight-self.fc_out.weight,"diff",solution_bias-self.fc_out.bias)
            self.fc_out.weight.data= solution_weight
            self.fc_out.bias.data= solution_bias
    
            return 
            # opt=torch.optim.LBFGS([*self.fc_out.parameters(),self.bias_in,self.weight_in],lr=0.1)
            # for i in range(5):
            #     def closure():
            #         opt.zero_grad()
            #         y_pred=self.forward(x)[0]
            #         loss=torch.nn.functional.mse_loss(y_pred,y_eval)
            #         loss.backward()
            #         return loss
            #     opt.step(closure)
            # return 
            rbf_basis=self.rbf_basis(x).sum(dim=1)
        
            # print(rbf_basis.shape,"rbf basis",y_eval.shape,"yeval")
            print(rbf_basis.max(),rbf_basis.min(),"rbf_basis")
            solution=torch.linalg. lstsq(rbf_basis,y_eval)
            scale=3
            solution_weight=solution.solution.T
            solution_bias=solution.residuals
            print("diff wieght",solution_weight-self.fc_out.weight,"diff",solution_bias-self.fc_out.bias)
            self.fc_out.weight.data= solution_weight
            self.fc_out.bias.data= solution_bias
            return
            print((solution_weight<=(self.fc_out.weight*scale)).shape)
            self.fc_out.weight.data=torch.where(((self.fc_out.weight/scale)<=solution_weight)&(solution_weight<=(self.fc_out.weight*scale)),   solution_weight,self.fc_out.weight)
            self.fc_out.bias.data=torch.where(((self.fc_out.bias/scale)<=solution_bias)&(solution_bias<=(self.fc_out.bias*scale)),   solution_bias,self.fc_out.bias)
            
            print(solution.residuals)
            print(solution.solution)
            # self.fc_out.bias.data=solution.residuals    
            print(solution.solution.shape,self.fc_out)
            
        self.grid_update_num+=1
        
class RBF(nn.Module):
    def __init__(self, in_features, out_features,dtype=None,device=None,ranges=None):
        super(RBF, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        
        # self.fc=PreactLinear(in_features, out_features*deg,bias=False)
        # self.weight=
        # if centers is None:
        self.weight=nn.Parameter(torch.zeros(in_features,out_features,**factory_kwargs))
        
        # torch.nn.init.kaiming_uniform_(self.weight.T , a= 2.963)
        gaussian_kaiming_uniform_(self.weight.T)
     
        self.centers = nn.Parameter(torch.zeros(in_features,out_features,**factory_kwargs))
        
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weight.T)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        # bound=bound/(self.out_features)
        if ranges is None:
            ranges=[-bound,bound]
        # print(ranges)
        # torch.nn. init.uniform_(self.centers, ranges[0], ranges[1])
        self.centers.data=torch.linspace(ranges[0],ranges[1],out_features,device=self.centers.device,dtype=self.centers.dtype).unsqueeze(0).repeat(in_features,1)
       
      
      
        beta =(get_rbf_width(2/out_features,math.log2(out_features)))
        # print("beta",beta)
        # self.beta=torch.nn.Parameter(torch.ones(in_features,out_features)*beta)
        self.beta=beta
        # self.beta=1
      
  
    def forward(self, x):
        center=self.centers
        x=x.unsqueeze(-1)*self.weight.unsqueeze(0) -center.unsqueeze(0)
        # print(center.shape,self.centers.shape)
        # print(center.shape,x.shape)
        distance=(x**2)
        y= torch.exp(-self.beta * distance)
        # y=torch.exp(-distance/(x.var()**2))
       
        
        
        # print(y.shape,"y.shape")
        return y
    def get_subset(self,in_id,target):
        target.weight.data=self.weight[in_id].clone()
        target.centers.data=self.centers[in_id].clone()
        target.beta=self.beta
        
def gaussian_kaiming_uniform_(tensor, gain=0.33, mode='fan_in'):
    
    fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    std = gain / math.sqrt(fan)
    # print(std,"std")
    bound = math.sqrt(3.0) * std  # Calculate the bound for the uniform distribution
    # bound=1
    # print(bound)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class RBFKANLayer(nn.Module):
    def __init__(self, in_features,out_features,num_basis,base_fn=...,basis_trainable=False,ranges=None,dtype=None,device=None):
        super(RBFKANLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        self.rbf = RBF(in_features, num_basis,ranges=ranges,**factory_kwargs).requires_grad_(basis_trainable)
        # self.linear = nn.Linear(num_centers, out_features)
        # self.linear2=nn.Linear(in_features*num_centers,out_features)
        # self.in_features=in_features
        # self.out_features=out_features
        # self.num_centers=num_centers
        # self.base_fn=base_fn
        # self.weight=nn.Parameter(torch.ones(in_features,num_centers,out_features))
        self.mask = torch.nn.Parameter(torch.ones(in_features*out_features, device=device)).requires_grad_(False)
        
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight=nn.Parameter(weight.reshape(in_features,num_basis,out_features))    
        self.bias=nn.Parameter(torch.zeros(out_features,**factory_kwargs))
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        if base_fn is ...:
            base_fn=torch.cos
        if base_fn is not None:
            self.base_fn=base_fn
            self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
            torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        else:
            self.base_fn=None
    def proper_mask(self):
        return self.mask.reshape(self.in_features,self.out_features)
        # return self.mask.reshape(self.out_features,self.in_features).T
           
    def reset_parameters(self):
        out_features=self.out_features
        in_features=self.in_features
        num_basis=self.num_centers
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight.data=weight.reshape(in_features,num_basis,out_features)
        # self.bias.data=torch.zeros(out_features,**factory_kwargs)
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        
        self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        
        
    def forward(self, x):
        
        if self.base_fn is not None:
            base=self.base_fn(x).unsqueeze(-1)*self.scale_base.unsqueeze(0)
            preact=x.unsqueeze(1).repeat(1,self.out_features,1)
            y = self.rbf(x)  # [batch, in_features, num_centers]
        
            y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
            postact=(y.sum(dim=2)+base).permute(0,2,1)
        
            return y.sum(dim=(1,2))+self.bias+base.sum(dim=1) ,preact,postact,postact
        else:
            preact=x.unsqueeze(1).repeat(1,self.out_features,1)
            y = self.rbf(x)
            y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
            postact=(y.sum(dim=2)+self.bias).permute(0,2,1)
            return y.sum(dim=(1,2))+self.bias,preact,postact,postact
class RBFKANLayerBias(nn.Module):
    def __init__(self, in_features,out_features,num_basis,base_fn=...,basis_trainable=False,ranges=None,dtype=None,device=None):
        super(RBFKANLayerBias, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features=in_features
        self.out_features=out_features
        self.num_centers=num_basis
        self.rbf = RBF(in_features, num_basis,ranges=ranges,**factory_kwargs).requires_grad_(basis_trainable)
        # self.linear = nn.Linear(num_centers, out_features)
        # self.linear2=nn.Linear(in_features*num_centers,out_features)
        # self.in_features=in_features
        # self.out_features=out_features
        # self.num_centers=num_centers
        # self.base_fn=base_fn
        # self.weight=nn.Parameter(torch.ones(in_features,num_centers,out_features))
        self.mask = torch.nn.Parameter(torch.ones(in_features*out_features, device=device)).requires_grad_(False)
        
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight=nn.Parameter(weight.reshape(in_features,num_basis,out_features))    
        self.bias=nn.Parameter(torch.zeros(in_features,out_features,**factory_kwargs))
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        bound=bound/in_features
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        if base_fn is ...:
            base_fn=torch.cos
        if base_fn is not None:
            self.base_fn=base_fn
            self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
            torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        else:
            self.base_fn=None
    def proper_mask(self):
        return self.mask.reshape(self.in_features,self.out_features)
        # return self.mask.reshape(self.out_features,self.in_features).T
           
    def reset_parameters(self):
        out_features=self.out_features
        in_features=self.in_features
        num_basis=self.num_centers
        factory_kwargs = {'device': self.weight.device, 'dtype': self.weight.dtype}
        weight=torch.zeros(out_features,in_features*num_basis,**factory_kwargs)
        # torch.nn.init.kaiming_uniform_(weight, a=2.963)
        gaussian_kaiming_uniform_(weight)
        
        self.weight.data=weight.reshape(in_features,num_basis,out_features)
        # self.bias.data=torch.zeros(out_features,**factory_kwargs)
    
        
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
        
        self.scale_base=torch.nn.Parameter(torch.ones(in_features,out_features,**factory_kwargs))
        torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        
        
    def forward(self, x):
        
    
        base=self.base_fn(x).unsqueeze(-1)*self.scale_base.unsqueeze(0)
        bias=self.bias
        preact=x.unsqueeze(1).repeat(1,self.out_features,1)
        y = self.rbf(x)  # [batch, in_features, num_centers]
        
        
        y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
        postspline=(y.sum(dim=2)+base)*self.proper_mask()
        postact=postspline+ bias*self.proper_mask()
        postspline=postspline.permute(0,2,1)
        postact=postact.permute(0,2,1)
        return postact.sum(dim=2),preact,postact,postspline
        # return y.sum(dim=(1,2))+bias.sum(dim=0)+base.sum(dim=0),preact,postact,postspline
           
            # return postact.sum(dim=2),preact,postact,postact
            # return y.sum(dim=(1,2))+self.bias+base.sum(dim=1) ,preact,postact,postact
       
    def get_subset(self,in_id,out_id):
        subset=RBFKANLayerBias(len(in_id),len(out_id),self.num_centers,base_fn=self.base_fn,dtype=self.weight.dtype,device=self.weight.device)
        self.rbf.get_subset(in_id,subset.rbf)
        subset.weight.data=self.weight[in_id][:,:,out_id].clone()
        # subset.bias.data=self.bias[out_id].clone()
        subset.bias.data=self.bias[in_id][:,out_id].clone()
        subset.proper_mask().data=self.proper_mask()[in_id][:,out_id].clone()
        # self.mask.data=self.mask.reshape(self.out_features,self.in_features)[out_id][:,in_id].clone().reshape(-1)
        subset.scale_base.data=self.scale_base[in_id][:,out_id].clone()
        return subset
       
    def update_grid_from_samples(self, x,range_min,range_max):
        return 
# RBFKANLayer= RBFKANLayerBias
        # else:
        #     preact=x.unsqueeze(1).repeat(1,self.out_features,1)
        #     y = self.rbf(x)
        #     y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
        #     postact=(y.sum(dim=2)+self.bias).permute(0,2,1)
        #     return y.sum(dim=(1,2))+self.bias,preact,postact,postact
          

    # def forward(self, x):
    #     # print(self.proper_mask())
    #     if self.base_fn is not None:
    #         base=self.base_fn(x).unsqueeze(-1)*self.scale_base.unsqueeze(0)
    #         preact=x.unsqueeze(1).repeat(1,self.out_features,1)
    #         y = self.rbf(x)  # [batch, in_features, num_centers]
        
    #         y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
    #         postact=((y.sum(dim=2)+base)*self.proper_mask()).permute(0,2,1)
    #         return postact.sum(dim=2)+self.bias ,preact,postact,postact
    #         # return y.sum(dim=(1,2))+base.sum(dim=1)+self.bias ,preact,postact,postact
        
    #         return y.sum(dim=(1,2)) ,preact,postact,postact
    #     else:
    #         preact=x.unsqueeze(1).repeat(1,self.out_features,1)
    #         y = self.rbf(x)
    #         y=y.unsqueeze(-1) *self.weight.unsqueeze(0)
    #         y=y*self.proper_mask().unsqueeze(0).unsqueeze(2)
    #         postact=(y.sum(dim=2)).permute(0,2,1)
    #         return y.sum(dim=(1,2)),preact,postact,postact
        
       
 