import torch
import torch.nn as nn
import math
     
# def get(distance,k):
    # return (2/k*1/distance)**2
import torch.nn
nn.Linear
torch.nn.init.kaiming_uniform_

def gaussian_kaiming_uniform_(tensor,fan,gain=0.113):
    
   
    # print("gain",gain,"beta",beta)
    std = gain / math.sqrt(fan)
    # print(std,"std")
    bound = math.sqrt(3.0) * std  # Calculate the bound for the uniform distribution
    # bound=1
    # print(bound)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class KANBasis(nn.Module):
    def __init__(self, in_features, out_features,num_basis,kernel=...,dtype=None,device=None,ranges=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features=out_features
        self.num_basis = num_basis
        # self.out_features=out_features
        
        self.weights=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.bias=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.kernel=kernel
        beta=self.get_beta(2/num_basis,max( math.log2(num_basis),1))
        self.beta=torch.nn.Parameter(torch.tensor(beta,**factory_kwargs),requires_grad=False)
        # self.range=ranges
        self.reset_parameters()
    @staticmethod
    def get_beta(distance,k):
        return (2/k*1/distance)**2
    def reset_parameters(self):
        fan=self.out_features*self.num_basis
        gaussian_kaiming_uniform_(self.weights,fan)
        torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan) if fan > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)
        # # torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5)/self.num_basis,)
        # gaussian_kaiming_uniform_(self.weights)
        # if ranges is None:
        #     fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     bound=bound 
        #     ranges=[-bound,bound]
        # # print(ranges)
        # # ranges=[-1,1]
        # self.bias.data=torch.linspace(ranges[0],ranges[1],self.num_basis,device=self.bias.device,dtype=self.bias.dtype).unsqueeze(1).repeat(1,self.in_features)
        # # torch.nn.init.uniform_(self.bias,ranges[0],ranges[1])
    def forward(self, x):
        # x=x.unsqueeze(-1)*self.weights+self.bias
        assert x.shape[-1]==self.in_features
        # print(x.shape,self.weights.shape)
        x=torch.einsum('...i,ji->...ji',x,self.weights)+self.bias
        
        if self.kernel is not ...:
            y=self.kernel(x)
        else:
            distance=(x**2)
            y= torch.exp(- distance*self.beta)
        return y
    def extra_repr(self):
        return super().extra_repr() + f'in_features={self.in_features}, num_basis={self.num_basis}'



class KANLayer(nn.Module):
    def __init__(self, in_features,out_features,num_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
        super(KANLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features=in_features
        self.out_features=out_features
        self.num_basis=num_basis
        self.basis=KANBasis(in_features=in_features,out_features=out_features,num_basis=num_basis,kernel=kernel,dtype=dtype,device=device,ranges=ranges).requires_grad_(basis_trainable)
        self.weights=torch.nn.Parameter(torch.empty(out_features,num_basis*in_features,**factory_kwargs))
        self.bias=torch.nn.Parameter(torch.empty(out_features,**factory_kwargs))
        self.mask=torch.nn.Parameter(torch.ones(out_features*in_features,device=device)).requires_grad_(False)
        if base_fn is ...:
            base_fn=torch.cos
        if base_fn is not None:
            self.base_fn=base_fn
            self.scale_base=torch.nn.Parameter(torch.ones(out_features,in_features,**factory_kwargs))
        else:
            self.base_fn=None
        # self.reset_parameters()
        self.reset_parameters()
    def matched_shape_weight(self):
        return self.weights.view(self.out_features,self.num_basis,self.in_features)
    def reset_parameters(self):
        self.basis.reset_parameters()
        fan=self.out_features*self.num_basis
        gaussian_kaiming_uniform_(self.weights,fan)
        
        bound = 1 / math.sqrt(fan) if fan > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)
        

   
    def forward(self, x,return_acts=True,acts_has_grad=None):
        batch_shape=x.shape[:-1]
        assert x.shape[-1]==self.in_features
        
        basis=self.basis(x)
       
        if return_acts:
            
            # grad_context=torch.enable_grad  if acts_has_grad else torch.no_grad
            
            # with grad_context():
            preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
            # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
            postspline=torch.einsum('...ij,kij->...kj',basis,self.matched_shape_weight())
            # print(postspline.shape,self.bias.shape)
            assert postspline.shape==batch_shape+(self.out_features,self.in_features)
            postact=postspline
            
            if self.base_fn is not None:
                base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
                # print(postact.shape,base.shape,self.bias.shape)
                postact=postact+base+(self.bias/self.out_features).unsqueeze(-1)
            
            return postact.sum(dim=-1),preact,postact,postspline

        else:
          
            y=torch.einsum('...ij,kij->...k',basis,self.matched_shape_weight())+self.bias
            if self.base_fn is not None:
                base=torch.nn.functional.linear(self.base_fn(x),self.scale_base)
                y=y+base
            return y,None,None,None
        
        
    def initialize_grid_from_parent(self, parent, x):
        from LBFGS import LBFGS
        x=x.detach()
        with torch.no_grad():
            y=parent(x)[0]
        opt=LBFGS(self.parameters(),lr=0.1,tolerance_change=1e-32,tolerance_grad=1e-32,tolerance_ys=1e-32)
        for i in range(20):
            def closure():
                opt.zero_grad()
                y_pred=self.forward(x)[0]
                loss=torch.nn.functional.mse_loss(y_pred,y)
                loss.backward()
                return loss
            opt.step(closure)
  
        
        
# x=torch.empty(1000,5).uniform_(-100,100)
# model=KANLayer(5,10,20)
# y=model(x)[0]
# from matplotlib import pyplot as plt
# plt.plot(y.detach().numpy())
# plt.show()