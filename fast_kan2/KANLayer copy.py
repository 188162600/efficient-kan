import torch
import torch.nn as nn
import math
     
# def get(distance,k):
    # return (2/k*1/distance)**2
import torch.nn
torch.nn.Linear
class KANBasis(nn.Module):
    def __init__(self, in_features, num_basis,kernel=...,dtype=None,device=None,ranges=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.num_basis = num_basis
        # self.out_features=out_features
        
        self.weights=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.bias=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.kernel=kernel
        beta=self.get_beta(2/num_basis,max( math.log2(num_basis),1))
        self.beta=torch.nn.Parameter(torch.tensor(beta,**factory_kwargs),requires_grad=False)
        # self.range=ranges
        self.reset_parameters(ranges)
    @staticmethod
    def get_beta(distance,k):
        return (2/k*1/distance)**2
    def reset_parameters(self,ranges=None):
        torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5)/self.in_features)
        
        if ranges is None:
            fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            bound=bound/self.in_features   
            ranges=[-bound,bound]
        self.bias.data=torch.linspace(ranges[0],ranges[1],self.num_basis,device=self.bias.device,dtype=self.bias.dtype).unsqueeze(1).repeat(1,self.in_features)
    def forward(self, x):
        # x=x.unsqueeze(-1)*self.weights+self.bias
        assert x.shape[-1]==self.in_features
        # print(x.shape,self.weights.shape)
        x=torch.einsum('...i,ji->...ji',x,self.weights)+self.bias
        
        if self.kernel is not ...:
            y=self.kernel(x)
        else:
            distance=(x**2)
            y= torch.exp(-self.beta * distance)
        return y
    def extra_repr(self):
        return super().extra_repr() + f'in_features={self.in_features}, num_basis={self.num_basis}'
class MultBasis(nn.Module):
    def __init__(self,in_features,num_basis,dtype=None,device=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.num_basis = num_basis
        self.weights=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.bias=nn.Parameter(torch.empty(num_basis,in_features,**factory_kwargs))
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weights)
        self.weights.data[0]=1
        torch.nn.init.ones_(self.bias)
        self.bias.data[0]=0
    def forward(self, x):
        assert x.shape[-1]==self.in_features
        return (torch.einsum('...i,ji->...ji',x,self.weights)+self.bias).prod(dim=-1)
    
# class KANLayer(nn.Module):
#     def __init__(self, in_features,out_features,num_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
#         super(KANLayer, self).__init__()
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.in_features=in_features
#         self.out_features=out_features
#         self.num_basis=num_basis
#         self.basis=KANBasis(in_features,num_basis,out_features,kernel=kernel,dtype=dtype,device=device,ranges=ranges).requires_grad_(basis_trainable)
#         self.weights=torch.nn.Parameter(torch.empty(out_features,num_basis*in_features,**factory_kwargs))
#         self.bias=torch.nn.Parameter(torch.empty(out_features,**factory_kwargs))
#         self.mask=torch.nn.Parameter(torch.ones(out_features*in_features,device=device)).requires_grad_(False)
#         if base_fn is ...:
#             base_fn=torch.cos
#         if base_fn is not None:
#             self.base_fn=base_fn
#             self.scale_base=torch.nn.Parameter(torch.ones(out_features,in_features,**factory_kwargs))
#         else:
#             self.base_fn=None
#     def matched_shape_weight(self):
#         return self.weights.view(self.out_features,self.num_basis,self.in_features)
#     def reset_parameters(self):
       
#         torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5))
#         # self.weights.data=weights.reshape(self.out_features,self.num_basis,self.in_features)
#         torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
#         fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
#     def forward(self, x,return_acts=True,acts_has_grad=None):
#         assert x.shape[-1]==self.in_features
#         basis=self.basis(x)
#         if acts_has_grad is None:
#             acts_has_grad=torch.is_grad_enabled()
#         preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
#                 # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
#         postspline=torch.einsum('...ij,kij->...kj',basis,self.matched_shape_weight())
#         postact=postspline
#         if self.base_fn is not None:
#             base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
#             postact=postact+base
#         return postact.sum(dim=-1)+self.bias,preact,postact,postspline
    
#     def forward(self, x,return_acts=True,acts_has_grad=None):
#         assert x.shape[-1]==self.in_features
#         basis=self.basis(x)
#         if acts_has_grad is None:
#             acts_has_grad=torch.is_grad_enabled()
            
#         if return_acts:
            
#             grad_context=torch.enable_grad  if acts_has_grad else torch.no_grad
            
#             with grad_context():
#                 preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
#                 # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
#                 postspline=torch.einsum('...ij,kij->...kj',basis,self.weights)
#                 postact=postspline+self.bias/self.in_features
#                 if self.base_fn is not None:
#                     base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
#                     postact=postact+base
        
#         else:
#             postact=None
#             preact=None
#             postspline=None
#         y=torch.einsum('...ij,kij->...k',basis,self.weights)+self.bias
#         if self.base_fn is not None:
#             base=torch.nn.functional.linear(self.base_fn(x),self.scale_base)
#             y=y+base
#         return y,preact,postact,postspline
        
        
#     def initialize_grid_from_parent(self, parent, x):
#         from LBFGS import LBFGS
#         x=x.detach()
#         with torch.no_grad():
#             y=parent(x)[0]
#         opt=LBFGS(self.parameters(),lr=0.1,tolerance_change=1e-32,tolerance_grad=1e-32,tolerance_ys=1e-32)
#         for i in range(20):
#             def closure():
#                 opt.zero_grad()
#                 y_pred=self.forward(x)[0]
#                 loss=torch.nn.functional.mse_loss(y_pred,y)
#                 loss.backward()
#                 return loss
#             opt.step(closure)

class KANLayer(nn.Module):
    def __init__(self, in_features,out_features,num_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
        super(KANLayer, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features=in_features
        self.out_features=out_features
        self.num_basis=num_basis
        self.basis=KANBasis(in_features,num_basis,kernel=kernel,dtype=dtype,device=device,ranges=ranges).requires_grad_(basis_trainable)
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
        self.reset_parameters()
    def matched_shape_weight(self):
        return self.weights.view(self.out_features,self.num_basis,self.in_features)
    def reset_parameters(self):
   
        torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5))
        # self.weights.data=weights.reshape(self.out_features,self.num_basis,self.in_features)
        torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
        fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
   
    def forward(self, x,return_acts=True,acts_has_grad=None):
        assert x.shape[-1]==self.in_features
        basis=self.basis(x)
        if acts_has_grad is None:
            acts_has_grad=torch.is_grad_enabled()
            
        if return_acts:
            
            # grad_context=torch.enable_grad  if acts_has_grad else torch.no_grad
            
            # with grad_context():
            preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
            # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
            postspline=torch.einsum('...ij,kij->...kj',basis,self.matched_shape_weight())
            # print(postspline.shape,self.bias.shape)
            postact=postspline+self.bias.unsqueeze(-1)/self.in_features
            if self.base_fn is not None:
                base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
                postact=postact+base
            return postact.sum(dim=-1),preact,postact,postspline

        else:
          
            y=torch.einsum('...ij,kij->...k',basis,self.matched_shape_weight())+self.bias
            if self.base_fn is not None:
                base=torch.nn.functional.linear(self.base_fn(x),self.scale_base)
                y=y+base
            return y,None,None,None
    def forward(self,x):
        basis=self.basis(x)
        y=torch.einsum('...ij,kij->...k',basis,self.matched_shape_weight())+self.bias
        if self.base_fn is not None:
            base=torch.nn.functional.linear(self.base_fn(x),self.scale_base)
            y=y+base
        return y
        
        
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
            
class KANLayer2(nn.Module):
    def __init__(self, in_features,out_features,num_basis,mul_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.mult_basis=MultBasis(in_features,mul_basis,**factory_kwargs)
        self.layer=KANLayer(in_features+mul_basis,out_features,num_basis,base_fn=base_fn,kernel=kernel,basis_trainable=basis_trainable,ranges=ranges,dtype=dtype,device=device)
    def forward(self, x):
        return self.layer(torch.cat([x,self.mult_basis(x)],dim=-1))
if __name__ == '__main__':
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    x=torch.empty(1000,2).uniform_(-1,1).to(device)
    # y=torch.sin(x*2*math.pi)
    y=torch.exp(x[:,0]**2+torch.sin(x[:,0]*2*math.pi)).unsqueeze(-1)
    model=torch.nn.Sequential(
        KANLayer2(2,1,10,10),
        KANLayer2(1,1,10,10),
    ).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=0.01)
    for i in range(1000):
        opt.zero_grad()
        y_pred=model(x)
        
        loss=torch.nn.functional.mse_loss(y_pred,y)
        loss.backward()
        opt.step()
        print(loss)
        
# class KANLayer(nn.Module):
#     def __init__(self, in_features,out_features,num_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
#         super(KANLayer, self).__init__()
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.in_features=in_features
#         self.out_features=out_features
#         self.num_basis=num_basis
#         self.basis=KANBasis(in_features,num_basis,kernel=kernel,dtype=dtype,device=device,ranges=ranges).requires_grad_(basis_trainable)
#         self.weights=torch.nn.Parameter(torch.empty(out_features,num_basis,in_features,**factory_kwargs))
#         self.bias=torch.nn.Parameter(torch.empty(out_features,**factory_kwargs))
#         self.mask=torch.nn.Parameter(torch.ones(out_features*in_features,device=device)).requires_grad_(False)
#         if base_fn is ...:
#             base_fn=torch.cos
#         if base_fn is not None:
#             self.base_fn=base_fn
#             self.scale_base=torch.nn.Parameter(torch.ones(out_features,in_features,**factory_kwargs))
#         else:
#             self.base_fn=None
       
#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5))
#         torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
#         fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
#     def forward(self, x,return_acts=True,acts_has_grad=None):
#         assert x.shape[-1]==self.in_features
#         basis=self.basis(x)
#         if acts_has_grad is None:
#             acts_has_grad=torch.is_grad_enabled()
#         preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
#                 # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
#         postspline=torch.einsum('...ij,kij->...kj',basis,self.weights)
#         postact=postspline
#         if self.base_fn is not None:
#             base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
#             postact=postact+base
#         return postact.sum(dim=-1)+self.bias,preact,postact,postspline
    
 
        
        
#     def initialize_grid_from_parent(self, parent, x):
#         from LBFGS import LBFGS
#         x=x.detach()
#         with torch.no_grad():
#             y=parent(x)[0]
#         opt=LBFGS(self.parameters(),lr=0.1,tolerance_change=1e-32,tolerance_grad=1e-32,tolerance_ys=1e-32)
#         for i in range(20):
#             def closure():
#                 opt.zero_grad()
#                 y_pred=self.forward(x)[0]
#                 loss=torch.nn.functional.mse_loss(y_pred,y)
#                 loss.backward()
#                 return loss
#             opt.step(closure)

# if __name__ == '__main__':
#     torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
#     def test_shape(in_features,out_features,num_basis,batch_size):
#         x=torch.randn(batch_size,in_features)
#         kan=KANLayer(in_features,out_features,num_basis)
#         y,preact,postact,postspline=kan(x,return_acts=True,acts_has_grad=True)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact.shape==(batch_size,out_features,in_features)
#         assert postact.shape==(batch_size,out_features,in_features)
#         assert postspline.shape==(batch_size,out_features,in_features)
#         y,preact,postact,postspline=kan(x,return_acts=True,acts_has_grad=False)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact.shape==(batch_size,out_features,in_features)
#         assert postact.shape==(batch_size,out_features,in_features)
#         assert postspline.shape==(batch_size,out_features,in_features)
#         y,preact,postact,postspline=kan(x,return_acts=False,acts_has_grad=False)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact is None
#         assert postact is None
#         assert postspline is None
#     def test_accuracy(batch_size,num_basis):
        
        
#         x=torch.empty(batch_size,1).uniform_(-1,1)
#         y=x**2
#         torch.manual_seed(0)
#         kan=KANLayer(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
        
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
#         torch.manual_seed(0)
#         kan=KANLayer2(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
        
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
        
#         from RBFKANLayer import RBFKANLayer 
#         torch.manual_seed(0)
#         kan=RBFKANLayer(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
#     # test_shape(3,4,5,100)
#     test_accuracy(1000,10)
    

    

# class KANLayer(nn.Module):
#     def __init__(self, in_features,out_features,num_basis,base_fn=...,kernel=...,basis_trainable=True,ranges=None,dtype=None,device=None):
#         super(KANLayer, self).__init__()
        
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.in_features=in_features
#         self.out_features=out_features
#         self.num_basis=num_basis
#         self.basis=KANBasis(in_features,num_basis,kernel=kernel,dtype=dtype,device=device,ranges=ranges).requires_grad_(basis_trainable)
#         self.weights=torch.nn.Parameter(torch.empty(out_features,num_basis,in_features,**factory_kwargs))
#         self.bias=torch.nn.Parameter(torch.empty(out_features,**factory_kwargs))
#         self.mask=torch.nn.Parameter(torch.ones(out_features*in_features,device=device)).requires_grad_(False)
#         if base_fn is ...:
#             base_fn=torch.cos
#         if base_fn is not None:
#             self.base_fn=base_fn
#             self.scale_base=torch.nn.Parameter(torch.ones(out_features,in_features,**factory_kwargs))
#         else:
#             self.base_fn=None
       
#     def reset_parameters(self):
#         torch.nn.init.kaiming_uniform_(self.weights,a=math.sqrt(5))
#         torch.nn.init.kaiming_uniform_(self.scale_base,a=math.sqrt(5))
#         fan_in, _ =torch.nn. init._calculate_fan_in_and_fan_out(self.weights)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         torch.nn. init.uniform_(self.bias, -bound, bound)       # self.base_linear=PreactLinear(in_features,out_features)
#     def forward(self, x,return_acts=True,acts_has_grad=None):
#         assert x.shape[-1]==self.in_features
#         basis=self.basis(x)
#         if acts_has_grad is None:
#             acts_has_grad=torch.is_grad_enabled()
#         preact=x.unsqueeze(-2).expand(*x.shape[:-1],self.out_features,self.in_features)
#                 # postspline=(basis.unsqueeze(-3)*self.weights).sum(dim=-1)
#         postspline=torch.einsum('...ij,kij->...kj',basis,self.weights)
#         postact=postspline
#         if self.base_fn is not None:
#             base=torch.einsum("...i,ji->...ji",self.base_fn(x),self.scale_base) 
#             postact=postact+base
#         return postact.sum(dim=-1)+self.bias,preact,postact,postspline
    
 
        
        
#     def initialize_grid_from_parent(self, parent, x):
#         from LBFGS import LBFGS
#         x=x.detach()
#         with torch.no_grad():
#             y=parent(x)[0]
#         opt=LBFGS(self.parameters(),lr=0.1,tolerance_change=1e-32,tolerance_grad=1e-32,tolerance_ys=1e-32)
#         for i in range(20):
#             def closure():
#                 opt.zero_grad()
#                 y_pred=self.forward(x)[0]
#                 loss=torch.nn.functional.mse_loss(y_pred,y)
#                 loss.backward()
#                 return loss
#             opt.step(closure)

# if __name__ == '__main__':
#     torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
#     def test_shape(in_features,out_features,num_basis,batch_size):
#         x=torch.randn(batch_size,in_features)
#         kan=KANLayer(in_features,out_features,num_basis)
#         y,preact,postact,postspline=kan(x,return_acts=True,acts_has_grad=True)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact.shape==(batch_size,out_features,in_features)
#         assert postact.shape==(batch_size,out_features,in_features)
#         assert postspline.shape==(batch_size,out_features,in_features)
#         y,preact,postact,postspline=kan(x,return_acts=True,acts_has_grad=False)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact.shape==(batch_size,out_features,in_features)
#         assert postact.shape==(batch_size,out_features,in_features)
#         assert postspline.shape==(batch_size,out_features,in_features)
#         y,preact,postact,postspline=kan(x,return_acts=False,acts_has_grad=False)
#         print(y.shape)
#         assert y.shape==(batch_size,out_features)
#         assert preact is None
#         assert postact is None
#         assert postspline is None
#     def test_accuracy(batch_size,num_basis):
        
        
#         x=torch.empty(batch_size,1).uniform_(-1,1)
#         y=x**2
#         torch.manual_seed(0)
#         kan=KANLayer(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
        
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
#         torch.manual_seed(0)
#         kan=KANLayer2(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
        
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
        
#         from RBFKANLayer import RBFKANLayer 
#         torch.manual_seed(0)
#         kan=RBFKANLayer(1,1,num_basis)
#         opt=torch.optim.Adam(kan.parameters(),lr=0.01)
#         for i in range(1000):
#             y_pred,_,_,_=kan(x)
#             loss=torch.nn.functional.mse_loss(y_pred,y)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#         print (loss)
#     # test_shape(3,4,5,100)
#     test_accuracy(1000,10)
    

    
