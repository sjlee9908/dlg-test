import torch
import torch.nn.functional as F
class DLGRunner():
    def __init__(self, gt_data_size, gt_onehot_label, device, config):
        self.dummy_data = torch.randn(gt_data_size).to(device)
        self.dummy_label = torch.randn(gt_onehot_label.size()).to(device)
        self.true_label = gt_onehot_label
        self.has_label = config.has_label
        self.init_dummy_data = self.dummy_data.detach().clone()

        self.lr = config.lr
        self.device = device
        self.precision = config.precision
        self.iter = config.iter
        self.optim_name = config.optim

    def run_dlg(self, org_dx_dy, net, criterion):
        if self.has_label == True:
            self.dummy_label = self.true_label.detach().clone().to(self.device)

        if self.precision == "float64":
            org_dx_dy = [g.double() for g in org_dx_dy]
            self.dummy_data = self.dummy_data.double()
            self.dummy_label = self.dummy_label.double()
            net = net.double() 
        elif self.precision == "float16":
            org_dx_dy = [g.half() for g in org_dx_dy]
            self.dummy_data = self.dummy_data.half()
            self.dummy_label = self.dummy_label.half()
            net = net.half()
        elif self.precision == "float32":
            org_dx_dy = [g.float() for g in org_dx_dy]
            net = net.float()
        
        self.dummy_data.requires_grad_(True)
        
        if self.has_label:
            self.dummy_label.requires_grad_(False)
            client_params = [self.dummy_data]
        else:
            self.dummy_label.requires_grad_(True)
            client_params = [self.dummy_data, self.dummy_label]

        self.optim = self._get_optim(client_params, self.optim_name)

        for iters in range(self.iter):
            def closure():
                self.optim.zero_grad()

                dummy_pred = net(self.dummy_data) 
                dummy_onehot_label = F.softmax(self.dummy_label, dim=-1)
                dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
                dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
                
                grad_diff = 0
                for gx, gy in zip(dummy_dy_dx, org_dx_dy): 
                    grad_diff += ((gx - gy) ** 2).sum()
                grad_diff.backward()
                
                return grad_diff
            
            self.optim.step(closure)
            if iters % 10 == 0: 
                current_loss = closure()
                print(iters, "%.4f" % current_loss.item())

        return self.init_dummy_data, self.dummy_data

    def _get_optim(self, params, optim):
        if optim == "LBFGS": return torch.optim.LBFGS(params)        
        elif optim == "AdamW":  return torch.optim.AdamW(params, lr=self.lr)        
        elif optim == "Adam": return torch.optim.Adam(params, lr=self.lr)        
        elif optim == "SGD": return torch.optim.SGD(params, lr=self.lr, momentum=0.9)        
        elif optim == "RMSprop": return torch.optim.RMSprop(params, lr=self.lr)        
        elif optim == "Adadelta": return torch.optim.Adadelta(params, lr=self.lr)        
        elif optim == "Adagrad": return torch.optim.Adagrad(params, lr=self.lr)        
        elif optim == "Adamax": return torch.optim.Adamax(params, lr=self.lr)        
        elif optim == "NAdam": return torch.optim.NAdam(params, lr=self.lr)        
        elif optim == "RAdam": return torch.optim.RAdam(params, lr=self.lr)
        else: raise ValueError(f"Unknown optimizer: {optim}")