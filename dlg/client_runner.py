import torch

class ClientRunner():
    def __init__(self, net, config):
        self.precision = config.precision
        self.epoch = config.epoch
        self.noise_level = config.noise_level
        self.net = net
    

    def run_client(self, data, label, criterion):
        if self.precision == "float64":
            data = data.double()
            label = label.double()
            self.net = self.net.double() 
        elif self.precision == "float16":
            data = data.half()
            label = label.half()
            self.net = self.net.half()             
        elif self.precision == "float32":
            data = data.float()
            label = label.float()
            self.net = self.net.float()

        pred = self.net(data)
        y = criterion(pred, label)
        dy_dx = torch.autograd.grad(y, self.net.parameters())
        dy_dx = self._add_noise(dy_dx)

        return dy_dx


    def _add_noise(self, dy_dx):
        if self.noise_level != 0:
            noise_ratio = self.noise_level * 0.1
            
            noisy_dy_dx = []
            for g in dy_dx:
                noise_scale = g.mean() * noise_ratio
                noise = torch.randn_like(g) * noise_scale
                noisy_dy_dx.append(g + noise)
                
            dy_dx = tuple(noisy_dy_dx)

        return dy_dx