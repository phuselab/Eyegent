import torch

class RaceModel(object):

    def __init__(self, num_racers, num_walks, DDM_sigma, ndt, dt, device):
        self.num_walks = num_walks
        self.DDM_sigma = DDM_sigma
        self.ndt = ndt
        self.curr_timestamp = 0
        self.dt = dt
        self.num_racers = num_racers
        self.device = device

    def DDM_simulate(self, starting_points, values):
        timestamps = torch.arange(self.curr_timestamp, self.curr_timestamp + self.num_walks, dtype=torch.float32, device=self.device) * self.dt * 1000
        mask = timestamps > self.ndt
        increments = torch.zeros([self.num_racers, self.num_walks], dtype=torch.float64, device=self.device)
        if torch.any(mask):
            values[values < torch.finfo(torch.float32).eps] = -float('inf')
            increments[:, mask] = values[:, None] * self.dt + torch.sqrt(torch.tensor(self.DDM_sigma * self.dt, device=self.device)) * torch.randn([self.num_racers, mask.sum().item()], device=self.device)
        E = torch.cumsum(increments, dim=1) + starting_points[:, None]
        self.curr_timestamp += self.num_walks * self.dt * 1000
        return E
