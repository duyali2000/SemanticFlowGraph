import torch

PyTorch_over_1_6 = float('.'.join(torch.__version__.split('.')[0:2])) >= 1.6


class MixedPrecisionManager:
    def __init__(self, activated):
        assert (not activated) or PyTorch_over_1_6, 'Cannot use AMP for PyTorch version < 1.6'

        self.activated = activated

        if self.activated:
            self.scaler = torch.cuda.amp.GradScaler()

    def context(self):
        return torch.cuda.amp.autocast() if self.activated else NullContextManager()

    def backward(self, loss):
        if self.activated:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, colbert, optimizer):
        if self.activated:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)

            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(colbert.parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()


class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource

    def __enter__(self):
        return self.dummy_resource

    def __exit__(self, *args):
        pass


class IndexManager:
    def __init__(self, dim):
        self.dim = dim

    def save(self, tensor, path_prefix):
        torch.save(tensor, path_prefix)


def load_index_part(filename, verbose=True):
    part = torch.load(filename)

    if type(part) == list:  # for backward compatibility
        part = torch.cat(part)

    return part
