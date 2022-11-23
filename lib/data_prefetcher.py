import torch

class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            self.next_input, self.next_target, self.next_coarse, self._, self.next_name = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self.next_coarse = None
            self.next_name = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_coarse = self.next_coarse.cuda(non_blocking=True)

            self.next_input = self.next_input.float() #if need
            self.next_target = self.next_target.float() #if need
            self.next_coarse = self.next_coarse.float()
            self.next_name = self.next_name

    def next(self, output_name=False):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        coarse = self.next_coarse
        name = self.next_name
        self.preload()
        if output_name:
            return input, target, coarse, name
        else:
            return input, target, coarse
