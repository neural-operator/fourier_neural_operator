from ..transforms.data_processors import DefaultDataProcessor
from einops import rearrange

import torch

class TheWellDataProcessor(DefaultDataProcessor):
    def __init__(self, normalizer):
        super().__init__()
        self.normalizer = normalizer
    
    def to(self, device):
        self.device = device
        self.normalizer = self.normalizer.to(self.device)
        return self
        
    def preprocess(self, data_dict, step=None):
        """
        Code adapted from the_well.data.data_formatter.DefaultChannelsFirstFormatter
        """

        if step is None or step == 0:
            x = data_dict["input_fields"].to(self.device)
            x = rearrange(x, "b t ... c -> b (t c) ...")
            if "constant_fields" in data_dict:
                flat_constants = rearrange(data_dict["constant_fields"], "b ... c -> b c ...")
                x = torch.cat(
                    [
                        x,
                        flat_constants,
                    ],
                    dim=1,
                )
        else:
            x = data_dict["x"]
        y = data_dict["output_fields"].to(self.device)

        # if stepping, roll y forward
        if step is not None:
            y = y[:, step:step+1, ...]
        y = rearrange(y, "b t ... c -> b (t c) ...")
            # Otherwise x is already preprocessed

        if self.normalizer is not None:
            x = self.normalizer.transform(x)
            y = self.normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict, step=None):
        """
        Code adapted from the_well.data.data_formatter.DefaultChannelsFirstFormatter
        """

        #TODO@DAVID: handle step in autoregressive mode
        y = data_dict["y"]
        if self.normalizer is not None and not self.training:
            y = self.normalizer.inverse_transform(y)
            output = self.normalizer.inverse_transform(output)

        # if in autoregressive mode
        if step is not None:
            data_dict["x"] = output
    
        data_dict["y"] = y
        return output, data_dict
