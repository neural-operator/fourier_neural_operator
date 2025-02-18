from .data_processors import DataProcessor
import torch

class AutoregressiveDataProcessor(DataProcessor):
    """AutoregressiveDataProcessor is a simple processor 
    to pre/post process data for use in the autoregressive trainer. 
    """
    def __init__(
        self, T, timestep=1, in_normalizer=None, out_normalizer=None, debug=True
    ):
        """
        Parameters
        ----------
        T : int
            number of timesteps to stack for input
        timestep : int
            number of timesteps to step forward per step in the rollout, default 1
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        """
        super().__init__()
        self.T = T
        self.timestep = timestep
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.device = "cpu"
        self.model = None
        self.debug = debug

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, step, batched=True):
        """preprocess a batch of data into the format
        expected in model's forward call

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        data_dict : dict
            input data dictionary with one key "u"
        step: int
            timestep of autoregressive prediction
        batched : bool, optional
            whether data contains a batch dim, by default True

        Returns
        -------
        dict
            preprocessed data_dict
        """

        # roll time into channels
        if step == 0:
            x = data_dict["u"][..., :self.T].to(self.device)
        else:
            x = data_dict["x"] # we roll x forward in self.postprocess

        y = data_dict["u"][..., self.T+step:self.T+self.timestep+step].to(self.device)


        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.out_normalizer is not None and self.training:
            y = self.out_normalizer.transform(y)

        if self.debug:
            print(f"post norm, {x.shape=}")


        n_samples_x = x.shape[0]
        data_dict["x_channels"] = x.shape[1]
        spatial_res_x = x.shape[2:-1]
        # reshape n,c, d_1, ... t into n, c*t, d_1, ...
        x = x.permute(0, -1, *list(range(1, x.ndim-1))).view(n_samples_x, -1, *spatial_res_x)

        if self.debug:
            print(f"{x.shape=}")

        n_samples_y = y.shape[0]
        spatial_res_y = y.shape[2:-1]
        data_dict["y_channels"] = y.shape[1]

        # reshape n,c, d_1, ... t into n, c*t, d_1, ...
        y = y.permute(0, -1, *list(range(1, y.ndim-1))).view(n_samples_y, -1, *spatial_res_y)
        if self.debug:
            print(f"{y.shape=}")

        data_dict["x"] = x
        data_dict["y"] = y
        return data_dict

    def postprocess(self, output, data_dict, step):
        """postprocess model outputs and data_dict
        into format expected by training or val loss

        By default, training loss is computed on normalized out and y
        and eval loss is computed on unnormalized out and y

        Parameters
        ----------
        output : torch.Tensor
            raw model outputs
        data_dict : dict
            dictionary containing single batch
            of data

        Returns
        -------
        out, data_dict
            postprocessed outputs and data dict
        """

        # permute back to b,c, x,y,t
        x = data_dict["x"]
        y = data_dict["y"]
        if self.debug:
            print(f"pre permute cat {x.shape=}")
        x = x.view(x.shape[0], data_dict["x_channels"], -1, *x.shape[2:]).permute(0,1, *list(range(3, x.ndim+1)), 2)
        y = y.view(y.shape[0], data_dict["y_channels"], -1, *y.shape[2:]).permute(0,1, *list(range(3, y.ndim+1)), 2)
        output = output.view(output.shape[0], data_dict["y_channels"], -1, *output.shape[2:]).permute(0,1, *list(range(3, output.ndim+1)), 2)

        if self.debug:
            print(f"post permute {x.shape=} {output.shape=}")
        x = torch.cat((x[...,self.timestep:], output), dim=-1)

        if self.out_normalizer and not self.training:
            x = self.in_normalizer.inverse_transform(x)
            output = self.out_normalizer.inverse_transform(output)
    
        data_dict["x"] = x
        data_dict["y"] = y
        if data_dict.get("full_y") is None:
            data_dict["full_y"] = data_dict["y"]
        else:
            data_dict["full_y"] = torch.cat((data_dict["full_y"],data_dict["y"]), dim=-1)
        return output, data_dict

    def forward(self, **data_dict):
        """forward call wraps a model
        to perform preprocessing, forward, and post-
        processing all in one call

        Returns
        -------
        output, data_dict
            postprocessed data for use in loss
        """
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output = self.postprocess(output)
        return output, data_dict