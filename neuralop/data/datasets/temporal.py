from functools import partialmethod
from pathlib import Path
from typing import List, Union, Optional

import torch

from .tensor_dataset import GeneralKeyedTensorDataset
from ..transforms.autoregressive_data_processor import AutoregressiveDataProcessor
from ..transforms.normalizers import UnitGaussianNormalizer

class TemporalDataset:
    """PTDataset is a base Dataset class for our library.
            PTDatasets contain input-output pairs a(x), u(x) and may also
            contain additional information, e.g. function parameters,
            input geometry or output query points.

            datasets may implement a download flag at init, which provides
            access to a number of premade datasets for sample problems provided
            in our Zenodo archive. 

        All datasets are required to expose the following attributes after init:

        train_db: torch.utils.data.Dataset of training examples
        test_db:  ""                       of test examples
        data_processor: neuralop.data.transforms.DataProcessor to process data examples
            optional, default is None
        """
    def __init__(self,
                 root_dir: Union[Path, str],
                 dataset_name: str,
                 n_train: int,
                 n_tests: List[int],
                 batch_size: int,
                 test_batch_sizes: List[int],
                 train_resolution: int,
                 test_resolutions: List[int],
                 T: int, 
                 temporal_resolution: int,
                 timestep: int=1,
                 encoding="channel-wise",
                 input_subsampling_rate=None,
                 output_subsampling_rate=None,
                 channel_dim=1,
                 channels_squeezed=True):
        """PTDataset

        Parameters
        ----------
        root_dir : Union[Path, str]
            root at which to download data files
        dataset_name : str
            prefix of pt data files to store/access
        n_train : int
            number of train instances
        n_tests : List[int]
            number of test instances per test dataset
        batch_size : int
            batch size of training set
        test_batch_sizes : List[int]
            batch size of test sets
        train_resolution : int
            resolution of data for training set
        test_resolutions : List[int], optional
            resolution of data for testing sets, by default [16,32]
        encode_input : bool, optional
            whether to normalize inputs in provided DataProcessor,
            by default False
        encode_output : bool, optional
            whether to normalize outputs in provided DataProcessor,
            by default True
        encoding : str, optional
            parameter for input/output normalization. Whether
            to normalize by channel ("channel-wise") or 
            by pixel ("pixel-wise"), default "channel-wise"
        input_subsampling_rate : int or List[int], optional
            rate at which to subsample each input dimension, by default None
        output_subsampling_rate : int or List[int], optional
            rate at which to subsample each output dimension, by default None
        channel_dim : int, optional
            dimension of saved tensors to index data channels, by default 1
        channels_squeezed : bool, optional
            If the channels dim is 1, whether that is explicitly kept in the saved tensor. 
            If not, we need to unsqueeze it to explicitly have a channel dim. 
            Only applies when there is only one data channel, as in our example problems
            Defaults to True
        """
        
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        self.root_dir = root_dir

        # save dataloader properties for later
        self.batch_size = batch_size
        self.test_resolutions = test_resolutions
        self.test_batch_sizes = test_batch_sizes

        self.T = T
        self.timestep = timestep
            
        # Load train data
        
        data = torch.load(
        Path(root_dir).joinpath(f"{dataset_name}_temporal_{train_resolution}_t{temporal_resolution}_train.pt").as_posix(),
        weights_only=True # no behavior change, suppress warning
        )

        u_train = data["u"].type(torch.float32).clone()
        if channels_squeezed:
            u_train = u_train.unsqueeze(channel_dim)        

        # optionally subsample along data indices
        ## Input subsampling 
        input_data_dims = data["u"].ndim - 3 # batch, channels, time are ignored
        # convert None and 0 to 1
        if not input_subsampling_rate:
            input_subsampling_rate = 1
        if not isinstance(input_subsampling_rate, list):
            # expand subsampling rate along dims if one per dim is not provided
            input_subsampling_rate = [input_subsampling_rate] * (input_data_dims)
        # make sure there is one subsampling rate per data dim
        assert len(input_subsampling_rate) == input_data_dims
        # Construct full indices along which to grab X. No subsampling along time
        train_input_indices = [slice(0, n_train, None)] + [slice(None, None, rate) for rate in input_subsampling_rate] + [slice(None)]
        train_input_indices.insert(channel_dim, slice(None))
        u_train = u_train[train_input_indices]
        
        del data

        # Fit optional encoders to train data
        # Actual encoding happens within DataProcessor
        if encoding == "channel-wise":
            reduce_dims = list(range(u_train.ndim))
            # preserve mean for each channel
            reduce_dims.pop(channel_dim)
            encoder = UnitGaussianNormalizer(dim=reduce_dims)
            encoder.fit(u_train)
        elif encoding == "pixel-wise":
            reduce_dims = [0]
            encoder = UnitGaussianNormalizer(dim=reduce_dims)
        
            encoder.fit(u_train)
        else:
            encoder = None

        # Save train dataset
        self._train_db = GeneralKeyedTensorDataset({'u': u_train})

        self._test_dbs = {}

        # construct test datasets

        for res, n_test in zip(test_resolutions, n_tests):
            data = torch.load(
            Path(root_dir).joinpath(f"{dataset_name}_temporal_{res}_t{temporal_resolution}_test.pt").as_posix(),
            weights_only=True # no behavior change, suppress warning
            )

            u_test = data["u"].type(torch.float32).clone()
            if channels_squeezed:
                u_test = u_test.unsqueeze(channel_dim)        

            # optionally subsample along data indices
            ## Input subsampling 
            input_data_dims = data["u"].ndim - 3 # batch, channels, time are ignored
            # convert None and 0 to 1
            if not input_subsampling_rate:
                input_subsampling_rate = 1
            if not isinstance(input_subsampling_rate, list):
                # expand subsampling rate along dims if one per dim is not provided
                input_subsampling_rate = [input_subsampling_rate] * (input_data_dims)
            # make sure there is one subsampling rate per data dim
            assert len(input_subsampling_rate) == input_data_dims
            # Construct full indices along which to grab X. No subsampling along time
            test_input_indices = [slice(0, n_test, None)] + [slice(None, None, rate) for rate in input_subsampling_rate] + [slice(None)]
            test_input_indices.insert(channel_dim, slice(None))
            u_test = u_test[test_input_indices]
            
            self._test_dbs[res] = GeneralKeyedTensorDataset({'u': u_test})
            del data


        # create DataProcessor
        self._data_processor = AutoregressiveDataProcessor(T=self.T,
                                                           timestep=self.timestep,
                                                           in_normalizer=encoder,
                                                            out_normalizer=encoder)

    @property
    def data_processor(self):
        return self._data_processor
    
    @property
    def train_db(self):
        return self._train_db
    
    @property
    def test_dbs(self):
        return self._test_dbs
    