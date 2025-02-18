from torch.utils.data.dataset import Dataset


class TensorDataset(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        assert (x.size(0) == y.size(0)), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return {'x': x, 'y':y}

    def __len__(self):
        return self.x.size(0)

class GeneralTensorDataset(Dataset):
    def __init__(self, sets, transforms):
        assert len(sets) == len(transforms), "Size mismatch between number of tensors and transforms"
        self.n = len(sets)
        if self.n > 1:
            for j in range(1,self.n):
                assert sets[j].size(0) == sets[0].size(0), "Size mismatch between tensors"
        
        self.sets = sets
        self.transforms = transforms

    def __getitem__(self, index):
        if self.n > 1:
            items = []
            for j in range(self.n):
                items.append(self.sets[j][index])
                if self.transforms[j] is not None:
                    items[j] = self.transforms[j](items[j])
        else:
            items = self.sets[0][index]
            if self.transforms[0] is not None:
                    items = self.transforms[0](items)
        
        return items

    def __len__(self):
        return self.sets[0].size(0)

class GeneralKeyedTensorDataset(Dataset):
    def __init__(self, sets, transforms=None):
        if transforms is not None:
            assert len(sets) == len(transforms), "Size mismatch between number of tensors and transforms"
            assert sets.keys() == transforms.keys(), "Transforms must be keyed to the keys of sets."
        else:
            transforms = {key: None for key in sets.keys()}

        self.n = len(sets)
        size = None
        for key in sets.keys():
            if not size:
                size = sets[key].size(0)
            else:
                assert sets[key].size(0) == size, f"Size mismatch between sets on {key}."
        self.size = size

        self.sets = sets
        self.transforms = transforms

    def __getitem__(self, index):
        sample = {}
        for key, set in self.sets.items():
            sample[key] = set[index]
            if self.transforms[key] is not None:
                sample[key] = self.transforms[key](sample[key])
            
        return sample

    def __len__(self):
        return self.size
