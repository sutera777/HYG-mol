# HYG-mol/src/data/loader.py
import random
import torch
import traceback
from collections import Counter
from torch_geometric.data import Data, Batch
from .dataset import MoleculeData  

def collate_fn(batch):

    valid_batch = []
    for item in batch:
        try:

            if item is None:
                print("Warning: None item in batch")
                continue

            invalid = False
            for attr in ['x', 'edge_index', 'y']:
                if not hasattr(item, attr) or getattr(item, attr) is None:
                    print(f"Warning: Item missing required attribute '{attr}'")
                    invalid = True
                    break
            if invalid:
                continue

            try:

                new_data = Data(
                    x=item.x.clone(),
                    edge_index=item.edge_index.clone(),
                    y=item.y.clone(),
                )

                if hasattr(item, 'hyperedge_attr') and item.hyperedge_attr is not None:

                    if item.hyperedge_attr.size(1) != 5:
                        resized_attr = torch.zeros((item.hyperedge_attr.size(0), 5), dtype=torch.float32,
                                                   device=item.hyperedge_attr.device)
                        min_dim = min(item.hyperedge_attr.size(1), 5)
                        resized_attr[:, :min_dim] = item.hyperedge_attr[:, :min_dim]
                        new_data.hyperedge_attr = resized_attr
                    else:
                        new_data.hyperedge_attr = item.hyperedge_attr.clone()
                else:

                    num_hyperedges = item.edge_index[1].max().item() + 1 if item.edge_index.numel() > 0 else 1
                    new_data.hyperedge_attr = torch.zeros((num_hyperedges, 5), dtype=torch.float32)
                    if num_hyperedges > 0:
                        new_data.hyperedge_attr = torch.zeros((num_hyperedges, 5), dtype=torch.float32)

                if hasattr(item, 'smiles'):
                    new_data.smiles = item.smiles

                for attr_name in ['batch', 'ptr', 'pos', 'edge_attr']:
                    if hasattr(item, attr_name) and getattr(item, attr_name) is not None:
                        tensor_value = getattr(item, attr_name)
                        if torch.is_tensor(tensor_value):
                            setattr(new_data, attr_name, tensor_value.clone())
                valid_batch.append(new_data)
            except Exception as e:
                print(f"Error creating new Data object: {e}")
                import traceback
                traceback.print_exc()
                continue
        except Exception as e:
            print(f"Error processing batch item: {e}")
            continue


    if not valid_batch:
        print("Warning: Empty batch encountered - all items were invalid.")
        return None

    try:

        y_dims = [item.y.dim() for item in valid_batch]

        if y_dims:
            from collections import Counter
            dim_counter = Counter(y_dims)
            most_common_dim = dim_counter.most_common(1)[0][0]

            for i, item in enumerate(valid_batch):
                if item.y.dim() != most_common_dim:
                    if most_common_dim == 1 and item.y.dim() > 1:
                        valid_batch[i].y = item.y.reshape(-1)
                    elif most_common_dim > 1 and item.y.dim() == 1:
                        valid_batch[i].y = item.y.unsqueeze(-1)
        else:
            print("Warning: No valid y dimensions detected")
            return None

        if most_common_dim > 1:
            y_sizes = [item.y.shape[1] if item.y.dim() > 1 else 1 for item in valid_batch]
            if len(set(y_sizes)) > 1:
                print(f"Inconsistent number of labels detected: {y_sizes}")

                size_counter = Counter(y_sizes)
                most_common_size = size_counter.most_common(1)[0][0]

                items_to_remove = []
                for i, item in enumerate(valid_batch):
                    current_size = item.y.shape[1] if item.y.dim() > 1 else 1
                    if current_size != most_common_size:
                        items_to_remove.append(i)

                for i in sorted(items_to_remove, reverse=True):
                    valid_batch.pop(i)

        if not valid_batch:
            print("Warning: No valid items after dimension adjustment")
            return None

        has_hyperedge_attr = [hasattr(item, 'hyperedge_attr') and item.hyperedge_attr is not None for item in
                              valid_batch]
        if any(has_hyperedge_attr) and not all(has_hyperedge_attr):
            print("Warning: Inconsistent hyperedge attributes in batch (some items have them, some don't)")

        batch = Batch.from_data_list(valid_batch)
        return batch
    except Exception as e:
        print(f"Critical error creating batch: {e}")
        import traceback
        traceback.print_exc()
        return None


class CustomDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.length = len(dataset) if isinstance(dataset, list) else len(dataset)
        self.valid_indices = self._validate_dataset()

        if len(self.valid_indices) == 0:
            raise ValueError("No valid data found in dataset.")
        print(f"Valid samples: {len(self.valid_indices)}/{self.length}")

    def _validate_dataset(self):
        valid_indices = []
        for i in range(self.length):
            try:
                data = self.dataset[i] if isinstance(self.dataset, list) else self.dataset.get(i)
                if data is None:
                    print(f"Warning: Dataset item {i} is None")
                    continue
                if MoleculeData.verify_data(data):
                    valid_indices.append(i)
                else:
                    print(f"Warning: Dataset item {i} failed validation")
            except Exception as e:
                print(f"Error validating data point {i}: {e}")
        return valid_indices

    def __iter__(self):
        indices = self.valid_indices.copy()
        if self.shuffle:
            random.shuffle(indices)

        batch_start = 0
        while batch_start < len(indices):
            batch_indices = indices[batch_start:batch_start + self.batch_size]
            batch_start += self.batch_size

            if not batch_indices:
                continue

            try:

                batch_data = []
                for idx in batch_indices:
                    try:
                        item = self.dataset[idx] if isinstance(self.dataset, list) else self.dataset.get(idx)
                        if item is not None and MoleculeData.verify_data(item):
                            batch_data.append(item)
                        else:
                            print(f"Skipping invalid item at index {idx} during batch creation")
                    except Exception as e:
                        print(f"Error loading item {idx}: {e}")
                        continue

                if not batch_data: 
                    print(f"Warning: Batch starting at index {batch_start - self.batch_size} has no valid items")
                    continue


                batch = collate_fn(batch_data)
                if batch is not None:
                    yield batch
                else:
                    print(f"Skipping invalid batch starting at index {batch_start - self.batch_size}")
            except Exception as e:
                print(f"Error creating batch for indices {batch_indices}: {e}")
                import traceback
                traceback.print_exc()
                continue

    def __len__(self):
        return (len(self.valid_indices) + self.batch_size - 1) // self.batch_size
