import random
import copy
from torch.utils.data import Dataset

class BufferDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]
            
class MemoryBuffer:
    def __init__(self, budget):
        self.nof_context = 0
        self.budget = budget 
        self.data_dict = dict()

    def add_data(self, dataset):
        """
        Add new data for a given context.
        Reduce existing memory proportionally, then add new data.
        dataset: PyTorch Dataset object
        """
        new_context_id = self.nof_context
        self.nof_context += 1

        # Reduce existing memory proportionally
        new_size_per_context = self._reduce_buffer()

        # Select new samples for current context
        dataset_size = len(dataset)
        if dataset_size < new_size_per_context:
            selected_indices = list(range(dataset_size))
        else:                                                   
            selected_indices = random.sample(range(dataset_size), new_size_per_context)
        selected_samples = [dataset[idx] for idx in selected_indices]
        self.data_dict[new_context_id] = selected_samples

    def _reduce_buffer(self):
        """
        Reduce the memory buffer to fit within the budget.
        This method is called when adding new data.
        """
        new_size_per_context = self.budget // self.nof_context
        for context_id in self.data_dict:
            current_samples = self.data_dict[context_id]
            if len(current_samples) <= new_size_per_context:
                continue
            self.data_dict[context_id] = random.sample(current_samples, new_size_per_context)

        return new_size_per_context
    
    def get_dataset(self, batch_size=None):
        """
        Get a new dataset that combines all contexts.
        Each context's data is shuffled and concatenated.
        """
        all_samples = []

        for samples in self.data_dict.values():
            all_samples.extend(samples)
        if not all_samples:
            return None 
        if batch_size is not None:
            random.shuffle(all_samples)
            all_samples = all_samples[:batch_size]
        return BufferDataset(all_samples)
                
    def __len__(self):
        return sum(len(samples) for samples in self.data_dict.values())

    def __repr__(self):
        return f"MemoryBuffer(nof_context={self.nof_context}, total_samples={len(self)})"
