from torch.utils.data import Dataset, Subset, ConcatDataset
import random
from typing import Union, Optional
import numpy as np

def remove_class_samples(dataset: Union[Dataset, Subset, ConcatDataset], 
                        target_class: int, 
                        removal_ratio: float,
                        random_seed: Optional[int] = None) -> Union[Subset, ConcatDataset]:
    """
    Remove a specified ratio of samples from a target class in a PyTorch dataset.
    
    Args:
        dataset: The input dataset (Dataset, Subset, or ConcatDataset)
        target_class: The class label from which to remove samples
        removal_ratio: Ratio of samples to remove (0.0 to 1.0)
        random_seed: Optional random seed for reproducible results
    
    Returns:
        A new dataset with the specified samples removed
    """
    if not 0.0 <= removal_ratio <= 1.0:
        raise ValueError("removal_ratio must be between 0.0 and 1.0")
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Handle different dataset types
    if isinstance(dataset, ConcatDataset):
        return _remove_from_concat_dataset_fast(dataset, target_class, removal_ratio)
    elif isinstance(dataset, Subset):
        return _remove_from_subset_fast(dataset, target_class, removal_ratio)
    else:
        # Regular Dataset
        return _remove_from_dataset_fast(dataset, target_class, removal_ratio)

def _remove_from_dataset_fast(dataset: Dataset, target_class: int, removal_ratio: float) -> Subset:
    """Fast removal from a regular Dataset using single pass."""
    target_indices = []
    keep_indices = []
    
    # Single pass through dataset
    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == target_class:
            target_indices.append(i)
        else:
            keep_indices.append(i)
    
    # Fast numpy-based sampling
    target_indices = np.array(target_indices)
    num_to_remove = int(len(target_indices) * removal_ratio)
    
    if num_to_remove > 0:
        # Use numpy for faster random sampling
        remove_mask = np.random.choice(len(target_indices), size=num_to_remove, replace=False)
        keep_mask = np.ones(len(target_indices), dtype=bool)
        keep_mask[remove_mask] = False
        remaining_target_indices = target_indices[keep_mask].tolist()
    else:
        remaining_target_indices = target_indices.tolist()
    
    # Combine and return
    final_indices = remaining_target_indices + keep_indices
    return Subset(dataset, final_indices)

def _remove_from_subset_fast(subset: Subset, target_class: int, removal_ratio: float) -> Subset:
    """Fast removal from a Subset using vectorized operations."""
    subset_indices = np.array(subset.indices)
    
    # Get labels in batch if possible, otherwise single pass
    target_mask = []
    for original_idx in subset_indices:
        _, label = subset.dataset[original_idx]
        target_mask.append(label == target_class)
    
    target_mask = np.array(target_mask)
    target_positions = np.where(target_mask)[0]
    
    # Fast sampling
    num_to_remove = int(len(target_positions) * removal_ratio)
    
    if num_to_remove > 0:
        remove_positions = np.random.choice(target_positions, size=num_to_remove, replace=False)
        keep_mask = np.ones(len(subset_indices), dtype=bool)
        keep_mask[remove_positions] = False
        remaining_indices = subset_indices[keep_mask].tolist()
    else:
        remaining_indices = subset_indices.tolist()
    
    return Subset(subset.dataset, remaining_indices)

def _remove_from_concat_dataset_fast(concat_dataset: ConcatDataset, target_class: int, removal_ratio: float) -> ConcatDataset:
    """Fast removal from ConcatDataset with parallel processing capability."""
    new_datasets = []
    
    # Process each sub-dataset
    for sub_dataset in concat_dataset.datasets:
        modified_sub_dataset = remove_class_samples(sub_dataset, target_class, removal_ratio)
        if len(modified_sub_dataset) > 0:
            new_datasets.append(modified_sub_dataset)
    
    return ConcatDataset(new_datasets)

# Optimized version for datasets with accessible labels attribute
def remove_class_samples_optimized(dataset: Union[Dataset, Subset, ConcatDataset], 
                                 target_class: int, 
                                 removal_ratio: float,
                                 labels_attr: str = 'targets',
                                 random_seed: int = None) -> Union[Subset, ConcatDataset]:
    """
    Optimized version that uses pre-computed labels when available.
    
    Args:
        dataset: The input dataset
        target_class: The class label from which to remove samples
        removal_ratio: Ratio of samples to remove (0.0 to 1.0)
        labels_attr: Name of the labels attribute (e.g., 'targets', 'labels')
        random_seed: Optional random seed for reproducible results
    """
    if not 0.0 <= removal_ratio <= 1.0:
        raise ValueError("removal_ratio must be between 0.0 and 1.0")
    
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Try to use pre-computed labels for maximum speed
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        if hasattr(base_dataset, labels_attr):
            labels = getattr(base_dataset, labels_attr)
            if isinstance(labels, (list, tuple)):
                labels = np.array(labels)
            
            # Get labels for subset indices
            subset_labels = labels[dataset.indices]
            target_mask = subset_labels == target_class
            target_positions = np.where(target_mask)[0]
            
            num_to_remove = int(len(target_positions) * removal_ratio)
            
            if num_to_remove > 0:
                remove_positions = np.random.choice(target_positions, size=num_to_remove, replace=False)
                keep_mask = np.ones(len(dataset.indices), dtype=bool)
                keep_mask[remove_positions] = False
                remaining_indices = np.array(dataset.indices)[keep_mask].tolist()
            else:
                remaining_indices = dataset.indices
            
            return Subset(base_dataset, remaining_indices)
    
    elif hasattr(dataset, labels_attr) and not isinstance(dataset, ConcatDataset):
        # Direct access to labels
        labels = getattr(dataset, labels_attr)
        if isinstance(labels, (list, tuple)):
            labels = np.array(labels)
        
        target_mask = labels == target_class
        target_indices = np.where(target_mask)[0]
        other_indices = np.where(~target_mask)[0]
        
        num_to_remove = int(len(target_indices) * removal_ratio)
        
        if num_to_remove > 0:
            remove_indices = np.random.choice(target_indices, size=num_to_remove, replace=False)
            remaining_target_indices = np.setdiff1d(target_indices, remove_indices)
            final_indices = np.concatenate([remaining_target_indices, other_indices])
        else:
            final_indices = np.arange(len(dataset))
        
        return Subset(dataset, final_indices.tolist())
    
    # Fallback to general method
    return remove_class_samples(dataset, target_class, removal_ratio, random_seed)