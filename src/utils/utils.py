# src/utils/utils.py
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
from sklearn.model_selection import train_test_split

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def scaffold_split(dataset, valid_size=0.1, test_size=0.1, random_state=None,
                   split_type='pure_scaffold', balance_threshold=0.3):

    import numpy as np
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from collections import defaultdict
    from sklearn.model_selection import train_test_split

 
    if random_state is not None:
        np.random.seed(random_state)
    def generate_scaffold(smiles, include_chirality=False):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    def extract_labels():
        if dataset.task_type != 'classification':
 
 
            dummy_labels = np.zeros(len(dataset.processed_data))
            median_val = np.median([data.y.item() if data.y.numel() == 1 else data.y.view(-1)[0].item()
                                    for data in dataset.processed_data])
            dummy_labels = np.array(
                [1 if (data.y.item() if data.y.numel() == 1 else data.y.view(-1)[0].item()) > median_val else -1
                 for data in dataset.processed_data])
            return dummy_labels
        if dataset.is_multi_label:
 
            try:
                all_labels = []
                for data in dataset.processed_data:
                    if data.y.dim() == 0:  
                        all_labels.append(data.y.item())
                    elif data.y.dim() == 1:  
                        all_labels.append(data.y[0].item())
                    elif data.y.dim() == 2:  
                        all_labels.append(data.y[0, 0].item())
                    else:
                        all_labels.append(data.y.view(-1)[0].item())
                return np.array(all_labels)
            except Exception as e:
                print(f"Error extracting labels for multi-label dataset: {e}")
                return np.random.choice([-1, 1], size=len(dataset.processed_data))
        else:
 
            try:
                return np.array([data.y.item() if data.y.numel() == 1 else data.y.view(-1)[0].item()
                                 for data in dataset.processed_data])
            except Exception as e:
                print(f"Error extracting labels: {e}")
                return np.random.choice([-1, 1], size=len(dataset.processed_data))

 
    def compute_class_distribution(indices, labels):
        if len(indices) == 0:
            return {}
        subset_labels = labels[indices]
        unique, counts = np.unique(subset_labels, return_counts=True)
        return dict(zip(unique, counts / len(subset_labels)))

 
    def compute_balance_score(distribution):
        if not distribution:
            return 0
        values = np.array(list(distribution.values()))
 
 
        return 1 - np.sum((values - 1 / len(values)) ** 2)

 
    def all_splits_have_classes(splits, labels, min_classes=2):
        for split in splits:
            if len(split) == 0:
                return False
            unique_classes = np.unique(labels[split])
            if len(unique_classes) < min_classes:
                return False
        return True

 
    def pure_scaffold_split():
        scaffolds = {}
        for i, smiles in enumerate(dataset.valid_smiles):
            scaffold = generate_scaffold(smiles)
            if scaffold is None:
                continue
            scaffolds.setdefault(scaffold, []).append(i)

 
        scaffold_groups = sorted(scaffolds.items(), key=lambda x: len(x[1]), reverse=True)

 
        train_indices, valid_indices, test_indices = [], [], []
        n_total = len(dataset.processed_data)
        n_test = int(test_size * n_total)
        n_valid = int(valid_size * n_total)

        for scaffold, indices in scaffold_groups:
            if len(test_indices) < n_test:
                test_indices.extend(indices)
            elif len(valid_indices) < n_valid:
                valid_indices.extend(indices)
            else:
                train_indices.extend(indices)

 
        if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
            print("Warning: Pure scaffold split resulted in empty splits. Redistributing...")
 
            all_indices = list(range(len(dataset.processed_data)))
            np.random.shuffle(all_indices)

            n_test = int(test_size * len(all_indices))
            n_valid = int(valid_size * len(all_indices))

            test_indices = all_indices[:n_test]
            valid_indices = all_indices[n_test:n_test + n_valid]
            train_indices = all_indices[n_test + n_valid:]

        return train_indices, valid_indices, test_indices

 
    def balanced_scaffold_split(labels):
        scaffolds = {}
        for i, smiles in enumerate(dataset.valid_smiles):
            scaffold = generate_scaffold(smiles)
            if scaffold is None:
                continue
            scaffolds.setdefault(scaffold, []).append(i)

        scaffold_groups = list(scaffolds.items())

        train_indices, valid_indices, test_indices = [], [], []
        n_total = len(dataset.processed_data)
        n_test = int(test_size * n_total)
        n_valid = int(valid_size * n_total)
 
        target_sizes = {
            'test': n_test,
            'valid': n_valid,
            'train': n_total - n_test - n_valid
        }
 
        scaffold_groups = sorted(scaffold_groups, key=lambda x: len(x[1]), reverse=True)
 
        for scaffold, indices in scaffold_groups:
            if len(indices) == 0:
                continue
 
            current_dists = {
                'train': compute_class_distribution(train_indices, labels),
                'valid': compute_class_distribution(valid_indices, labels),
                'test': compute_class_distribution(test_indices, labels)
            }
            potential_dists = {
                'train': compute_class_distribution(train_indices + indices, labels),
                'valid': compute_class_distribution(valid_indices + indices, labels),
                'test': compute_class_distribution(test_indices + indices, labels)
            }
 
            current_scores = {k: compute_balance_score(v) for k, v in current_dists.items()}
            potential_scores = {k: compute_balance_score(v) for k, v in potential_dists.items()}
 
            balance_improvements = {k: potential_scores[k] - current_scores[k] for k in current_scores}
 
            current_sizes = {
                'train': len(train_indices),
                'valid': len(valid_indices),
                'test': len(test_indices)
            }
 
            for split, size in current_sizes.items():
                if size >= target_sizes[split]:
                    balance_improvements[split] = -float('inf')
 
            if len(test_indices) < target_sizes['test']:
 
                if len(test_indices) + len(indices) <= target_sizes['test'] and potential_scores[
                    'test'] >= balance_threshold:
                    test_indices.extend(indices)
                    continue
            if len(valid_indices) < target_sizes['valid']:
 
                if len(valid_indices) + len(indices) <= target_sizes['valid'] and potential_scores[
                    'valid'] >= balance_threshold:
                    valid_indices.extend(indices)
                    continue
 
            best_split = max(balance_improvements.items(), key=lambda x: x[1])[0]
            if best_split == 'train':
                train_indices.extend(indices)
            elif best_split == 'valid':
                valid_indices.extend(indices)
            else:
                test_indices.extend(indices)

 
        if dataset.task_type == 'classification' and not all_splits_have_classes(
                [train_indices, valid_indices, test_indices], labels):
            print("Warning: Initial scaffold assignment did not produce balanced class distributions.")
            print("Applying post-processing to ensure class balance...")
 
            class_indices = defaultdict(list)
            for i, label in enumerate(labels):
                class_indices[label].append(i)
 
            all_classes = set(labels)
 
            for split_name, split_indices in [('test', test_indices), ('valid', valid_indices)]:
                split_classes = set(labels[split_indices])
                missing_classes = all_classes - split_classes
                for cls in missing_classes:
 
                    cls_in_train = [idx for idx in class_indices[cls] if idx in train_indices]
                    if cls_in_train:
 
                        idx_to_move = np.random.choice(cls_in_train)
                        train_indices.remove(idx_to_move)
                        if split_name == 'test':
                            test_indices.append(idx_to_move)
                        else:
                            valid_indices.append(idx_to_move)
                        print(f"Moved a sample of class {cls} from train to {split_name} set for balance.")
 
            train_classes = set(labels[train_indices])
            missing_train_classes = all_classes - train_classes
            for cls in missing_train_classes:
 
                cls_in_valid = [idx for idx in class_indices[cls] if idx in valid_indices]
                if cls_in_valid:
                    idx_to_move = np.random.choice(cls_in_valid)
                    valid_indices.remove(idx_to_move)
                    train_indices.append(idx_to_move)
                    print(f"Moved a sample of class {cls} from valid to train set for balance.")
                else:
 
                    cls_in_test = [idx for idx in class_indices[cls] if idx in test_indices]
                    if cls_in_test:
                        idx_to_move = np.random.choice(cls_in_test)
                        test_indices.remove(idx_to_move)
                        train_indices.append(idx_to_move)
                        print(f"Moved a sample of class {cls} from test to train set for balance.")
 
        if len(train_indices) == 0 or len(valid_indices) == 0 or len(test_indices) == 0:
            print("Warning: One or more splits are empty after scaffold assignment.")
            print("Falling back to stratified random split.")
            return stratified_random_split(labels)

        return train_indices, valid_indices, test_indices

 
    def stratified_random_split(labels):
        indices = np.arange(len(labels))

 
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size,
            stratify=labels if dataset.task_type == 'classification' else None,
            random_state=random_state
        )

 
        train_idx, valid_idx = train_test_split(
            train_val_idx,
            test_size=valid_size / (1 - test_size),
            stratify=labels[train_val_idx] if dataset.task_type == 'classification' else None,
            random_state=random_state
        )

        return list(train_idx), list(valid_idx), list(test_idx)

 
    labels = extract_labels()

 
    if split_type == 'pure_scaffold':
        print("\nUsing pure scaffold split without class balance considerations")
        train_indices, valid_indices, test_indices = pure_scaffold_split()
    elif split_type == 'stratified_random':
        print("\nUsing stratified random split")
        train_indices, valid_indices, test_indices = stratified_random_split(labels)
    else:  
        print("\nUsing balanced scaffold split with class distribution consideration")
        train_indices, valid_indices, test_indices = balanced_scaffold_split(labels)

    print(f"\nData split | Train: {len(train_indices)}, Valid: {len(valid_indices)}, Test: {len(test_indices)}")

 
    if dataset.task_type == 'classification':
        for split_name, split_indices in zip(["Train", "Valid", "Test"],
                                             [train_indices, valid_indices, test_indices]):
            try:
                dist = compute_class_distribution(split_indices, labels)
                print(f"{split_name} class distribution:")
                for cls, prop in dist.items():
                    print(f"  Class {cls}: {prop:.2%}")
            except Exception as e:
                print(f"Error checking class distribution in {split_name} split: {e}")

    return train_indices, valid_indices, test_indices



def plot_metrics(train_losses, val_metrics_history, task_type, dataset_name, output_dir):

    if not train_losses:
        print("Warning: No training losses provided to plot_metrics. Skipping plot generation.")
        return

    plt.figure(figsize=(15, 6))
    
 
    plt.subplot(1, 2, 1)
    epochs_axis = range(1, len(train_losses) + 1)
    plt.plot(epochs_axis, train_losses, label='Train Loss', color='royalblue', linewidth=2)
    plt.title(f'{dataset_name.capitalize()} - Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
 
    plt.subplot(1, 2, 2)
    if task_type == 'classification':
        if val_metrics_history.get('auc') and len(val_metrics_history['auc']) == len(epochs_axis):
             plt.plot(epochs_axis, val_metrics_history['auc'], label='Validation AUC', marker='o', linestyle='--', color='darkorange')
        if val_metrics_history.get('aupr') and len(val_metrics_history['aupr']) == len(epochs_axis):
             plt.plot(epochs_axis, val_metrics_history['aupr'], label='Validation AUPR', marker='s', linestyle=':', color='forestgreen')
        plt.ylabel('Metric Value', fontsize=12)
        plt.title(f'{dataset_name.capitalize()} - Validation Metrics', fontsize=14)
    else:  # Regression
        ax1 = plt.gca()
        ax2 = ax1.twinx() 
        
        if val_metrics_history.get('rmse') and len(val_metrics_history['rmse']) == len(epochs_axis):
            ax1.plot(epochs_axis, val_metrics_history['rmse'], label='Validation RMSE', marker='o', linestyle='--', color='darkorange')
        if val_metrics_history.get('mae') and len(val_metrics_history['mae']) == len(epochs_axis):
            ax1.plot(epochs_axis, val_metrics_history['mae'], label='Validation MAE', marker='s', linestyle=':', color='indianred')
        ax1.set_ylabel('Error (RMSE, MAE)', fontsize=12, color='darkorange')
        ax1.tick_params(axis='y', labelcolor='darkorange')
        
        if val_metrics_history.get('r2') and len(val_metrics_history['r2']) == len(epochs_axis):
            ax2.plot(epochs_axis, val_metrics_history['r2'], label='Validation R²', marker='^', linestyle='-.', color='purple')
        ax2.set_ylabel('R² Score', fontsize=12, color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')

 
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        plt.title(f'{dataset_name.capitalize()} - Validation Metrics', fontsize=14)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plot_filename = f"{output_dir}/metrics_plot.png"
    plt.savefig(plot_filename)
    plt.close()  
    print(f"Saved training metrics plot to {plot_filename}")

