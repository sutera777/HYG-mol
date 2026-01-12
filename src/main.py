# HYG-mol/src/main.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import json
from datetime import datetime
from tqdm import tqdm


import config 
from utils.utils import set_seed, plot_metrics, scaffold_split
from data.dataset import MoleculeDataset
from data.dataloader import CustomDataLoader
from models.networks import HyperGraphNet, AttentionHyperGraphNet
from training.trainer import train, evaluate, train_regression, evaluate_regression
from explainability.analyzer import MolecularExplainabilityAnalyzer


from rdkit import Chem


def main():

    parser = argparse.ArgumentParser(description="Train and evaluate a Hypergraph Network on molecular datasets.")
    

    parser.add_argument('--dataset', type=str, required=True, choices=config.SUPPORTED_DATASETS, 
                        help="Name of the dataset to use.")
    parser.add_argument('--task_type', type=str, default=None, choices=['classification', 'regression'],
                        help="Task type. If not set, it will be inferred from the dataset.")
    parser.add_argument('--max_samples', type=int, default=None, 
                        help="Maximum number of samples to load (for debugging).")
    

    parser.add_argument('--split_type', type=str, default='balanced_scaffold', 
                        choices=['balanced_scaffold', 'pure_scaffold', 'stratified_random'],
                        help="Method for splitting dataset into train/valid/test sets.")
    parser.add_argument('--balance_threshold', type=float, default=0.3,
                        help="Balance threshold for balanced scaffold split (0-1, lower is stricter).")


    parser.add_argument('--model_type', type=str, default='attention', choices=['standard', 'attention'],
                        help="Model architecture to use.")
    parser.add_argument('--feature_type', type=str, default='combined', choices=['combined', 'chemberta_only', 'traditional_only'],
                        help="Node feature type for the model.")
    parser.add_argument('--hidden_channels', type=int, default=128, 
                        help="Number of hidden channels in the model.")
    parser.add_argument('--heads', type=int, default=4, 
                        help="Number of attention heads (for attention model).")


    parser.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")

    parser.add_argument('--explain_smiles', type=str, default=None,
                        help="Comma-separated list of SMILES strings to analyze.")
    parser.add_argument('--analyze_examples', type=int, default=5,
                        help="Number of random test examples to analyze (0 to disable). Overridden by --explain_smiles.")
    parser.add_argument('--task_index_to_analyze', type=int, default=0,
                        help="For multi-task datasets, the index of the task to analyze (default: 0).")
    parser.add_argument('--generate_report', action='store_true', help="Generate an HTML report for explanations.")
    

    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--mode', type=str, default='train_eval', choices=['train_eval', 'eval_only', 'explain_only'],
                        help="Execution mode: train and evaluate, evaluate a pre-trained model, or only run explainability.")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save results (default: results/{dataset}_{timestamp}).")
    parser.add_argument('--gpu_id', type=int, default=0, help="ID of the GPU to use.")

    args = parser.parse_args()


    set_seed(args.seed)
    

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        device = torch.device(f"cuda:0")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available. Using CPU.")


    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(config.RESULTS_DIR, f"{args.dataset}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    best_model_path = os.path.join(output_dir, f"{args.dataset}_best_model.pt")


    print("\n[Phase 1/4] Loading and preparing data...")
    try:
        data_path = config.DATASET_PATHS.get(args.dataset)
        if not data_path:
            raise ValueError(f"Dataset '{args.dataset}' path not found in config.py.")
            
        dataset = MoleculeDataset(
            data_path,
            args.dataset,
            feature_type=args.feature_type,
            task_type=args.task_type,
            max_samples=args.max_samples
        )
        

        args.task_type = dataset.task_type
        print(f"Dataset '{args.dataset}' loaded. Task type: {args.task_type}, Multi-label: {dataset.is_multi_label}")
        print(f"Dataset size: {len(dataset)}, Node features: {dataset.num_node_features}, Classes/Targets: {dataset.num_classes}")


        train_idx, valid_idx, test_idx = scaffold_split(
            dataset,
            split_type=args.split_type,
            balance_threshold=args.balance_threshold,
            random_state=args.seed
        )
        train_subset = [dataset.get(i) for i in train_idx]
        valid_subset = [dataset.get(i) for i in valid_idx]
        test_subset = [dataset.get(i) for i in test_idx]


        train_loader = CustomDataLoader(train_subset, batch_size=args.batch_size, shuffle=True) if train_subset else None
        valid_loader = CustomDataLoader(valid_subset, batch_size=args.batch_size, shuffle=False) if valid_subset else None
        test_loader = CustomDataLoader(test_subset, batch_size=args.batch_size, shuffle=False) if test_subset else None
        
        print(f"DataLoaders created. Train batches: {len(train_loader) if train_loader else 0}, "
              f"Valid batches: {len(valid_loader) if valid_loader else 0}, "
              f"Test batches: {len(test_loader) if test_loader else 0}")
        
    except Exception as e:
        print(f"FATAL: Error during data preparation: {e}")
        return


    if args.model_type == 'standard':
        model = HyperGraphNet(
            num_node_features=dataset.num_node_features, hidden_channels=args.hidden_channels,
            num_classes=dataset.num_classes, task_type=args.task_type, is_multi_label=dataset.is_multi_label
        ).to(device)
        print("\nUsing standard HyperGraphNet model.")
    else: 
        model = AttentionHyperGraphNet(
            num_node_features=dataset.num_node_features, hidden_channels=args.hidden_channels,
            num_classes=dataset.num_classes, task_type=args.task_type, is_multi_label=dataset.is_multi_label,
            heads=args.heads, hyperedge_dim=config.HYPEREDGE_DIM
        ).to(device)
        print(f"\nUsing AttentionHyperGraphNet with {args.heads} heads.")
    
    print(f"Model architecture:\n{model}")
    

    if args.mode in ['train_eval']:
        print("\n[Phase 2/4] Training model...")
        
        if not train_loader or not valid_loader:
            print("FATAL: Train or validation loader is empty. Cannot start training.")
            return

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if args.task_type == 'regression':
            criterion = nn.MSELoss()
            train_func, eval_func = train_regression, evaluate_regression
            best_metric_is_lower, best_val_metric = True, float('inf')
        else: 
            criterion = nn.BCEWithLogitsLoss()
            train_func, eval_func = train, evaluate
            best_metric_is_lower, best_val_metric = False, 0.0

        train_losses, val_metrics_history = [], {'main': [], 'secondary': []}
        no_improve_epochs = 0
        
        for epoch in range(1, args.epochs + 1):
            epoch_train_loss = train_func(model, device, train_loader, optimizer, criterion, epoch)
            train_losses.append(epoch_train_loss)
            
            metrics = eval_func(model, device, valid_loader, "validation")
            
            if args.task_type == 'regression':
                val_rmse, val_mae = metrics[1], metrics[2]
                current_val_metric = val_rmse
                val_metrics_history['main'].append(val_rmse)
                val_metrics_history['secondary'].append(val_mae)
                print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | Val RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}")
            else:
                val_auc, val_aupr = metrics[0], metrics[1]
                current_val_metric = val_auc
                val_metrics_history['main'].append(val_auc)
                val_metrics_history['secondary'].append(val_aupr)
                print(f"Epoch {epoch:03d} | Train Loss: {epoch_train_loss:.4f} | Val AUC: {val_auc:.4f}, AUPR: {val_aupr:.4f}")
            
            improved = (current_val_metric < best_val_metric) if best_metric_is_lower else (current_val_metric > best_val_metric)
            if improved:
                best_val_metric = current_val_metric
                no_improve_epochs = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Validation metric improved. Model saved to {best_model_path}")
            else:
                no_improve_epochs += 1
                if no_improve_epochs >= args.patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs.")
                    break
        
        plot_metrics(train_losses, val_metrics_history, args.task_type, args.dataset, output_dir)
        print("Training finished.")

    print("\n[Phase 3/4] Evaluating on the test set...")
    if not os.path.exists(best_model_path):
        print(f"WARNING: Best model not found at '{best_model_path}'. Cannot perform final evaluation or explanation.")
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Best model loaded for final evaluation.")

        if not test_loader:
             print("Test loader is empty. Skipping final evaluation.")
        else:
            if args.task_type == 'regression':
                test_metrics = evaluate_regression(model, device, test_loader, "test")
                print(f"Final Test Metrics | MSE: {test_metrics[0]:.4f}, RMSE: {test_metrics[1]:.4f}, MAE: {test_metrics[2]:.4f}, R²: {test_metrics[3]:.4f}")
            else:
                test_auc, test_aupr, _, _ = evaluate(model, device, test_loader, "test")
                print(f"Final Test Metrics | AUC: {test_auc:.4f}, AUPR: {test_aupr:.4f}")


    print("\n[Phase 4/4] Performing explainability analysis...")
    smiles_to_analyze_list = []
    if args.explain_smiles:
        smiles_to_analyze_list = [s.strip() for s in args.explain_smiles.split(',') if s.strip()]
    elif args.analyze_examples > 0 and test_subset:
        num_to_analyze = min(args.analyze_examples, len(test_subset))
        selected_indices = np.random.choice(len(test_subset), num_to_analyze, replace=False)
        smiles_to_analyze_list = [test_subset[i].smiles for i in selected_indices]
    
    if not smiles_to_analyze_list:
        print("No molecules specified or available for analysis. Skipping.")
    elif not os.path.exists(best_model_path):
        print("Best model not found. Skipping analysis.")
    else:
        explainer = MolecularExplainabilityAnalyzer(model, device, dataset)
        analysis_results = []
        
        for i, smiles in enumerate(tqdm(smiles_to_analyze_list, desc="Analyzing molecules")):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"Invalid SMILES '{smiles}', skipping.")
                continue


            try:
                data_idx = dataset.valid_smiles.index(smiles)
                mol_data = dataset.get(data_idx)
                true_value_for_task = mol_data.y.view(-1)[args.task_index_to_analyze].item() if mol_data.y is not None else None
            except ValueError:
                print(f"SMILES '{smiles}' not found in the processed dataset. Cannot retrieve true label.")
                mol_data = None
                true_value_for_task = None

            if mol_data is None:
                 print(f"Could not find data for SMILES '{smiles}', skipping analysis.")
                 continue

            analysis = explainer.analyze_prediction(
                mol_data=mol_data,
                mol_smiles=smiles,
                true_value=true_value_for_task,
                task_index_to_analyze=args.task_index_to_analyze
            )
            analysis_results.append(analysis)


            vis_path = os.path.join(output_dir, f"explanation_mol_{i+1}.png")
            explainer.visualize_explanation(smiles, analysis_result=analysis, save_path=vis_path)


        json_path = os.path.join(output_dir, "analysis_results.json")
        with open(json_path, 'w') as f:

            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    if isinstance(obj, np.floating): return float(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)
            json.dump(analysis_results, f, cls=NumpyEncoder, indent=2)
        print(f"Analysis results saved to {json_path}")
        


    print("\nScript finished successfully.")


if __name__ == '__main__':
    main()
