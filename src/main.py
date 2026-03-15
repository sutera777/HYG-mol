# src/main.py
import os
import argparse
import json
import torch
import numpy as np
import torch.nn as nn 
from datetime import datetime
import gc
 
from src import config  
from src.utils.utils import set_seed, scaffold_split, plot_metrics
from src.data.dataset import MoleculeDataset, MoleculeData, MoleculeHypergraph
from src.data.dataloader import CustomDataLoader
from src.models.networks import HyperGraphNet, AttentionHyperGraphNet
from src.training.trainer import train, evaluate, train_regression, evaluate_regression
from src.explainability.analyzer import MolecularExplainabilityAnalyzer
from rdkit import Chem

def main():
    parser = argparse.ArgumentParser(description="Train a Hypergraph Network on molecular datasets.")
    parser.add_argument('--task_type', type=str, default='classification',
                        choices=['classification', 'regression'],
                        help="Task type (classification/regression). This is often determined by the dataset.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['bace', 'BBBP', 'sider', 'clintox', 'tox21', 'toxcast', 'esol', 'FreeSolv',
                                 'Lipophilicity', 'qm7', 'qm8'],
                        help="Name of the dataset to use.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")
    parser.add_argument('--epochs', type=int, default=100, help="Maximum number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the Adam optimizer.")
    parser.add_argument('--hidden_channels', type=int, default=128, help="Number of hidden channels in the HyperGraphNet.")
    parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to load from the dataset (for debugging).")
    parser.add_argument('--feature_type', type=str, default='combined',
                        choices=['combined', 'chemberta_only', 'traditional_only'],
                        help="Feature type to use for node representation.")
    parser.add_argument('--split_type', type=str, default='pure_scaffold',
                        choices=['balanced_scaffold', 'pure_scaffold', 'stratified_random'],
                        help="Method for splitting dataset into train/valid/test sets.")
    parser.add_argument('--balance_threshold', type=float, default=0.3,
                        help="Balance threshold for balanced scaffold split (0-1, lower is stricter).")
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'attention'],
                        help="Model type: standard or attention-based hypergraph network")
    parser.add_argument('--attention_mode', type=str, default='node',
                        choices=['node', 'edge'],
                        help="Attention mode for the attention-based model")
    parser.add_argument('--heads', type=int, default=4,
                        help="Number of attention heads for the attention-based model")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="是否打印详细的分析过程信息")                        
    parser.add_argument('--analyze_examples', type=int, default=5,
                        help="Number of test examples to analyze (0 to disable, ignored if --explain_smiles is set).")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Directory to save results (default: results/{dataset}_{timestamp})")
    parser.add_argument('--task_index_to_analyze', type=int, default=0,
                        help="对于多任务数据集，要分析的任务索引 (默认为0)。")
    parser.add_argument('--explain_smiles', type=str, default=None,
                        help="Comma-separated list of SMILES strings to analyze for explainability. "
                             "Overrides --analyze_examples if provided.")
    parser.add_argument('--load_model_only', action='store_true',
                        help="If set, skip training and load the best model for analysis/testing.")
 

    args = parser.parse_args()

    global FEATURE_TYPE
    FEATURE_TYPE = args.feature_type

    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

 
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/{args.dataset}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    if torch.cuda.is_available():
        print(f"[Before dataset load] Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB | Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

 
 
    try:
        dataset = MoleculeDataset(
            f'data/{args.dataset}.csv',
            args.dataset,
            task_type=args.task_type,
            feature_type=args.feature_type,
            max_samples=args.max_samples
        )
        args.task_type = dataset.task_type 
        print(f"Dataset '{args.dataset}' loaded. Determined task type: {args.task_type}. Multi-label: {dataset.is_multi_label}")
        if len(dataset) == 0:
            raise ValueError("Dataset is empty after processing!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Number of node features: {dataset.num_node_features}")
        print(f"Number of classes/targets: {dataset.num_classes}")

 
 
        train_idx, valid_idx, test_idx = scaffold_split(
            dataset,
            valid_size=0.1,
            test_size=0.1,
            random_state=args.seed,
            split_type=args.split_type,
            balance_threshold=args.balance_threshold
        )
        print(f"Data split | Train: {len(train_idx)}, Valid: {len(valid_idx)}, Test: {len(test_idx)}")

 
        train_loader = None
        valid_loader = None
        if not args.load_model_only:
            if not train_idx or not valid_idx:
                raise ValueError("Train/Valid splits are empty, cannot train model.")
            train_subset = [dataset.get(i) for i in train_idx]
            valid_subset = [dataset.get(i) for i in valid_idx]
            train_loader = CustomDataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
            valid_loader = CustomDataLoader(valid_subset, batch_size=args.batch_size, shuffle=False)
            print(f"Train/Valid DataLoaders initialized | Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}")
        else:
            print("Skipping training data loaders initialization (load_model_only is set).")

 
        test_subset = [dataset.get(i) for i in test_idx]
        if not test_subset and not args.explain_smiles:
            print("Warning: Test subset is empty and no specific SMILES provided for analysis. Test evaluation might be skipped.")
        test_loader = CustomDataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
        print(f"Test DataLoader initialized | Test batches: {len(test_loader)}")


    except Exception as e:
        print(f"Error during data loading or preparation: {e}")
        import traceback
        traceback.print_exc()
        return

 
    model = None 
    best_model_path = f'{args.dataset}_best_model.pt'

 
    if not args.load_model_only:
 
        print("\nInitializing model for training...")
        if args.model_type == 'standard':
            model = HyperGraphNet(
                num_node_features=dataset.num_node_features,
                hidden_channels=args.hidden_channels,
                num_classes=dataset.num_classes,
                task_type=args.task_type,
                is_multi_label=dataset.is_multi_label
            ).to(device)
            print("Using standard HyperGraphNet model")
        else:
            model = AttentionHyperGraphNet(
                num_node_features=dataset.num_node_features,
                hidden_channels=args.hidden_channels,
                num_classes=dataset.num_classes,
                task_type=args.task_type,
                is_multi_label=dataset.is_multi_label,
                attention_mode=args.attention_mode,
                heads=args.heads,
                hyperedge_dim=config.HYPEREDGE_INPUT_DIM
            ).to(device)
            print(f"Using AttentionHyperGraphNet with {args.attention_mode} attention mode and {args.heads} heads")

        if args.task_type == 'regression':
            criterion = nn.MSELoss(reduction='none')
            best_metric_is_lower = True
            best_val_metric = float('inf')
        else:
            criterion = nn.BCEWithLogitsLoss(reduction='none')
            best_metric_is_lower = False
            best_val_metric = 0.0

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        print("\nModel, Criterion, Optimizer initialized successfully.")
        print(f"Model architecture: {model}")
        print(f"Loss function: {criterion.__class__.__name__} (reduction='none')")
        print(f"Task type: {args.task_type}, Best metric is lower: {best_metric_is_lower}")

        print("\nStarting training...")
        no_improve_epochs = 0
        train_losses = []
        val_metrics_history = {'auc': [], 'aupr': [], 'mse':[], 'rmse': [], 'mae': [], 'r2': []}
        best_epoch = -1

        for epoch in range(args.epochs):
            try:
                epoch_train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
                train_losses.append(epoch_train_loss)
            except ValueError as train_err:
                print(f"Error during training epoch {epoch}: {train_err}")
                print("Stopping training.")
                break
            except Exception as e:
                print(f"Unexpected error during training epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                print("Attempting to continue to next epoch...")
                train_losses.append(float('nan'))
                continue

            try:
                if args.task_type == 'regression':
                    val_mse, val_rmse, val_mae, val_r2 = evaluate_regression(model, device, valid_loader, "validation")
                    val_metrics_history['mse'].append(val_mse)
                    val_metrics_history['rmse'].append(val_rmse)
                    val_metrics_history['mae'].append(val_mae)
                    val_metrics_history['r2'].append(val_r2)
                    current_val_metric = val_rmse
                    print(f'Epoch: {epoch:03d}, Train Loss: {epoch_train_loss:.4f} | Val RMSE: {val_rmse:.4f}, Val MAE: {val_mae:.4f}, Val R²: {val_r2:.4f}')
                else:
                    val_auc, val_aupr, test_y_true, test_y_scores = evaluate(model, device, valid_loader, "validation")
                    val_metrics_history['auc'].append(val_auc)
                    val_metrics_history['aupr'].append(val_aupr)
                    current_val_metric = val_auc
                    print(f'Epoch: {epoch:03d}, Train Loss: {epoch_train_loss:.4f} | Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}')
            except Exception as e:
                print(f"Error during validation epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                if args.task_type == 'regression':
                    val_metrics_history['rmse'].append(float('inf'))
                    val_metrics_history['mae'].append(float('inf'))
                    val_metrics_history['r2'].append(-float('inf'))
                    current_val_metric = float('inf')
                else:
                    val_metrics_history['auc'].append(0.0)
                    val_metrics_history['aupr'].append(0.0)
                    current_val_metric = 0.0

            improved = False
            if best_metric_is_lower:
                if current_val_metric < best_val_metric:
                    best_val_metric = current_val_metric
                    improved = True
            else:
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    improved = True
            if improved:
                print(f"Validation metric improved to {best_val_metric:.4f}. Saving model...")
                no_improve_epochs = 0
                best_epoch = epoch
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_metric': best_val_metric,
                        'args': args
                    }, best_model_path) 
                except Exception as save_err:
                    print(f"Error saving model: {save_err}")
            else:
                no_improve_epochs += 1
                print(f"Validation metric did not improve for {no_improve_epochs} epoch(s).")
                if no_improve_epochs >= args.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("\nTraining finished.")
        try:
            plot_metrics(train_losses, val_metrics_history, args.task_type, args.dataset, output_dir=output_dir)
        except Exception as plot_err:
            print(f"Error plotting metrics: {plot_err}")

 
    print("\nLoading best model for testing and analysis...")
    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device)
 
 
 
            if args.model_type == 'standard':
                model = HyperGraphNet(
                    num_node_features=dataset.num_node_features,
                    hidden_channels=args.hidden_channels,
                    num_classes=dataset.num_classes,
                    task_type=args.task_type,
                    is_multi_label=dataset.is_multi_label
                ).to(device)
            else:
                model = AttentionHyperGraphNet(
                    num_node_features=dataset.num_node_features,
                    hidden_channels=args.hidden_channels,
                    num_classes=dataset.num_classes,
                    task_type=args.task_type,
                    is_multi_label=dataset.is_multi_label,
                    attention_mode=args.attention_mode,
                    heads=args.heads,
                    hyperedge_dim=5
                ).to(device)

            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded from epoch {checkpoint.get('epoch', 'N/A')}")

 
            print("\nEvaluating on test set...")
            if args.task_type == 'regression':
                test_mse, test_rmse, test_mae, test_r2 = evaluate_regression(model, device, test_loader, "test")
                print(f'\nFinal Test Results (from best model):')
                print(f'Test MSE:  {test_mse:.4f}')
                print(f'Test RMSE: {test_rmse:.4f}')
                print(f'Test MAE:  {test_mae:.4f}')
                print(f'Test R²:   {test_r2:.4f}')
            else:
                test_auc, test_aupr, test_y_true, test_y_scores = evaluate(model, device, test_loader, "test")
                print(f'\nFinal Test Results (from best model):')
                print(f'Test AUC:  {test_auc:.4f}')
                print(f'Test AUPR: {test_aupr:.4f}')
 
                if test_y_true.size > 0 and test_y_scores.size > 0:
                    results_filename = f"{args.dataset}_test_results.npz"
                    np.savez(results_filename, y_true=test_y_true, y_scores=test_y_scores)
                    print(f"Test results saved to {results_filename}")
                else:
                    print("No valid test results to save.")
        except Exception as e:
            print(f"Error loading or testing model: {e}")
            import traceback
            traceback.print_exc()
            model = None 
    else:
        print(f"Best model file '{best_model_path}' not found. Please train the model first or provide a valid path.")
        model = None 

 
    if model is None:
        print("\nSkipping explainability analysis as no valid model is available.")
        return 

    specified_smiles_list = []
    if args.explain_smiles:
        specified_smiles_list = [s.strip() for s in args.explain_smiles.split(',') if s.strip()]
        print(f"\nPerforming explainability analysis for specified SMILES: {specified_smiles_list}")
    elif args.analyze_examples > 0:
        if args.verbose: 
            print("\nPerforming molecular explainability analysis on random test examples...")
        num_to_analyze = min(args.analyze_examples, len(test_subset))
        selected_indices = np.random.choice(len(test_subset), num_to_analyze, replace=False)
        for i in selected_indices:
            data_item = test_subset[i]
            smiles = data_item.smiles
 
            true_value = None
            if hasattr(data_item, 'y') and data_item.y is not None and data_item.y.numel() > 0:
                if data_item.y.dim() == 0:
                    true_value = data_item.y.item()
                elif data_item.y.dim() == 1:
                    true_value = data_item.y[0].item()
                elif data_item.y.dim() == 2 and data_item.y.shape[1] > args.task_index_to_analyze:
                    true_value = data_item.y[0, args.task_index_to_analyze].item()
                else:
                    true_value = data_item.y.view(-1)[0].item()
 
 
 
            specified_smiles_list.append(smiles) 
    else:
        print("\nExplainability analysis is disabled (--analyze_examples 0 and --explain_smiles not provided).")
        return

    if not specified_smiles_list:
        print("\nNo SMILES strings found for explainability analysis. Exiting.")
        return

    try:
        explainer = MolecularExplainabilityAnalyzer(model, device, dataset, feature_type=args.feature_type, verbose=args.verbose)
        analysis_results = []

        for i, smiles_to_analyze in enumerate(specified_smiles_list):
            if args.verbose: 
                print(f"\nAnalyzing molecule {i + 1}/{len(specified_smiles_list)}: {smiles_to_analyze[:40]}...")
            mol = Chem.MolFromSmiles(smiles_to_analyze)
            if mol is None:
                print(f"  Error: Invalid SMILES string: {smiles_to_analyze}. Skipping.")
                analysis_results.append({'error': f"Invalid SMILES: {smiles_to_analyze}", 'molecule': {'smiles': smiles_to_analyze}})
                continue

            temp_hypergraph = MoleculeHypergraph(feature_type=args.feature_type) 
            try:
                temp_hypergraph.build_from_mol(mol)
                node_features = temp_hypergraph.node_features
                hyperedge_index = temp_hypergraph.hyperedge_index
                hyperedge_attr = temp_hypergraph.generate_enhanced_hyperedge_attributes()

                x_tensor = torch.FloatTensor(node_features)
                edge_index_tensor = torch.LongTensor(hyperedge_index)
                hyperedge_attr_tensor = torch.FloatTensor(hyperedge_attr)

                single_mol_data = MoleculeData(
                    x=x_tensor,
                    edge_index=edge_index_tensor,
                    hyperedge_attr=hyperedge_attr_tensor,
                    smiles=smiles_to_analyze
                )
                true_value_for_analysis = None
                if smiles_to_analyze in dataset.valid_smiles:
                    original_idx = dataset.valid_smiles.index(smiles_to_analyze)
                    if original_idx < len(dataset.processed_data):
                        original_data_item = dataset.processed_data[original_idx]
                        if original_data_item.y is not None and original_data_item.y.numel() > 0:
                            if original_data_item.y.dim() == 0:
                                true_value_for_analysis = original_data_item.y.item()
                            elif original_data_item.y.dim() == 1:
                                true_value_for_analysis = original_data_item.y[0].item()
                            elif original_data_item.y.dim() == 2 and original_data_item.y.shape[1] > args.task_index_to_analyze:
                                true_value_for_analysis = original_data_item.y[0, args.task_index_to_analyze].item()
                            else:
                                true_value_for_analysis = original_data_item.y.view(-1)[0].item()

            except Exception as e:
                print(f"  Error building hypergraph for {smiles_to_analyze}: {e}. Skipping.")
                analysis_results.append({'error': f"Error building hypergraph for {smiles_to_analyze}: {e}", 'molecule': {'smiles': smiles_to_analyze}})
                continue

            try:
                analysis = explainer.analyze_prediction(
                    single_mol_data,
                    smiles_to_analyze,
                    true_value=true_value_for_analysis,
                    task_index_to_analyze=args.task_index_to_analyze
                )
                analysis_results.append(analysis)

                true_display = analysis['prediction'].get('true_value_display', '未知')
                is_missing = analysis['prediction'].get('is_missing_label', False)
                if args.verbose:             
                    print(f"  Prediction: {analysis['prediction']['value']:.4f}")
                    print(f"  True value: {true_display}")
                    if is_missing:
                        print(f"  Note: True label is missing for this sample")
                    print(f"  Confidence: {analysis['prediction']['confidence']:.4f}")
                    print("  Key structures:")
                    for struct in analysis['attention']['important_hyperedges'][:3]:
                        print(f"    - {struct['label']}: {struct['attention']:.4f}")

                vis_path = os.path.join(output_dir, f"molecule_{i + 1}_explanation.png")
                explainer.visualize_explanation(smiles_to_analyze, analysis, save_path=vis_path)

            except Exception as e:
                print(f"  Error during analysis or visualization for {smiles_to_analyze}: {e}")
                import traceback
                traceback.print_exc()
                continue


        if analysis_results:
            try:
                import json
                class NumpyEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, np.ndarray):
                            return obj.tolist()
                        return json.JSONEncoder.default(self, obj)

                with open(os.path.join(output_dir, "analysis_results.json"), "w") as f:
                    json.dump(analysis_results, f, cls=NumpyEncoder, indent=2)
                print(f"Analysis results saved to analysis_results.json")
            except Exception as e:
                print(f"Error saving analysis results: {e}")
    except Exception as e:
        print(f"Error during explainability analysis: {e}")
        import traceback
        traceback.print_exc()




if __name__ == '__main__':
    main()
    print("\nScript finished.")
