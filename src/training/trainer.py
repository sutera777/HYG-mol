# HYG-mol/src/training/trainer.py

import torch
import numpy as np
from tqdm import tqdm
import traceback
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error, mean_absolute_error, r2_score

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    progress = tqdm(train_loader, desc=f'Epoch {epoch:03d}') 
    needs_edge_attr = hasattr(model, 'model_type') and model.model_type == 'attention'
    for batch_idx, batch in enumerate(progress):
        if batch is None:
            print(f"Skipping empty batch {batch_idx}")
            continue
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
 
            if needs_edge_attr and hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
                output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=batch.hyperedge_attr)
            else:
                output = model(batch.x, batch.edge_index, batch.batch)
            target = batch.y.float() 
            if target.dim() == 1:
                target = target.unsqueeze(1) 
            if output.shape != target.shape:
                if output.dim() == 1 and target.dim() == 2 and target.shape[1] == 1:
                    output = output.unsqueeze(1)
                elif output.dim() == 2 and output.shape[1] > 1 and target.dim() == 2 and target.shape[1] == 1 and not model.is_multi_label:
                     print(f"Warning: Output shape {output.shape} doesn't match single-label target shape {target.shape}. Taking first output column.")
                     raise ValueError(f"Shape mismatch in single-label task: Output={output.shape}, Target={target.shape}. Check model output layer.")
                else:
                    raise ValueError(
                        f"CRITICAL: Output shape {output.shape} and Target shape {target.shape} do not match "
                        f"after basic adjustment. Check data loading and model definition. "
                        f"Is 'is_multi_label' ({model.is_multi_label}) set correctly?"
                    )
            is_valid = ~torch.isnan(target) 
            if model.task_type == "classification":
 
 
                 binary_target = (target + 1) / 2
 
                 is_valid = is_valid & (target != 0)
                 loss_mat = criterion(output, binary_target)
            else: 
                loss_mat = criterion(output, target)
 
            loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))
            num_valid = torch.sum(is_valid)
            if num_valid == 0:
                print(f"Batch {batch_idx}: No valid labels, skipping loss calculation.")
 
                progress.set_postfix({'batch_loss': 0.0, 'avg_loss': total_loss / max(1, num_batches)})
                continue 
            loss = torch.sum(loss_mat) / num_valid 
 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({'batch_loss': loss.item(), 'avg_loss': total_loss / num_batches})
        except ValueError as ve: 
             print(f"Error in training batch {batch_idx}: {ve}")
 
 
             print("Skipping problematic batch.")
             continue 
        except Exception as e:
            print(f"Unexpected error in training batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            print("Skipping problematic batch.")
            continue
 
    if num_batches == 0:
 
         print("Warning: No valid batches processed in this epoch.")
         return 0.0 
 
    return total_loss / num_batches
 
 
 
def train_regression(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    progress = tqdm(train_loader, desc=f'Epoch {epoch}')
 
    needs_edge_attr = hasattr(model, 'model_type') and model.model_type == 'attention'
    for batch_idx, batch in enumerate(progress):
        if batch is None:
            continue
        batch = batch.to(device)
        optimizer.zero_grad()
 
        if needs_edge_attr and hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
            output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=batch.hyperedge_attr)
        else:
            output = model(batch.x, batch.edge_index, batch.batch)
        target = batch.y.float()
        if target.dim() == 1:
            target = target.view(-1, 1)
        loss = criterion(output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
        progress.set_postfix({'loss': loss.item()})
    return total_loss / num_batches if num_batches > 0 else 0
 
 
def evaluate(model, device, loader, phase="validation"):
    model.eval()
    y_scores_list = [] 
    y_true_list = []
 
    needs_edge_attr = hasattr(model, 'model_type') and model.model_type == 'attention'
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {phase}")):
            if batch is None:
                print(f"Skipping empty batch {batch_idx} in {phase}")
                continue
            try:
                batch = batch.to(device)
 
 
                if needs_edge_attr:
                    if hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
 
                        if batch.hyperedge_attr.size(1) != 5:
                            adjusted_attr = torch.zeros((batch.hyperedge_attr.size(0), 5), device=device)
                            min_dim = min(batch.hyperedge_attr.size(1), 5)
                            adjusted_attr[:, :min_dim] = batch.hyperedge_attr[:, :min_dim]
                            output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=adjusted_attr)
                        else:
                            output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=batch.hyperedge_attr)
                    else:
 
                        num_hyperedges = batch.edge_index[1].max().item() + 1 if batch.edge_index.size(1) > 0 else 1
                        hyperedge_attr = torch.zeros((num_hyperedges, 5), device=device)
                        output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=hyperedge_attr)
 
                else:
 
                    output = model(batch.x, batch.edge_index, batch.batch)
                pred = torch.sigmoid(output) 
                target = batch.y.float() 
 
 
                if pred.dim() == 1:
                    pred = pred.unsqueeze(1)
                if target.dim() == 1:
                    target = target.unsqueeze(1)
 
                if pred.shape != target.shape:
 
                    if pred.shape[1] > 1 and target.shape[1] == 1 and not model.is_multi_label:
                        print(f"Warning: Evaluation - Output shape {pred.shape} doesn't match single-label target shape {target.shape}. Taking first output column.")
                        pred = pred[:, 0:1] 
                    else:
 
                        raise ValueError(
                            f"Evaluation shape mismatch: Pred={pred.shape}, Target={target.shape}. "
                            f"Is 'is_multi_label' ({model.is_multi_label}) set correctly?"
                        )
 
 
                y_scores_list.append(pred.cpu())
                y_true_list.append(target.cpu())
            except ValueError as ve: 
                 print(f"Error processing evaluation batch {batch_idx}: {ve}")
                 print("Skipping problematic batch.")
                 continue
            except Exception as e:
                print(f"Unexpected error in evaluation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                print("Skipping problematic batch.")
                continue
 
    if not y_scores_list or not y_true_list:
        print(f"Warning: No valid predictions collected for {phase}. Returning default metrics.")
        return 0.5, 0.5, np.array([]), np.array([])
    try:
 
        y_scores = torch.cat(y_scores_list, dim=0).numpy() 
        y_true = torch.cat(y_true_list, dim=0).numpy()   
 
        if y_scores.ndim == 1:
            y_scores = y_scores.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
 
        if y_scores.shape[0] != y_true.shape[0]:
             raise ValueError(f"CRITICAL: Mismatch in number of samples after concatenation. Scores: {y_scores.shape[0]}, True: {y_true.shape[0]}")
 
        if y_scores.shape[1] != y_true.shape[1]:
 
             print(f"Warning: Mismatch in number of tasks. Scores: {y_scores.shape[1]}, True: {y_true.shape[1]}. Evaluating based on the minimum number of tasks.")
             min_tasks = min(y_scores.shape[1], y_true.shape[1])
             y_scores = y_scores[:, :min_tasks]
             y_true = y_true[:, :min_tasks]
 
 
        roc_list = []
        aupr_list = []
        valid_tasks_count = 0
        num_tasks = y_true.shape[1]
        print(f"\nCalculating metrics for {phase} ({num_tasks} tasks)...")
        for i in range(num_tasks):
            task_true = y_true[:, i]
            task_score = y_scores[:, i]
 
 
 
            is_valid_label = np.isin(task_true, [1.0, -1.0])
            is_not_nan = ~np.isnan(task_true)
            valid_mask = is_valid_label & is_not_nan
 
 
            if not np.any(valid_mask):
 
                continue
 
            valid_true = task_true[valid_mask]
            valid_score = task_score[valid_mask] 
 
            unique_classes = np.unique(valid_true)
            if len(unique_classes) < 2:
 
                continue
 
            binary_true = (valid_true + 1) / 2
            try:
                auc = roc_auc_score(binary_true, valid_score)
                aupr = average_precision_score(binary_true, valid_score)
                roc_list.append(auc)
                aupr_list.append(aupr)
                valid_tasks_count += 1
 
            except ValueError as metric_error:
 
                print(f"  Task {i}: Metric calculation error: {metric_error}. Skipping.")
                continue
            except Exception as e:
                print(f"  Task {i}: Unexpected metric error: {e}. Skipping.")
                continue
 
        if valid_tasks_count == 0:
            print(f"\nWARNING: No valid tasks found for metric calculation in {phase}. All tasks might have only one class or no valid labels.")
            return 0.5, 0.5 
 
        mean_auc = np.mean(roc_list) if roc_list else 0.5
        mean_aupr = np.mean(aupr_list) if aupr_list else 0.5 
        print(f"\n{phase} Average Metrics (over {valid_tasks_count}/{num_tasks} valid tasks):")
        print(f"ROC-AUC: {mean_auc:.4f}")
        print(f"AUPR:    {mean_aupr:.4f}")
        return mean_auc, mean_aupr, y_true, y_scores
    except ValueError as ve: 
        print(f"CRITICAL error during {phase} evaluation (post-processing): {ve}")
        import traceback
        traceback.print_exc()
        return 0.5, 0.5, np.array([]), np.array([])
    except Exception as e:
        print(f"CRITICAL unexpected error during {phase} evaluation (post-processing): {e}")
        import traceback
        traceback.print_exc()
        return 0.5, 0.5 
 
 
def evaluate_regression(model, device, loader, phase="validation"):
    model.eval()
    y_pred_list = []
    y_true_list = []
    needs_edge_attr = hasattr(model, 'model_type') and model.model_type == 'attention'
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating {phase}")):
            if batch is None:
                print(f"Skipping empty batch {batch_idx} in {phase}")
                continue
            try:
                batch = batch.to(device)
 
                if needs_edge_attr and hasattr(batch, 'hyperedge_attr') and batch.hyperedge_attr is not None:
                    output = model(batch.x, batch.edge_index, batch.batch, hyperedge_attr=batch.hyperedge_attr)
                else:
                    output = model(batch.x, batch.edge_index, batch.batch)
                target = batch.y.float()                            
 
                if output.dim() == 1:
                    output = output.unsqueeze(1)
                if target.dim() == 1:
                    target = target.unsqueeze(1)
 
                if output.shape != target.shape:
                    raise ValueError(
                        f"Evaluation shape mismatch: Output={output.shape}, Target={target.shape}. "
                        f"Is 'is_multi_label' ({model.is_multi_label}) set correctly for regression?"
                    )
 
                y_pred_list.append(output.cpu())
                y_true_list.append(target.cpu())
            except ValueError as ve:
                print(f"Error processing regression evaluation batch {batch_idx}: {ve}")
                print("Skipping problematic batch.")
                continue
            except Exception as e:
                print(f"Unexpected error in regression evaluation batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                print("Skipping problematic batch.")
                continue
    if not y_pred_list or not y_true_list:
        print(f"Warning: No valid predictions collected for {phase} regression. Returning default metrics.")
        return 0.0, float('inf'), float('inf'), -float('inf') 
    try:
 
        y_pred = torch.cat(y_pred_list, dim=0).numpy() 
        y_true = torch.cat(y_true_list, dim=0).numpy() 
        if y_pred.ndim == 1: y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1: y_true = y_true.reshape(-1, 1)
        if y_pred.shape[0] != y_true.shape[0]:
             raise ValueError(f"CRITICAL: Mismatch in number of samples after concatenation. Pred: {y_pred.shape[0]}, True: {y_true.shape[0]}")
        if y_pred.shape[1] != y_true.shape[1]:
             raise ValueError(f"CRITICAL: Mismatch in number of targets. Pred: {y_pred.shape[1]}, True: {y_true.shape[1]}")
        mse_list, rmse_list, mae_list, r2_list = [], [], [], []
        valid_targets_count = 0
        num_targets = y_true.shape[1]
        print(f"\nCalculating metrics for {phase} regression ({num_targets} targets)...")
        for i in range(num_targets):
            task_true = y_true[:, i]
            task_pred = y_pred[:, i]
 
            valid_mask = ~np.isnan(task_true)
            if not np.any(valid_mask):
 
                continue
            valid_true = task_true[valid_mask]
            valid_pred = task_pred[valid_mask] 
 
            if len(valid_true) < 2: 
 
                continue
            try:
                task_mse = mean_squared_error(valid_true, valid_pred)
                task_rmse = np.sqrt(task_mse)
                task_mae = mean_absolute_error(valid_true, valid_pred)
 
                if np.var(valid_true) < 1e-9: 
                     task_r2 = 0.0 if task_mse < 1e-9 else -float('inf') 
                     print(f"  Target {i}: Zero variance in true values. R² set to {task_r2:.4f}.")
                else:
                     task_r2 = r2_score(valid_true, valid_pred)
                mse_list.append(task_mse)
                rmse_list.append(task_rmse)
                mae_list.append(task_mae)
                r2_list.append(task_r2)
                valid_targets_count += 1
 
            except Exception as e:
                print(f"  Target {i}: Metric calculation error: {e}. Skipping.")
                continue
        if valid_targets_count == 0:
            print(f"\nWARNING: No valid targets found for metric calculation in {phase} regression.")
            return 0.0, float('inf'), float('inf'), -float('inf')
 
        mse = np.mean(mse_list) if mse_list else 0.0
        rmse = np.mean(rmse_list) if rmse_list else float('inf')
        mae = np.mean(mae_list) if mae_list else float('inf')
        r2 = np.mean(r2_list) if r2_list else -float('inf')
        print(f"\n{phase} Average Regression Metrics (over {valid_targets_count}/{num_targets} valid targets):")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")
        return mse, rmse, mae, r2
    except ValueError as ve:
        print(f"CRITICAL error during {phase} regression evaluation (post-processing): {ve}")
        import traceback
        traceback.print_exc()
        return 0.0, float('inf'), float('inf'), -float('inf')
    except Exception as e:
        print(f"CRITICAL unexpected error during {phase} regression evaluation (post-processing): {e}")
        import traceback
        traceback.print_exc()
        return 0.0, float('inf'), float('inf'), -float('inf')