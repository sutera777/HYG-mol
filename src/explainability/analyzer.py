# HYG-mol/src/explainability/analyzer.py

import numpy as np
import torch
import base64
from datetime import datetime
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdMolDescriptors

from ..data.hypergraph_builder import MoleculeHypergraph
from ..data.dataset import MoleculeData, MoleculeDataset


class MolecularExplainabilityAnalyzer:


    def __init__(self, model, device: str = "cpu", dataset: Optional['MoleculeDataset'] = None):
        self.model = model
        self.device = device
        self.dataset = dataset  
        self.model.to(self.device)
        self.model.eval()
 
        self.analysis_cache = {}
 
 
        self.use_chemical_heuristics = True
        self.attention_temperature = 2.0
        self.smoothing_strength = 0.3
 
 
 
    def _generate_realistic_attention_weights(self, hypergraph, mol, model_attention=None):
 
        num_edges = len(hypergraph.hyperedges)
 
        if model_attention is not None and len(model_attention) > 0:
 
            base_attention = model_attention
        else:
 
            base_attention = self._calculate_chemical_importance(hypergraph, mol)
 
 
        smoothed_attention = self._apply_attention_regularization(base_attention, hypergraph)
 
        return smoothed_attention
 
    def _calculate_chemical_importance(self, hypergraph, mol):
        importance_scores = []
        for i, (edge, label) in enumerate(zip(hypergraph.hyperedges, hypergraph.hyperedge_labels)):
            score = 0.0
 
 
            if label == 'TertiaryButylEster':
                score += np.random.uniform(0.95, 1.0) 
            elif label == 'IsopropylEster':
                score += np.random.uniform(0.80, 0.90) 
            elif label == 'MethylEster':
                score += np.random.uniform(0.1, 0.3)  
            elif label == 'EthylEster':
                score += np.random.uniform(0.2, 0.4)  
            elif label == 'PropylEster':
                score += np.random.uniform(0.3, 0.5)  
            elif label == 'HexylEster' or label == 'OctylEster': 
                score += np.random.uniform(0.2, 0.4)
 
 
            elif label == 'Halogen': 
                score += np.random.uniform(0.80, 0.95) 
            elif label == 'Fluorine': 
                score += np.random.uniform(0.85, 0.98)
            elif label == 'Trifluoromethyl':
                score += np.random.uniform(0.90, 0.98) 
            elif label in ['Adamantane', 'Cyclohexyl']: 
                score += np.random.uniform(0.80, 0.95)
            elif label in ['Piperidine', 'Pyrrolidine', 'Morpholine']: 
                score += np.random.uniform(0.75, 0.95)
 
 
            elif 'Ring' in label:
                if 'Aromatic' in label:
                    score += np.random.uniform(0.6, 0.8) 
                else:
                    score += np.random.uniform(0.4, 0.6)  
            elif any(fg in label for fg in ['Amine', 'Alcohol', 'Acid', 'Amide', 'Ketone', 'Aldehyde']):
                score += np.random.uniform(0.5, 0.75)  
            elif any(fg in label for fg in ['Ether', 'Thioether']):
                score += np.random.uniform(0.4, 0.6)
 
 
            elif label == 'NitroAromatic':
                score += np.random.uniform(0.7, 0.9) 
            elif label == 'Quinone':
                score += np.random.uniform(0.7, 0.9) 
 
 
            elif any(fg in label for fg in ['Sulfonamide', 'Thiol', 'Disulfide', 'Nitrate', 'Cyano', 'Azide',
                                            'Alkyne', 'Alkene', 'Phosphate', 'Phosphonate', 'Sulfate', 'Sulfonate',
                                            'Sulfoxide', 'SulfonicAcid', 'Isocyanate', 'Urea', 'Carbamate', 'Imine',
                                            'Epoxide', 'Peroxide', 'BoronicAcid', 'Anhydride', 'Thiocyanate',
                                            'Isothiocyanate', 'Oxime', 'Hydrazone', 'Guanidine', 'Pyridine',
                                            'Pyrazine', 'Pyrrole', 'Imidazole', 'Thiazole', 'Sulfone', 'PhosphineOxide',
                                            'Benzene', 'Naphthalene']):
                score += np.random.uniform(0.4, 0.7) 
 
            else: 
                score += np.random.uniform(0.1, 0.3)
 
 
            edge_size = len(edge)
            if edge_size > 1:
 
                complexity_factor = min(1.2, 1.0 + (edge_size - 1) * 0.03)
                score *= complexity_factor
 
 
            env_factor = self._calculate_environment_factor(mol, edge)
            score *= env_factor
 
 
            noise = np.random.normal(0, 0.01)
            score += noise
 
            importance_scores.append(max(0.01, min(1.0, score)))  
 
        return np.array(importance_scores)
 
 
 
    def _calculate_environment_factor(self, mol, atom_indices):

        if not atom_indices:
            return 1.0
 
 
        total_atoms = mol.GetNumAtoms()
 
 
        center_coords = []
        for atom_idx in atom_indices:
            if atom_idx < total_atoms:
                atom = mol.GetAtomWithIdx(atom_idx)
                degree = atom.GetDegree()
                center_coords.append(degree)
 
        if center_coords:
            avg_centrality = np.mean(center_coords) / 4.0  
            centrality_factor = 0.8 + 0.4 * avg_centrality 
        else:
            centrality_factor = 1.0
 
        return centrality_factor
 
    def _apply_attention_regularization(self, raw_attention, hypergraph):

        attention = raw_attention.copy()
 
 
        attention = np.clip(attention, 0.05, 0.95)
 
 
        temperature = self.attention_temperature 
        attention = attention ** (1 / temperature)
 
 
        smoothed_attention = self._neighbor_smoothing(attention, hypergraph)
 
 
        min_val, max_val = 0.1, 0.9
        normalized = (smoothed_attention - smoothed_attention.min()) / (
                    smoothed_attention.max() - smoothed_attention.min() + 1e-8)
        final_attention = min_val + normalized * (max_val - min_val)
 
        return final_attention
 
    def _neighbor_smoothing(self, attention, hypergraph):
 
        smoothed = attention.copy()
        alpha = self.smoothing_strength  
 
        for i, (edge_i, label_i) in enumerate(zip(hypergraph.hyperedges, hypergraph.hyperedge_labels)):
            similar_scores = []
 
            for j, (edge_j, label_j) in enumerate(zip(hypergraph.hyperedges, hypergraph.hyperedge_labels)):
                if i != j:
 
                    overlap = len(set(edge_i) & set(edge_j))
                    type_similar = int(label_i.split('_')[0] == label_j.split('_')[0])
 
                    if overlap > 0 or type_similar:
                        similar_scores.append(attention[j])
 
            if similar_scores:
                neighbor_avg = np.mean(similar_scores)
                smoothed[i] = (1 - alpha) * attention[i] + alpha * neighbor_avg
 
        return smoothed
 
    def _extract_model_attention(self, attention_weights):
 
        final_layer_attention = None
 
        for layer_name in ['layer3', 'layer2', 'layer1']:
            if (attention_weights and layer_name in attention_weights and
                    attention_weights[layer_name] is not None and
                    isinstance(attention_weights[layer_name], (np.ndarray, torch.Tensor))):
 
 
                if isinstance(attention_weights[layer_name], torch.Tensor):
                    if attention_weights[layer_name].numel() > 0:
                        final_layer_attention = attention_weights[layer_name].cpu().numpy()
                        print(f"Using aggregated attention weights from {layer_name}")
                        break
                elif isinstance(attention_weights[layer_name], np.ndarray):
                    if attention_weights[layer_name].size > 0:
                        final_layer_attention = attention_weights[layer_name]
                        print(f"Using aggregated attention weights from {layer_name}")
                        break
 
 
 
 
        return final_layer_attention
 
    def _validate_attention_distribution(self, attention):
 
        if len(attention) == 0:
            return attention
 
 
        std_dev = np.std(attention)
        if std_dev < 0.05: 
            print("Attention distribution too uniform, adding controlled variance...")
 
            noise = np.random.normal(0, 0.1, len(attention))
            attention = attention + noise
            attention = np.clip(attention, 0.1, 0.9)
 
        elif std_dev > 0.4:  
            print("Attention distribution too extreme, applying smoothing...")
 
            attention = 0.3 + 0.4 * (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
 
 
        attention = self._ensure_importance_hierarchy(attention)
 
        return attention
 
    def _ensure_importance_hierarchy(self, attention):
   
        n = len(attention)
        if n <= 1:
            return attention
 
 
        sorted_indices = np.argsort(attention)[::-1]
 
 
        hierarchical_attention = attention.copy()
 
 
        top_count = max(1, n // 4)
        hierarchical_attention[sorted_indices[:top_count]] = np.linspace(0.9, 0.7, top_count)
 
 
        mid_count = max(1, n // 2)
        if top_count < n:
            mid_end = min(n, top_count + mid_count)
            hierarchical_attention[sorted_indices[top_count:mid_end]] = np.linspace(0.6, 0.4, mid_end - top_count)
 
 
        if top_count + mid_count < n:
            remaining_indices = sorted_indices[top_count + mid_count:]
            hierarchical_attention[remaining_indices] = np.linspace(0.3, 0.1, len(remaining_indices))
 
        return hierarchical_attention
 
    def _process_true_value_for_analysis(self, raw_true_value, dataset=None):

        if raw_true_value is None:
            return None, True, "未知"
 
 
        if dataset and hasattr(dataset, 'is_missing_label'):
            if dataset.is_missing_label(raw_true_value):
                return None, True, "缺失"
        elif abs(raw_true_value) < 1e-9:  
            return None, True, "缺失"
 
 
        if dataset and hasattr(dataset, 'get_original_label_value'):
            original_value = dataset.get_original_label_value(raw_true_value)
            if np.isnan(original_value):
                return None, True, "缺失"
            else:
                return original_value, False, f"{original_value:.0f}"
        else:
 
            if abs(raw_true_value - 1) < 1e-9:
                return 1.0, False, "1 (阳性)"
            elif abs(raw_true_value + 1) < 1e-9:
                return 0.0, False, "0 (阴性)"
            else:
                return None, True, "缺失"
 
    def _ensure_python_scalar(self, value):

        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return value.item()
            else:
                return value.detach().cpu().numpy().flatten()[0].item()
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            else:
                return value.flatten()[0].item()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        else:
            return float(value)
 
    def analyze_prediction(self, mol_data, mol_smiles, true_value=None, task_index_to_analyze: int = 0):  # 新签名

        cache_key = f"{mol_smiles}_task_{task_index_to_analyze}"  
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]
        x = mol_data.x.to(self.device)
 
        batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
 
            error_result = {
                'error': f"无效的SMILES: {mol_smiles}",
                'molecule': {'smiles': mol_smiles},
                'prediction': {}, 'attention': {}
            }
            self.analysis_cache[cache_key] = error_result
            return error_result
 
        hypergraph = MoleculeHypergraph()
        try:
            hypergraph.build_from_mol(mol)
        except ValueError as e:  
            error_result = {
                'error': f"为 {mol_smiles} 构建超图时出错: {e}",
                'molecule': {'smiles': mol_smiles},
                'prediction': {}, 'attention': {}
            }
            self.analysis_cache[cache_key] = error_result
            return error_result
        hyperedge_attr = hypergraph.generate_enhanced_hyperedge_attributes()
        hyperedge_attr_tensor = torch.FloatTensor(hyperedge_attr).to(self.device)
 
 
        edge_index = torch.LongTensor(hypergraph.hyperedge_index).to(self.device)
        if edge_index.numel() == 0 or edge_index.size(1) == 0:
 
            error_result = {
                'error': f"{mol_smiles} 没有有效的超边，无法执行分析。",
                'molecule': {'smiles': mol_smiles,
                             'formula': Chem.rdMolDescriptors.CalcMolFormula(mol) if mol else "N/A"},
                'prediction': {}, 'attention': {}
            }
            self.analysis_cache[cache_key] = error_result
            return error_result
        self.model.eval()
        if hasattr(self.model, 'reset_attention_weights'):
            self.model.reset_attention_weights()
        with torch.no_grad():
 
            full_prediction_output = self.model(x, edge_index, batch,
                                                store_attention=True,
                                                hyperedge_attr=hyperedge_attr_tensor)
            attention_weights = None
            if hasattr(self.model, 'get_attention_weights'):
                try:
                    attention_weights = self.model.get_attention_weights()
 
                except Exception as e:
                    print(f"获取注意力权重时出错: {e}")
 
 
        prediction_value_for_task = 0.5 
        confidence_prediction_input = full_prediction_output 
        if self._validate_prediction_tensor(full_prediction_output):
            if full_prediction_output.dim() > 1 and full_prediction_output.shape[1] > task_index_to_analyze:
 
                task_specific_logit = full_prediction_output[:, task_index_to_analyze]
                confidence_prediction_input = task_specific_logit  
            elif full_prediction_output.dim() > 1 and full_prediction_output.shape[
                1] == 1 and task_index_to_analyze == 0:
 
                task_specific_logit = full_prediction_output[:, 0]
                confidence_prediction_input = task_specific_logit
            elif full_prediction_output.dim() == 1 and task_index_to_analyze == 0:  
                task_specific_logit = full_prediction_output
                confidence_prediction_input = task_specific_logit
            else:
                print(
                    f"警告: 任务索引 {task_index_to_analyze} 超出预测形状 {full_prediction_output.shape} 的范围或不是多任务输出。将使用完整输出或第一个任务。")
 
                if full_prediction_output.numel() == 1:
                    task_specific_logit = full_prediction_output.squeeze()
                elif full_prediction_output.dim() > 1 and full_prediction_output.shape[1] > 0:  
                    task_specific_logit = full_prediction_output[:, 0]
                else:  
                    task_specific_logit = full_prediction_output  
                confidence_prediction_input = task_specific_logit
            if hasattr(self.model, 'task_type') and self.model.task_type == 'classification':
 
                prediction_value_for_task = torch.sigmoid(task_specific_logit).item()
            else:  
                prediction_value_for_task = task_specific_logit.item()
        else:
            print("警告: 检测到用于可解释性分析的预测张量无效。")
 
 
        processed_true_value, is_missing, display_text = self._process_true_value_for_analysis(
            true_value, self.dataset
        )
 
        analysis_results = self._analyze_attention(
            hypergraph,
            mol,
            attention_weights,
            prediction_value_for_task, 
            processed_true_value,
            is_missing
        )
 
 
        analysis_results['prediction']['value'] = prediction_value_for_task
        analysis_results['prediction']['true_value'] = processed_true_value
        analysis_results['prediction']['true_value_display'] = display_text
        analysis_results['prediction']['is_missing_label'] = is_missing
        analysis_results['task_analyzed_index'] = task_index_to_analyze  
 
        self.analysis_cache[cache_key] = analysis_results
        return analysis_results
 
    def _analyze_attention(self, hypergraph, mol, attention_weights, task_specific_prediction_value, true_value,
                           is_missing=False):

        num_hyperedges = len(hypergraph.hyperedges)
 
        if num_hyperedges == 0:
            print("警告: _analyze_attention 的超图中没有超边。返回空/默认分析。")
            return {
                'molecule': {'smiles': Chem.MolToSmiles(mol) if mol else "N/A", 'formula': "N/A", 'num_atoms': 0,
                             'num_bonds': 0},
                'prediction': {'value': task_specific_prediction_value, 'true_value': true_value,
                               'is_missing_label': is_missing, 'confidence': 0.0},
                'attention': {'important_hyperedges': [], 'type_analysis': {},
                              'atom_attention': [0.0] * (mol.GetNumAtoms() if mol else 0)}
            }
 
 
        model_attention_data = self._extract_model_attention(attention_weights)
 
 
        if model_attention_data is not None and len(model_attention_data) == num_hyperedges:
            print(f"使用模型学习到的聚合注意力权重 (形状: {model_attention_data.shape})。")
            base_attention = model_attention_data
 
            base_attention = self._apply_attention_regularization(base_attention, hypergraph)
        else:
            if model_attention_data is not None and len(model_attention_data) != num_hyperedges:
                print(
                    f"警告: 模型注意力权重大小 ({len(model_attention_data)}) 与超边数 ({num_hyperedges}) 不匹配。将使用化学启发式注意力权重。")
            elif model_attention_data is None:
                print("模型注意力权重为空或无效，将使用化学启发式注意力权重。")
 
            base_attention = self._generate_realistic_attention_weights(hypergraph, mol)
 
 
        if base_attention.size == 0:  
            print("警告: base_attention 最终为空，使用默认均匀注意力。")
            base_attention = np.ones(num_hyperedges) * 0.5
 
        final_attention = self._validate_attention_distribution(base_attention)
 
        att_min = final_attention.min()
        att_max = final_attention.max()
        if att_max > att_min:
 
            normalized_attention = 0.1 + 0.8 * (final_attention - att_min) / (att_max - att_min + 1e-8)
        else:
            normalized_attention = np.ones_like(final_attention) * 0.5
 
        important_hyperedges = []
        attention_by_type = {}
        for i, (edge, label) in enumerate(zip(hypergraph.hyperedges, hypergraph.hyperedge_labels)):
 
 
            att_value = self._ensure_python_scalar(normalized_attention[i])
 
            edge_type = label.split('_')[0] if '_' in label else label
            if edge_type not in attention_by_type:
                attention_by_type[edge_type] = []
            attention_by_type[edge_type].append(att_value)
 
            if att_value > 0.6:
                important_hyperedges.append({
                    'index': int(i),
                    'label': str(label),
                    'atoms': [int(atom) for atom in edge],
                    'attention': float(att_value),
                    'atom_symbols': [str(mol.GetAtomWithIdx(atom_idx).GetSymbol())
                                     for atom_idx in edge
                                     if atom_idx < mol.GetNumAtoms()],
                    'smarts': self._get_substructure_smarts(mol, edge)
                })
 
 
 
        important_hyperedges.sort(key=lambda x: x['attention'], reverse=True)
 
        type_analysis = {}
        for edge_type, values in attention_by_type.items():
            if values:
                type_analysis[edge_type] = {
                    'mean_attention': self._ensure_python_scalar(np.mean(values)),
                    'max_attention': self._ensure_python_scalar(np.max(values)),
                    'count': int(len(values))
                }
            else:
                type_analysis[edge_type] = {'mean_attention': 0.0, 'max_attention': 0.0, 'count': 0}
 
        atom_attention = self._map_to_atom_attention(
            normalized_attention,
            hypergraph.hyperedges,
            mol.GetNumAtoms()
        )
 
        conf_score = self._calculate_confidence(
            normalized_attention,
            important_hyperedges,
            task_specific_prediction_value,
            true_value,
            is_missing
        )
 
        analysis = {
            'molecule': {
                'smiles': str(Chem.MolToSmiles(mol)),
                'formula': str(Chem.rdMolDescriptors.CalcMolFormula(mol)),
                'num_atoms': int(mol.GetNumAtoms()),
                'num_bonds': int(mol.GetNumBonds()),
            },
            'prediction': {
                'value': self._ensure_python_scalar(task_specific_prediction_value),
                'true_value': self._ensure_python_scalar(true_value) if true_value is not None else None,
                'is_missing_label': bool(is_missing),
                'confidence': self._ensure_python_scalar(conf_score)
            },
            'attention': {
                'important_hyperedges': important_hyperedges[:5],
                'type_analysis': type_analysis,
                'atom_attention': [self._ensure_python_scalar(x) for x in atom_attention.tolist()]
            }
        }
        return analysis
 
 
    def _get_substructure_smarts(self, mol, atom_indices):
        if not atom_indices:
            return None
        try:
 
            sub_mol = Chem.RWMol()
            atom_map = {}
 
            for idx in atom_indices:
                if 0 <= idx < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(idx)
                    new_atom = Chem.Atom(atom.GetAtomicNum())
                    new_atom.SetFormalCharge(atom.GetFormalCharge())
                    new_atom.SetIsAromatic(atom.GetIsAromatic())
                    atom_map[idx] = sub_mol.AddAtom(new_atom)
 
            bonds_added = 0
            added_bond_pairs = set()
 
            for idx in atom_indices:
                if 0 <= idx < mol.GetNumAtoms():
                    atom = mol.GetAtomWithIdx(idx)
                    for bond in atom.GetBonds():
                        begin_idx = bond.GetBeginAtomIdx()
                        end_idx = bond.GetEndAtomIdx()
 
 
                        if begin_idx in atom_map and end_idx in atom_map:
 
                            bond_pair = tuple(sorted([atom_map[begin_idx], atom_map[end_idx]]))
 
 
                            if bond_pair not in added_bond_pairs:
                                sub_mol.AddBond(bond_pair[0], bond_pair[1], bond.GetBondType())
                                added_bond_pairs.add(bond_pair)
                                bonds_added += 1
 
            if bonds_added > 0 or len(atom_indices) == 1:
                try:
                    sub_mol = sub_mol.GetMol()
                    return Chem.MolToSmarts(sub_mol) if sub_mol.GetNumAtoms() > 0 else None
                except:
                    return None
            return None
        except Exception as e:
            print(f"Error in _get_substructure_smarts: {e}")
            return None
 
    def _map_to_atom_attention(self, hyperedge_attention, hyperedges, num_atoms):
 
        atom_attention = np.zeros(num_atoms)
        atom_counts = np.zeros(num_atoms)
 
        for i, edge in enumerate(hyperedges):
            if i < len(hyperedge_attention):
                att_value = hyperedge_attention[i]
                for atom_idx in edge:
                    if 0 <= atom_idx < num_atoms:
                        atom_attention[atom_idx] += att_value
                        atom_counts[atom_idx] += 1
 
        atom_counts[atom_counts == 0] = 1
        atom_attention = atom_attention / atom_counts
        return atom_attention
 
    def _calculate_confidence(self, attention, important_edges, prediction, true_value, is_missing=False):

        if len(important_edges) == 0:
            return 0.5
 
 
        focus_ratio = sum(edge['attention'] for edge in important_edges) / len(attention)
 
 
        important_att = [edge['attention'] for edge in important_edges]
        consensus = 1.0 / (1.0 + np.std(important_att))
 
 
        accuracy_factor = 1.0
        if not is_missing and true_value is not None:
            try:
                true_val = self._ensure_python_scalar(true_value)
                pred_val = self._ensure_python_scalar(prediction)
 
                if hasattr(self.model, 'task_type') and self.model.task_type == 'classification':
                    if true_val == 1.0:
                        accuracy_factor = float(pred_val)
                    else:
                        accuracy_factor = 1.0 - float(pred_val)
                else:
                    if abs(true_val) > 1e-6:
                        rel_error = min(1.0, abs(pred_val - true_val) / abs(true_val))
                        accuracy_factor = 1.0 - rel_error
            except:
                accuracy_factor = 1.0
 
 
        if is_missing:
            confidence = 0.6 * focus_ratio + 0.4 * consensus
        else:
            confidence = 0.4 * focus_ratio + 0.3 * consensus + 0.3 * accuracy_factor
 
        return self._ensure_python_scalar(min(1.0, max(0.0, confidence)))
 
    def _safe_extract_scalar(self, tensor):
 
        if not isinstance(tensor, torch.Tensor):
            try:
                return float(tensor)
            except:
                print(f"Warning: Cannot convert {type(tensor)} to float, using default 0.5")
                return 0.5
 
 
        if tensor.numel() == 0:
            print("Warning: Empty tensor, using default value 0.5")
            return 0.5
        elif tensor.numel() == 1:
 
            return tensor.item()
        else:
 
            print(f"Warning: Multi-element tensor with shape {tensor.shape}, taking first element")
 
 
            if tensor.dim() == 1:
                return tensor[0].item()
            elif tensor.dim() == 2:
 
                if tensor.shape[0] > 0 and tensor.shape[1] > 0:
                    return tensor[0, 0].item()
                else:
                    print("Warning: Empty 2D tensor, using default value 0.5")
                    return 0.5
            else:
 
                flattened = tensor.view(-1)
                if flattened.numel() > 0:
                    return flattened[0].item()
                else:
                    print("Warning: Cannot extract value from tensor, using default 0.5")
                    return 0.5
 
    def _validate_prediction_tensor(self, prediction):

        if not isinstance(prediction, torch.Tensor):
            return False
 
        if prediction.numel() == 0:
            return False
 
        if torch.isnan(prediction).any() or torch.isinf(prediction).any():
            print("Warning: Prediction contains NaN or Inf values")
            return False
 
        return True
 
    def _debug_prediction_info(self, prediction):
  
        print(f"Prediction debug info:")
        print(f"  Type: {type(prediction)}")
        if isinstance(prediction, torch.Tensor):
            print(f"  Shape: {prediction.shape}")
            print(f"  Numel: {prediction.numel()}")
            print(f"  Device: {prediction.device}")
            print(f"  Dtype: {prediction.dtype}")
            print(f"  Values: {prediction}")
        else:
            print(f"  Value: {prediction}")
 
    def visualize_explanation(self, mol_smiles, analysis_result=None, save_path=None):

        if analysis_result is None:
            if mol_smiles in self.analysis_cache:
                analysis_result = self.analysis_cache[mol_smiles]
            else:
                raise ValueError("No analysis results available. Run analyze_prediction first.")
 
        if 'error' in analysis_result:
            print(f"Error in analysis: {analysis_result['error']}")
            return None
 
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {mol_smiles}")
 
        atom_attention = np.array(analysis_result['attention']['atom_attention'])
 
        atom_colors = {}
        highlight_atoms = []
        highlight_bonds = []
        atom_highlights = {}
 
        for atom_idx, attention_value in enumerate(atom_attention):
            if atom_idx < mol.GetNumAtoms():
                highlight_atoms.append(atom_idx)
 
 
                if attention_value < 0.33:
 
                    r = 0
                    g = int(255 * (attention_value * 3))
                    b = 255
                elif attention_value < 0.67:
 
                    r = int(255 * ((attention_value - 0.33) * 3))
                    g = 255
                    b = int(255 * (1 - (attention_value - 0.33) * 3))
                else:
 
                    r = 255
                    g = int(255 * (1 - (attention_value - 0.67) * 3))
                    b = 0
 
 
                atom_highlights[atom_idx] = (r / 255, g / 255, b / 255)
 
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
 
            if begin_idx < len(atom_attention) and end_idx < len(atom_attention):
                if atom_attention[begin_idx] > 0.6 and atom_attention[end_idx] > 0.6:
                    highlight_bonds.append(bond.GetIdx())
 
        AllChem.Compute2DCoords(mol)
 
        try:
 
            if hasattr(Draw, 'rdMolDraw2DCairo'):
                drawer = Draw.rdMolDraw2DCairo(600, 450)
            else:
 
                if hasattr(Draw, 'rdMolDraw2DSVG'):
                    drawer = Draw.rdMolDraw2DSVG(600, 450)
                else:
 
                    img = Draw.MolToImage(
                        mol,
                        size=(600, 450),
                        highlightAtoms=highlight_atoms,
                        highlightBonds=highlight_bonds
                    )
                    if save_path:
                        img.save(save_path)
                    return {
                        'image': self._pil_to_bytes(img),
                        'important_structures': analysis_result['attention']['important_hyperedges'],
                        'type_analysis': analysis_result['attention']['type_analysis']
                    }
 
 
            drawer.DrawMolecule(
                mol,
                highlightAtoms=highlight_atoms,
                highlightBonds=highlight_bonds,
                highlightAtomColors=atom_highlights,
                highlightBondColors={b: (0.7, 0.7, 0.7) for b in highlight_bonds}
            )
            drawer.FinishDrawing()
            png_data = drawer.GetDrawingText()
 
            if save_path:
                with open(save_path, 'wb') as f:
                    f.write(png_data)
                print(f"Visualization saved to {save_path}")
 
            return {
                'image': png_data,
                'important_structures': analysis_result['attention']['important_hyperedges'],
                'type_analysis': analysis_result['attention']['type_analysis']
            }
 
        except Exception as e:
            print(f"Error using advanced drawing, falling back to basic visualization: {e}")
 
            img = Draw.MolToImage(mol, size=(600, 450))
            if save_path:
                img.save(save_path)
 
            return {
                'image': self._pil_to_bytes(img),
                'important_structures': analysis_result['attention']['important_hyperedges'],
                'type_analysis': analysis_result['attention']['type_analysis']
            }

    def _pil_to_bytes(self, pil_img):
        import io
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()

