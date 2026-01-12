# HYG-mol/src/data/dataset.py
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel

import config
from .hypergraph_builder import MoleculeHypergraph

CHEMBERTA_TOKENIZER = None
CHEMBERTA_MODEL = None
def load_chemberta_if_needed(feature_type):
    global CHEMBERTA_MODEL, CHEMBERTA_TOKENIZER
    if feature_type != 'traditional_only' and CHEMBERTA_MODEL is None:
        print("Loading ChemBERTa model and tokenizer once...")
        CHEMBERTA_MODEL = AutoModel.from_pretrained(config.CHEMBERTA_PATH).to('cpu')
        CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(config.CHEMBERTA_PATH)
        print("ChemBERTa loaded.")

class MoleculeData(Data):
    def __init__(self, x=None, edge_index=None, y=None, hyperedge_attr=None, smiles=None, **kwargs):


        super(MoleculeData, self).__init__(**kwargs)


        self.hyperedge_attr = None
        self.smiles = None


        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.FloatTensor(x)
            if x.size(0) == 0:
                print("Warning: Empty feature matrix provided")
            self.x = x


        if edge_index is not None:
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.LongTensor(edge_index)
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                print(f"Warning: Unusual edge_index shape: {edge_index.shape}")
            self.edge_index = edge_index


        if y is not None:
            if not isinstance(y, torch.Tensor):
                y = torch.FloatTensor(y)
            self.y = y


        if hyperedge_attr is not None:
            if not isinstance(hyperedge_attr, torch.Tensor):
                hyperedge_attr = torch.FloatTensor(hyperedge_attr)
            self.hyperedge_attr = hyperedge_attr


        if smiles is not None:
            self.smiles = smiles

    @classmethod
    def from_cache(cls, filepath):
        data = torch.load(filepath)
        if not cls.verify_data(data):
            raise ValueError("Invalid cached data")
        return data

    @staticmethod
    def verify_data(data):
        if not isinstance(data, Data):
            print("Invalid data type")
            return False


        if not hasattr(data, 'x') or not hasattr(data, 'edge_index') or not hasattr(data, 'y'):
            print("Missing required attributes in data")
            return False


        if data.x is None:
            print("Data contains None x attribute")
            return False
        if data.edge_index is None:
            print("Data contains None edge_index attribute")
            return False
        if data.y is None:
            print("Data contains None y attribute")
            return False


        if data.x.size(0) == 0:
            print("Empty x in data")
            return False
        if data.edge_index.numel() == 0:
            print("Empty edge_index in data")
            return False


        if not torch.isfinite(data.x).all():
            print("Non-finite values in data.x")
            return False
        if not torch.isfinite(data.y).all():
            print("Non-finite values in data.y")
            return False


        if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
            if not torch.isfinite(data.hyperedge_attr).all():
                print("Non-finite values in data.hyperedge_attr")
                return False


        num_nodes = data.x.size(0)
        if data.edge_index.numel() > 0: 

            if data.edge_index.size(0) != 2:
                print(f"Invalid edge_index shape: expected (2, N), got {data.edge_index.shape}")
                return False


            node_indices = data.edge_index[0]  
            edge_indices = data.edge_index[1]  


            if node_indices.min().item() < 0 or node_indices.max().item() >= num_nodes:
                print(
                    f"Invalid node indices in edge_index: range [{node_indices.min().item()}, {node_indices.max().item()}], num_nodes={num_nodes}")
                return False


            if edge_indices.min().item() < 0:
                print(f"Invalid edge indices in edge_index: min={edge_indices.min().item()} is negative")
                return False


            if hasattr(data, 'hyperedge_attr') and data.hyperedge_attr is not None:
                num_hyperedges = data.hyperedge_attr.size(0)
                if edge_indices.max().item() >= num_hyperedges:
                    print(
                        f"Invalid edge indices in edge_index: max={edge_indices.max().item()}, num_hyperedges={num_hyperedges}")
                    return False

        return True

    def get_hyperedge_attr(self):
        return self.hyperedge_attr

class MoleculeDataset(Dataset):
    def __init__(self, data_path, dataset_name, feature_type, task_type='classification', max_samples=None):
        super(MoleculeDataset, self).__init__()
        self.dataset_name = dataset_name.lower()
        self.task_type = task_type
        self.feature_type = feature_type
        self.cache_dir = os.path.join(config.PROCESSED_DATA_DIR, self.dataset_name)
        self.metadata_file = os.path.join(self.cache_dir, "metadata.pt")
        self.is_multi_label = False
        os.makedirs(self.cache_dir, exist_ok=True)
        load_chemberta_if_needed(self.feature_type)
        # 添加标签编码映射记录
        self.label_encoding = {
            'positive': 1,
            'negative': -1,
            'missing': 0,
            'original_to_encoded': {},
            'encoded_to_original': {}
        }

        if self._try_load_cache():
            print("\nLoaded processed data from cache")
            return

        print(f"\nLoading dataset from {data_path}")
        df = pd.read_csv(data_path)
        print(f"Total records in CSV: {len(df)}")
        df.columns = df.columns.str.strip().str.lower()

        # 使用适当的数据集处理方法
        dataset_handler = getattr(self, f"_process_{self.dataset_name}", None)
        if dataset_handler:
            dataset_handler(df)
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if max_samples is not None:
            print(f"\nUsing only the first {max_samples} samples for testing.")
            self.smiles_list = self.smiles_list[:max_samples]
            self.labels = self.labels[:max_samples]

        self.valid_smiles = []
        self.valid_indices = []
        self.processed_data = []
        self.hyperedge_attrs = {}

        print("\nProcessing molecules...")
        success_count = 0
        error_count = 0
        empty_count = 0

        for i, smiles in enumerate(tqdm(self.smiles_list)):
            if pd.isna(smiles):
                empty_count += 1
                print(f"\nSkipping empty SMILES at index {i}")
                continue
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    error_count += 1
                    print(f"\nFailed to parse SMILES {i}: {smiles}")
                    continue
                hypergraph = MoleculeHypergraph(
                    feature_type=self.feature_type, 
                    chemberta_model=CHEMBERTA_MODEL, 
                    tokenizer=CHEMBERTA_TOKENIZER
                )
                try:
                    hypergraph.build_from_mol(mol)
                except Exception as e:
                    error_count += 1
                    print(f"\nError building hypergraph for molecule {i} with SMILES {smiles}: {str(e)}")
                    continue

                if hypergraph.node_features is None or hypergraph.hyperedge_index is None:
                    error_count += 1
                    print(f"\nInvalid hypergraph for molecule {i} with SMILES {smiles}")
                    continue
                x = torch.FloatTensor(hypergraph.node_features)
                edge_index = torch.LongTensor(hypergraph.hyperedge_index)
                num_nodes = x.size(0)
                if edge_index.shape[0] != 2:
                    error_count += 1
                    print(
                        f"\nInvalid edge_index shape for molecule {i} with SMILES {smiles}: expected (2, N), got {edge_index.shape}")
                    continue

                # 分别检查节点索引和超边索引
                node_indices = edge_index[0]  # 第一行是节点索引
                edge_indices = edge_index[1]  # 第二行是超边索引

                if node_indices.max() >= num_nodes or node_indices.min() < 0:
                    error_count += 1
                    print(
                        f"\nInvalid node indices in edge_index for molecule {i} with SMILES {smiles}: range [{node_indices.min()}, {node_indices.max()}], num_nodes={num_nodes}")
                    continue

                num_hyperedges = len(hypergraph.hyperedges)
                if edge_indices.max() >= num_hyperedges or edge_indices.min() < 0:
                    error_count += 1
                    print(
                        f"\nInvalid edge indices in edge_index for molecule {i} with SMILES {smiles}: range [{edge_indices.min()}, {edge_indices.max()}], num_hyperedges={num_hyperedges}")
                    continue

                # 使用数据集特定的标签处理函数
                y = self.process_label(i)

                if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
                    error_count += 1
                    print(f"\nNon-finite values in molecule {i}")
                    continue

                hyperedge_attr = hypergraph.generate_enhanced_hyperedge_attributes()
                if hyperedge_attr is None:
                    print(f"Warning: No hyperedges found for molecule {i}. Creating default attribute.")
                    hyperedge_attr = np.zeros((1, 5), dtype=np.float32)  # 安全措施
                # 在将hyperedge_attr存储到字典之前进行类型检查
                if isinstance(hyperedge_attr, torch.Tensor):
                    self.hyperedge_attrs[smiles] = hyperedge_attr.cpu().numpy()
                else:
                    # 已经是numpy数组，直接存储
                    self.hyperedge_attrs[smiles] = hyperedge_attr
                # 确保传入MoleculeData的是tensor
                hyperedge_attr_tensor = torch.FloatTensor(hyperedge_attr)
                data = MoleculeData(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    hyperedge_attr=hyperedge_attr_tensor,  # 总是tensor
                    smiles=smiles
                )
                data.smiles = smiles
                self.processed_data.append(data)
                self.valid_smiles.append(smiles)
                self.valid_indices.append(i)

                success_count += 1

            except Exception as e:
                error_count += 1
                print(f"\nError processing molecule {i}: {str(e)}")
                continue
        if len(self.processed_data) == 0:
            raise ValueError("No valid molecules processed!")

        # 检查数据有效性
        self._check_data_validity()

        # 保存处理好的数据
        self._save_cache()

        # 打印处理摘要
        print("\nProcessing Summary:")
        print(f"Total molecules: {len(self.smiles_list)}")
        print(f"Successfully processed: {success_count}")
        print(f"Failed to process: {error_count}")
        print(f"Empty SMILES: {empty_count}")

        # 打印标签分布统计信息
        self._print_label_statistics()



    def _process_bace(self, df):
        """处理 BACE 数据集（单标签分类）"""
        required_cols = ['mol', 'class']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"bace dataset must contain columns: {required_cols}")

        self.smiles_list = df['mol'].values
        self.labels = df['class'].values
        # 将0转换为-1
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.label_cols = ['class']
        self.is_multi_label = False

        # 为 BACE 定义标签处理函数 - 返回形状为 [1] 的张量
        self.process_label = lambda idx: torch.FloatTensor([[float(self.labels[idx])]])

        print("\nLabel distribution in original data:")
        print(df['class'].value_counts(normalize=True))

    def _process_bbbp(self, df):
        """处理 BBBP 数据集（单标签分类）"""
        required_cols = ['smiles', 'p_np']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"BBBP dataset must contain columns: {required_cols}")

        self.smiles_list = df['smiles'].values
        self.labels = df['p_np'].values
        # 将0转换为-1
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.label_cols = ['p_np']
        self.is_multi_label = False

        # 为 BBBP 定义标签处理函数 - 返回形状为 [1] 的张量
        self.process_label = lambda idx: torch.FloatTensor([[float(self.labels[idx])]])

        print("\nLabel distribution in original data:")
        print(df['p_np'].value_counts(normalize=True))

    def _process_sider(self, df):
        """处理 SIDER 数据集（多标签分类）"""
        if 'smiles' not in df.columns:
            raise ValueError("SIDER dataset must contain a 'smiles' column")

        self.smiles_list = df['smiles'].values
        label_cols = [col for col in df.columns if col != 'smiles']
        self.label_cols = label_cols
        self.labels = df[label_cols].values.astype(float)
        # 将0转换为-1
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.is_multi_label = True

        # 为 SIDER 定义标签处理函数 - 返回形状为 [num_labels] 的张量
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0) # 返回 [1, num_labels]

        print("\nLabel distribution in original data:")
        print(df[label_cols].apply(lambda x: pd.value_counts(x, normalize=True)))

    def _process_clintox(self, df):
        """处理 ClinTox 数据集（单标签分类）"""
        if 'smiles' not in df.columns:
            raise ValueError("ClinTox dataset must contain a 'smiles' column")

        label_cols = ['fda_approved', 'ct_tox']
        required_cols = ['smiles'] + label_cols
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"ClinTox dataset must contain columns: {required_cols}")
        self.smiles_list = df['smiles'].values
        self.labels = df[label_cols].values.astype(float)
        # 将0转换为-1（适用于整个二维标签数组）
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.labels = np.nan_to_num(self.labels, nan=0)
        self.label_cols = label_cols
        self.is_multi_label = True
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0)
        # print("\nLabel distribution in original data:")
        # print(pd.value_counts(self.labels, normalize=True))

    def _process_tox21(self, df):
        """处理 Tox21 数据集（多标签分类）"""
        if 'smiles' not in df.columns:
            raise ValueError("Tox21 dataset must contain a 'smiles' column")

        self.smiles_list = df['smiles'].values
        label_cols = [col for col in df.columns if col.lower() not in ['smiles', 'mol_id']]
        self.label_cols = label_cols
        self.labels = df[label_cols].values.astype(float)
        # 将0转换为-1，并处理NaN
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.labels = np.nan_to_num(self.labels, nan=0)
        self.is_multi_label = True

        # 为 Tox21 定义标签处理函数 - 返回形状为 [num_labels] 的张量
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0)  # 返回 [1, num_labels]

        print("\nLabel distribution after processing:")
        print(df[label_cols].apply(lambda x: pd.value_counts(x, normalize=True)))

    def _process_toxcast(self, df):
        """处理 ToxCast 数据集（多标签分类）"""
        if 'smiles' not in df.columns:
            raise ValueError("ToxCast dataset must contain a 'smiles' column")

        self.smiles_list = df['smiles'].values
        label_cols = [col for col in df.columns if col != 'smiles']
        self.label_cols = label_cols
        self.labels = df[label_cols].values.astype(float)
        # 将0转换为-1，并处理NaN
        self.labels = np.where(self.labels == 0, -1, self.labels)
        self.labels = np.nan_to_num(self.labels, nan=0)
        self.is_multi_label = True

        # 为 ToxCast 定义标签处理函数 - 返回形状为 [num_labels] 的张量
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0) # 返回 [1, num_labels]

        print("\nLabel distribution in original data:")
        print(df[label_cols].apply(lambda x: pd.value_counts(x, normalize=True)))

    def _process_esol(self, df):
        """处理 ESOL 数据集（单目标回归）"""
        required_cols = ['smiles', 'measured log solubility in mols per litre']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"ESOL dataset must contain columns: {required_cols}")

        self.smiles_list = df['smiles'].values
        self.labels = df['measured log solubility in mols per litre'].values.astype(float)
        self.label_cols = ['log_solubility']
        self.is_multi_label = False
        self.task_type = 'regression'

        # 为 ESOL 定义标签处理函数 - 返回形状为 [1] 的张量
        self.process_label = lambda idx: torch.FloatTensor([[float(self.labels[idx])]]) # 返回 [1, 1]

        print("\nESOL dataset statistics:")
        print(f"Min solubility: {np.min(self.labels):.2f}")
        print(f"Max solubility: {np.max(self.labels):.2f}")
        print(f"Mean solubility: {np.mean(self.labels):.2f}")
        print(f"Std solubility: {np.std(self.labels):.2f}")

    def _process_freesolv(self, df):
        """处理 FreeSolv 数据集（单目标回归）"""
        required_cols = ['smiles', 'expt']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"FreeSolv dataset must contain columns: {required_cols}")

        self.smiles_list = df['smiles'].values
        self.labels = df['expt'].values.astype(float)
        self.label_cols = ['expt']
        self.is_multi_label = False
        self.task_type = 'regression'

        # 为 FreeSolv 定义标签处理函数 - 返回形状为 [1] 的张量
        self.process_label = lambda idx: torch.FloatTensor([[float(self.labels[idx])]]) # 返回 [1, 1]

        print("\nFreeSolv dataset statistics:")
        print(f"Min expt: {np.min(self.labels):.2f}")
        print(f"Max expt: {np.max(self.labels):.2f}")
        print(f"Mean expt: {np.mean(self.labels):.2f}")
        print(f"Std expt: {np.std(self.labels):.2f}")

    def _process_lipophilicity(self, df):
        """处理 Lipophilicity 数据集（单目标回归）"""
        required_cols = ['cmpd_chemblid', 'exp', 'smiles']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Lipophilicity dataset must contain columns: {required_cols}")

        self.smiles_list = df['smiles'].values
        self.labels = df['exp'].values.astype(float)
        self.label_cols = ['exp']
        self.is_multi_label = False
        self.task_type = 'regression'

        # 为 Lipophilicity 定义标签处理函数 - 返回形状为 [1] 的张量
        self.process_label = lambda idx: torch.FloatTensor([[float(self.labels[idx])]]) # 返回 [1, 1]

        print("\nLipophilicity dataset statistics:")
        print(f"Min exp: {np.min(self.labels):.2f}")
        print(f"Max exp: {np.max(self.labels):.2f}")
        print(f"Mean exp: {np.mean(self.labels):.2f}")
        print(f"Std exp: {np.std(self.labels):.2f}")

    def _process_qm8(self, df):
        """处理 QM8 数据集（多目标回归）"""
        required_cols = ['smiles']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("QM8 dataset must contain a 'smiles' column")

        self.smiles_list = df['smiles'].values
        label_cols = [col for col in df.columns if col != 'smiles']
        self.label_cols = label_cols
        self.labels = df[label_cols].values.astype(float)
        self.is_multi_label = True
        self.task_type = 'regression'

        # 为 QM8 定义标签处理函数 - 返回形状为 [num_targets] 的张量
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0) # 返回 [1, num_targets]

        print("\nQM8 dataset statistics:")
        print(f"Label columns: {label_cols}")
        print(f"Min values: {np.min(self.labels, axis=0)}")
        print(f"Max values: {np.max(self.labels, axis=0)}")
        print(f"Mean values: {np.mean(self.labels, axis=0)}")
        print(f"Std values: {np.std(self.labels, axis=0)}")

    def _process_qm9(self, df):
        """处理 QM9 数据集（多目标回归）"""
        required_cols = ['smiles']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("QM9 dataset must contain a 'smiles' column")

        self.smiles_list = df['smiles'].values
        label_cols = [col for col in df.columns if col not in ['mol_id', 'smiles']]
        self.label_cols = label_cols
        self.labels = df[label_cols].values.astype(float)
        self.is_multi_label = True
        self.task_type = 'regression'

        # 为 QM9 定义标签处理函数 - 返回形状为 [num_targets] 的张量
        self.process_label = lambda idx: torch.FloatTensor(self.labels[idx]).unsqueeze(0) # 返回 [1, num_targets]

        print("\nQM9 dataset statistics:")
        print(f"Label columns: {label_cols}")
        print(f"Min values: {np.min(self.labels, axis=0)}")
        print(f"Max values: {np.max(self.labels, axis=0)}")
        print(f"Mean values: {np.mean(self.labels, axis=0)}")
        print(f"Std values: {np.std(self.labels, axis=0)}")

    def _check_data_validity(self):
        """检查处理后的数据有效性"""
        if len(self.processed_data) == 0:
            print("Warning: No processed data to check validity.")
            return

        if self.task_type == 'classification':
            invalid_tasks = []

            if self.is_multi_label:
                # 多标签分类
                # 确保 num_labels > 0
                if not hasattr(self, 'label_cols') or not self.label_cols:
                     print("Warning: Cannot check validity, label_cols not defined.")
                     return
                num_tasks = len(self.label_cols)
                if num_tasks == 0:
                     print("Warning: Cannot check validity, num_tasks is zero.")
                     return

                for task_idx in range(num_tasks):
                    task_labels = []
                    for data in self.processed_data:
                        # 检查 y 是否存在且形状至少为 [1, num_tasks]
                        if hasattr(data, 'y') and data.y is not None and data.y.dim() == 2 and data.y.shape[0] == 1 and data.y.shape[1] > task_idx:
                             # --- 修改这里 ---
                            try:
                                # 使用正确的二维索引 [0, task_idx]
                                label_value = data.y[0, task_idx].item()
                                task_labels.append(label_value)
                            except IndexError:
                                print(f"Warning: IndexError accessing data.y[0, {task_idx}] for data with y shape {data.y.shape}. Skipping.")
                            except Exception as e:
                                print(f"Warning: Error accessing label for task {task_idx}: {e}. Skipping.")
                        # else: # 可选：添加警告信息
                        #    print(f"Warning: Skipping data with invalid y for task {task_idx}. Shape: {getattr(data, 'y', 'None')}")

                    # 过滤掉NaN值 (在转换为numpy之前确保列表不为空)
                    if not task_labels:
                        print(f"Warning: No valid labels found for task {task_idx} ({self.label_cols[task_idx]}).")
                        invalid_tasks.append((self.label_cols[task_idx], [])) # 记录为空
                        continue

                    valid_labels = [v for v in task_labels if not np.isnan(v)]
                    if not valid_labels:
                         print(f"Warning: All labels are NaN for task {task_idx} ({self.label_cols[task_idx]}) after filtering.")
                         invalid_tasks.append((self.label_cols[task_idx], [])) # 记录为空
                         continue

                    unique = np.unique(valid_labels)
                    if len(unique) < 2:
                        invalid_tasks.append((self.label_cols[task_idx], unique))
            else:
                # 单标签分类
                try:
                    label_values = []
                    for data in self.processed_data:
                        # 检查 y 是否存在且形状为 [1, 1]
                        if hasattr(data, 'y') and data.y is not None and data.y.dim() == 2 and data.y.shape == (1, 1):
                            # --- 修改这里 ---
                            try:
                                # 使用正确的二维索引 [0, 0]
                                label_value = data.y[0, 0].item()
                                label_values.append(label_value)
                            except IndexError:
                                print(f"Warning: IndexError accessing data.y[0, 0] for data with y shape {data.y.shape}. Skipping.")
                            except Exception as e:
                                print(f"Warning: Error accessing single label: {e}. Skipping.")
                        # else: # 可选：添加警告信息
                        #    print(f"Warning: Skipping data with invalid y for single-label task. Shape: {getattr(data, 'y', 'None')}")


                    if not label_values:
                         print("Warning: No valid labels found for single-label task.")
                         invalid_tasks.append(('main_label', []))
                    else:
                        unique = np.unique(label_values)
                        if len(unique) < 2:
                            invalid_tasks.append(('main_label', unique))

                except Exception as e:
                    print(f"Error analyzing single-label distribution: {str(e)}")

            if invalid_tasks:
                print("\nWARNING: Found tasks with only one class (or no valid labels):")
                for task, classes in invalid_tasks:
                    if classes:
                        print(f"  Task {task}: classes {classes}")
                    else:
                        print(f"  Task {task}: No valid labels found.")
                print("This may cause evaluation metrics like AUC/AUPR to fail or be unreliable!")

    def get_original_label_value(self, encoded_value):
        """将编码标签转换回原始值"""
        if hasattr(self, 'label_encoding') and 'encoded_to_original' in self.label_encoding:
            return self.label_encoding['encoded_to_original'].get(encoded_value, encoded_value)
        return encoded_value

    def is_missing_label(self, encoded_value):
        """检查是否为缺失标签"""
        if hasattr(self, 'label_encoding'):
            return encoded_value == self.label_encoding['missing']
        return False

    def _print_label_statistics(self):
        """打印标签统计信息"""
        try:
            if self.is_multi_label and self.dataset_name in ['sider', 'tox21', 'toxcast']:
                print("\nLabel distribution in processed data:")
                # 多标签多分类任务
                for idx, col in enumerate(self.label_cols):
                    if idx < self.labels.shape[1]:
                        label_values = self.labels[:, idx]
                        unique, counts = np.unique(label_values, return_counts=True)
                        print(f"{col}:")
                        for u, c in zip(unique, counts):
                            print(f"  Label {u}: {c} samples ({c / len(label_values) * 100:.2f}%)")
            elif self.dataset_name in ['qm8', 'qm9']:
                # 多目标回归任务
                try:
                    processed_labels_array = []
                    for data in self.processed_data:
                        if data.y.numel() > 0:
                            processed_labels_array.append(data.y.cpu().numpy())

                    processed_labels_array = np.array(processed_labels_array)

                    print("\nProcessed label statistics (per dimension):")
                    for i, col in enumerate(self.label_cols):
                        if i < processed_labels_array.shape[1]:
                            col_values = processed_labels_array[:, i]
                            print(
                                f"{col}: Min={col_values.min():.4f}, Max={col_values.max():.4f}, Mean={col_values.mean():.4f}, Std={col_values.std():.4f}")
                except Exception as e:
                    print(f"Error computing label statistics: {str(e)}")
            else:
                # 单标签任务
                try:
                    processed_labels = []
                    for data in self.processed_data:
                        if data.y.numel() >= 1:
                            processed_labels.append(data.y[0].item())

                    unique_labels = np.unique(processed_labels)
                    print("\nLabel distribution in processed data:")
                    for label in unique_labels:
                        count = sum(1 for y in processed_labels if y == label)
                        print(f"Label {label}: {count} samples ({count / len(processed_labels) * 100:.2f}%)")
                except Exception as e:
                    print(f"Error computing label distribution: {str(e)}")
        except Exception as e:
            print(f"Error in label statistics calculation: {str(e)}")

    def _save_hyperedge_attrs(self):
        """保存超边属性到缓存文件"""
        # 更新超边属性字典，确保包含所有已处理数据的超边属性
        for data in self.processed_data:
            if hasattr(data, 'smiles') and data.smiles and hasattr(data,
                                                                   'hyperedge_attr') and data.hyperedge_attr is not None:
                # 添加类型检查
                if isinstance(data.hyperedge_attr, torch.Tensor):
                    self.hyperedge_attrs[data.smiles] = data.hyperedge_attr.cpu().numpy()
                else:
                    # 如果已经是numpy数组，直接存储
                    self.hyperedge_attrs[data.smiles] = data.hyperedge_attr

        hyperedge_attrs_file = os.path.join(self.cache_dir, "hyperedge_attrs.pt")
        torch.save(self.hyperedge_attrs, hyperedge_attrs_file)
        print(f"Saved hyperedge attributes for {len(self.hyperedge_attrs)} molecules")

    def _get_cache_filename(self, idx):
        return os.path.join(self.cache_dir, f"mol_{idx}.pt")

    def _save_metadata(self):
        # 检测数据集是否包含超边属性
        has_hyperedge_attr = False
        if self.processed_data and len(self.processed_data) > 0:
            sample_item = self.processed_data[0]
            has_hyperedge_attr = hasattr(sample_item, 'hyperedge_attr') and sample_item.hyperedge_attr is not None
        metadata = {
            'dataset_name': self.dataset_name,
            'task_type': self.task_type,
            'num_classes': self._get_num_classes(),
            'label_cols': self.label_cols,
            'is_multi_label': self.is_multi_label,
            'valid_indices': self.valid_indices,
            'has_hyperedge_attr': has_hyperedge_attr,
            'label_encoding': getattr(self, 'label_encoding', {})  # 保存标签编码信息
        }
        torch.save(metadata, self.metadata_file)

    def _load_metadata(self):
        if os.path.exists(self.metadata_file):
            try:
                metadata = torch.load(self.metadata_file)
                if metadata.get('dataset_name') == self.dataset_name:
                    self.label_cols = metadata.get('label_cols', [])
                    self.is_multi_label = metadata.get('is_multi_label', False)
                    self.valid_indices = metadata.get('valid_indices', [])
                    self.task_type = metadata.get('task_type', self.task_type)
                    # 保存超边属性标志（用于其他地方判断）
                    self.has_hyperedge_attr = metadata.get('has_hyperedge_attr', False)
                    # 加载标签编码信息
                    self.label_encoding = metadata.get('label_encoding', {
                        'positive': 1,
                        'negative': -1,
                        'missing': 0,
                        'original_to_encoded': {},
                        'encoded_to_original': {}
                    })
                    return True
                return False
            except Exception as e:
                print(f"Error loading metadata: {str(e)}")
                return False
        return False

    def _try_load_cache(self):
        if not self._load_metadata():
            return False
        cache_files = [f for f in os.listdir(self.cache_dir) if
                       f.endswith(".pt") and not f == "metadata.pt" and not f == "hyperedge_attrs.pt"]
        if not cache_files:
            return False
        print("\nChecking cached data...")
        # 首先尝试加载超边属性字典（用作备份）
        self.hyperedge_attrs = {}
        hyperedge_attrs_file = os.path.join(self.cache_dir, "hyperedge_attrs.pt")
        if os.path.exists(hyperedge_attrs_file):
            try:
                self.hyperedge_attrs = torch.load(hyperedge_attrs_file)
                print(f"Loaded hyperedge attributes dictionary for {len(self.hyperedge_attrs)} molecules")
            except Exception as e:
                print(f"Error loading hyperedge attributes: {e}")
                self.hyperedge_attrs = {}

        self.processed_data = []
        self.valid_smiles = []
        success = 0
        failures = 0
        # 确定最大索引
        max_idx = -1
        for f in cache_files:
            try:
                idx = int(f.split('_')[1].split('.')[0])
                max_idx = max(max_idx, idx)
            except:
                continue
        # 加载缓存数据
        for idx in tqdm(range(max_idx + 1)):
            cache_file = self._get_cache_filename(idx)
            if os.path.exists(cache_file):
                try:
                    data = torch.load(cache_file)

                    # 显式验证基础属性
                    if not hasattr(data, 'x') or data.x is None or data.x.shape[0] == 0:
                        print(f"Skipping invalid cache file (bad x): {cache_file}")
                        failures += 1
                        continue
                    if not hasattr(data, 'edge_index') or data.edge_index is None:
                        print(f"Skipping invalid cache file (bad edge_index): {cache_file}")
                        failures += 1
                        continue
                    if not hasattr(data, 'y') or data.y is None:
                        print(f"Skipping invalid cache file (bad y): {cache_file}")
                        failures += 1
                        continue

                    # 获取SMILES和超边属性（如果存在）
                    smiles = getattr(data, 'smiles', None)
                    hyperedge_attr = getattr(data, 'hyperedge_attr', None)

                    # 如果数据没有超边属性但SMILES在字典中存在，尝试从字典恢复
                    if hyperedge_attr is None and smiles is not None and smiles in self.hyperedge_attrs:
                        hyperedge_attr_np = self.hyperedge_attrs[smiles]
                        hyperedge_attr = torch.FloatTensor(hyperedge_attr_np)
                        print(f"Recovered hyperedge attributes for molecule with SMILES {smiles}")

                    # 创建规范化的MoleculeData对象
                    cleaned_data = MoleculeData(
                        x=data.x.clone(),
                        edge_index=data.edge_index.clone(),
                        y=data.y.clone(),
                        hyperedge_attr=hyperedge_attr.clone() if hyperedge_attr is not None else None,
                        smiles=smiles
                    )
                    self.processed_data.append(cleaned_data)
                    if smiles is not None:
                        self.valid_smiles.append(smiles)
                    success += 1

                except Exception as e:
                    print(f"Error loading cache file {cache_file}: {str(e)}")
                    failures += 1
        print(f"Loaded {success}/{len(cache_files)} valid cached entries ({failures} failures)")
        return success > 0

    def _save_cache(self):
        print("\nSaving processed data to cache...")
        os.makedirs(self.cache_dir, exist_ok=True)
        for idx, data in enumerate(tqdm(self.processed_data)):
            try:
                # 确保数据在CPU上并复制到新对象
                smiles = getattr(data, 'smiles', None)
                hyperedge_attr = getattr(data, 'hyperedge_attr', None)

                # 创建包含所有属性的新MoleculeData对象
                data_cpu = MoleculeData(
                    x=data.x.cpu(),
                    edge_index=data.edge_index.cpu(),
                    y=data.y.cpu(),
                    hyperedge_attr=hyperedge_attr.cpu() if hyperedge_attr is not None else None,
                    smiles=smiles
                )
                torch.save(data_cpu, self._get_cache_filename(idx))
            except Exception as e:
                print(f"Error saving {idx}: {str(e)}")
        # 仍然保存超边属性字典作为备份（可选）
        self._save_hyperedge_attrs()
        self._save_metadata()
        print(f"Saved metadata to {self.metadata_file}")

    def len(self):
        return len(self.processed_data)

    def get(self, idx):
        return self.processed_data[idx]

    @property
    def num_node_features(self):
        if len(self.processed_data) == 0:
            raise ValueError("Dataset is empty!")
        return self.processed_data[0].x.size(1)

    def _get_num_classes(self):
        """获取分类数量或回归目标数量"""
        if self.task_type == 'regression':
            if self.is_multi_label:
                return len(self.label_cols)
            else:
                return 1
        elif self.is_multi_label:
            return len(self.label_cols)
        else:
            return 1

    @property
    def num_classes(self):
        if len(self.processed_data) == 0:
            raise ValueError("Dataset is empty!")
        return self._get_num_classes()
