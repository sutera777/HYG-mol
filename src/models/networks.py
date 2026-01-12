 # HYG-mol/src/models/networks.py
 
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv, global_mean_pool
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_add, scatter_mean
 
class HyperGraphNet(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes, task_type="classification",
                 is_multi_label=False):
        super(HyperGraphNet, self).__init__()
        self.task_type = task_type
        self.is_multi_label = is_multi_label
 
        self.conv1 = HypergraphConv(num_node_features, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, hidden_channels * 2)
        self.conv3 = HypergraphConv(hidden_channels * 2, hidden_channels)
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
 
 
        if self.task_type == "regression":
            if num_classes == 1:
                self.lin2 = nn.Linear(hidden_channels // 2, 1)
            else:
                self.lin2 = nn.Linear(hidden_channels // 2, num_classes)
        else:  
            if self.is_multi_label:
                self.lin2 = nn.Linear(hidden_channels // 2, num_classes)
            else:
                self.lin2 = nn.Linear(hidden_channels // 2, 1)
 
        self.dropout = nn.Dropout(0.2)
 
    def forward(self, x, hyperedge_index, batch):
 
        if x is None or hyperedge_index is None or batch is None:
            raise ValueError("Input tensors cannot be None")
 
 
        if isinstance(hyperedge_index, (list, np.ndarray)):
            hyperedge_index = torch.LongTensor(hyperedge_index).to(x.device)
 
 
        if x.device != hyperedge_index.device:
            hyperedge_index = hyperedge_index.to(x.device)
        if x.device != batch.device:
            batch = batch.to(x.device)
 
 
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input tensor, got shape {x.shape}")
        if x.size(0) == 0:
            raise ValueError("Empty feature matrix")
        if hyperedge_index.dim() != 2 or hyperedge_index.size(0) != 2:
            raise ValueError(f"Expected hyperedge_index with shape [2, E], got {hyperedge_index.shape}")
 
        if batch.max().item() >= x.size(0):
            print(f"Warning: Batch index {batch.max().item()} out of range for tensor size {x.size(0)}")
 
            batch = torch.clamp(batch, 0, x.size(0) - 1)
 
        try:
 
            x = self.conv1(x, hyperedge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv2(x, hyperedge_index)
            x = F.relu(x)
            x = self.dropout(x)
            x = self.conv3(x, hyperedge_index)
            x = F.relu(x)
            x = global_mean_pool(x, batch)
            x = self.lin1(x)
            x = F.relu(x)
            x = self.dropout(x)
 
            x = self.lin2(x)
 
 
            if self.task_type == "regression":
                if self.is_multi_label:
                    pass  
                else:
                    if x.dim() == 1:
                        x = x.unsqueeze(1)
            else: 
                if self.is_multi_label:
                    pass  
                else:
                    if x.dim() == 1:
                        x = x.unsqueeze(1)
 
            return x
        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            print(f"Input shapes - x: {x.shape}, hyperedge_index: {hyperedge_index.shape}, batch: {batch.shape}")
            raise
 
 
 
class CustomHypergraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim, out_channels,
                 heads=4, concat=True, negative_slope=0.2, dropout=0.0):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = nn.Dropout(dropout)
 
 
        self.node_lin = nn.Linear(node_dim, heads * out_channels, bias=False)
        self.edge_lin = nn.Linear(edge_dim, heads * out_channels, bias=False)
 
 
        self.attn_l = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.attn_r = nn.Parameter(torch.Tensor(1, heads, out_channels))
 
 
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
 
 
        self._attention_weights = {}  # { 'layerX': alpha_tensor }
 
 
        nn.init.xavier_uniform_(self.node_lin.weight)
        nn.init.xavier_uniform_(self.edge_lin.weight)
        nn.init.xavier_uniform_(self.attn_l)
        nn.init.xavier_uniform_(self.attn_r)
        nn.init.zeros_(self.bias)
 
    def forward(self, x, hyperedge_index, hyperedge_attr=None,
                store_attention=False, layer_name=None):
   
        device = x.device
        num_nodes = x.size(0)
 
 
        if hyperedge_index.size(1) == 0:
 
 
            output_dim = self.heads * self.out_channels if self.concat else self.out_channels
            return self.node_lin(x).view(num_nodes, output_dim) + self.bias
 
        node_idx, edge_idx_incidence = hyperedge_index  
 
 
 
        num_edges = int(hyperedge_index[1].max().item() + 1) if hyperedge_index.numel() > 0 else 0
        if num_edges == 0:  
            output_dim = self.heads * self.out_channels if self.concat else self.out_channels
            return self.node_lin(x).view(num_nodes, output_dim) + self.bias
 
 
        if hyperedge_attr is None:
            hyperedge_attr = torch.zeros((num_edges, self.edge_dim), device=device)
        else:
 
            if hyperedge_attr.size(0) != num_edges:
 
                if hyperedge_attr.size(0) < num_edges:
 
                    new_attr = torch.zeros((num_edges, hyperedge_attr.size(1)), device=device)
                    new_attr[:hyperedge_attr.size(0)] = hyperedge_attr
                    hyperedge_attr = new_attr
                else:
 
                    hyperedge_attr = hyperedge_attr[:num_edges]
 
            if hyperedge_attr.size(1) != self.edge_dim:
                corrected = torch.zeros((hyperedge_attr.size(0), self.edge_dim), device=device)
                min_dim = min(self.edge_dim, hyperedge_attr.size(1))
                corrected[:, :min_dim] = hyperedge_attr[:, :min_dim]
                hyperedge_attr = corrected
 
 
        node_feats = self.node_lin(x).view(-1, self.heads, self.out_channels)  
        edge_feats = self.edge_lin(hyperedge_attr).view(-1, self.heads,
                                                        self.out_channels)  
 
 
        alpha_l = (node_feats * self.attn_l).sum(dim=-1)  
        alpha_r = (edge_feats * self.attn_r).sum(dim=-1)  
 
 
        alpha_l_selected = alpha_l[node_idx]  
        alpha_r_selected = alpha_r[edge_idx_incidence]  
        alpha = alpha_l_selected + alpha_r_selected  
 
 
        alpha = pyg_softmax(alpha, edge_idx_incidence)  
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.dropout(alpha)
 
 
        m_node2edge = node_feats[node_idx] * alpha.unsqueeze(-1)  
 
        edge_aggr = scatter_add(m_node2edge, edge_idx_incidence, dim=0,
                                dim_size=num_edges)  
        edge_aggr = edge_aggr + edge_feats  
 
 
 
        m_edge2node = edge_aggr[edge_idx_incidence] * alpha.unsqueeze(-1)  
 
        node_aggr = scatter_add(m_edge2node, node_idx, dim=0, dim_size=num_nodes)  
 
 
        if self.concat:
            out = node_aggr.view(num_nodes, self.heads * self.out_channels)
        else:
            out = node_aggr.mean(dim=1)  
 
        out = out + self.bias
 
 
        if store_attention and layer_name is not None:
            alpha_per_incidence = alpha.mean(dim=1)
            aggregated_hyperedge_attention = scatter_mean(alpha_per_incidence, edge_idx_incidence, dim=0,
                                                          dim_size=num_edges)
            self._attention_weights[layer_name] = aggregated_hyperedge_attention.detach().cpu()
 
        return out
 
 
class AttentionHyperGraphNet(nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_classes,
                 task_type="classification", is_multi_label=False,
                 attention_mode='node', heads=4, dropout=0.2,
                 hyperedge_dim=5):
        super().__init__()
        self.task_type = task_type
        self.is_multi_label = is_multi_label
        self.hyperedge_dim = hyperedge_dim
        self.hidden_channels = hidden_channels
 
 
        self.hyperedge_encoder = nn.Sequential(
            nn.Linear(hyperedge_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
 
 
        self.conv1 = CustomHypergraphConv(
            node_dim=num_node_features,
            edge_dim=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=dropout
        )
 
        self.conv2 = CustomHypergraphConv(
            node_dim=hidden_channels,
            edge_dim=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=dropout
        )
 
        self.conv3 = CustomHypergraphConv(
            node_dim=hidden_channels,
            edge_dim=hidden_channels,
            out_channels=hidden_channels // heads,
            heads=heads,
            concat=True,
            negative_slope=0.2,
            dropout=dropout
        )
 
 
        self.lin1 = nn.Linear(hidden_channels, hidden_channels // 2)
 
        if self.task_type == "regression":
            self.lin2 = nn.Linear(hidden_channels // 2, num_classes if num_classes > 1 else 1)
        else:
 
            if self.is_multi_label:
                self.lin2 = nn.Linear(hidden_channels // 2, num_classes)
            else:
                self.lin2 = nn.Linear(hidden_channels // 2, 1 if num_classes == 1 else num_classes)
 
        self.dropout = nn.Dropout(dropout)
 
 
        self.attention_weights_dict = {}
 
    def reset_attention_weights(self):

        self.attention_weights_dict = {}
 
        self.conv1._attention_weights = {}
        self.conv2._attention_weights = {}
        self.conv3._attention_weights = {}
 
    def get_attention_weights(self):

        return self.attention_weights_dict
 
    def forward(self, x, hyperedge_index, batch, store_attention=False, hyperedge_attr=None):

        if hyperedge_attr is None:
            edge_idx_max = hyperedge_index[1].max().item() if hyperedge_index.numel() > 0 else 0
            num_edges = edge_idx_max + 1
            hyperedge_attr = torch.zeros((num_edges, self.hyperedge_dim), device=x.device)
        else:
 
            hyperedge_attr = self.hyperedge_encoder(hyperedge_attr)
 
 
        x1 = self.conv1(
            x, hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            store_attention=store_attention,
            layer_name='layer1'
        )
        x1 = F.relu(x1)
        x1 = self.dropout(x1)
 
 
        x2 = self.conv2(
            x1, hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            store_attention=store_attention,
            layer_name='layer2'
        )
        x2 = F.relu(x2)
        x2 = self.dropout(x2)
 
 
        x3 = self.conv3(
            x2, hyperedge_index,
            hyperedge_attr=hyperedge_attr,
            store_attention=store_attention,
            layer_name='layer3'
        )
        x3 = F.relu(x3)
 
 
        if store_attention:
 
            if 'layer1' in self.conv1._attention_weights:
                self.attention_weights_dict['layer1'] = self.conv1._attention_weights['layer1']
            if 'layer2' in self.conv2._attention_weights:
                self.attention_weights_dict['layer2'] = self.conv2._attention_weights['layer2']
            if 'layer3' in self.conv3._attention_weights:
                self.attention_weights_dict['layer3'] = self.conv3._attention_weights['layer3']
 
 
        x_pool = global_mean_pool(x3, batch)
 
 
        x_pool = self.lin1(x_pool)
        x_pool = F.relu(x_pool)
        x_pool = self.dropout(x_pool)
        out = self.lin2(x_pool)
 
        return out