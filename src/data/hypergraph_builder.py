# HYG-mol/src/data/hypergraph_builder.py
import warnings
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors, Crippen, MolSurf
from rdkit.Chem.Scaffolds import MurckoScaffold


class MoleculeHypergraph:
    def __init__(self):

        self.node_features = []  
        self.hyperedge_index = None  
        self.mol = None  
        self.hyperedges = [] 
        self.hyperedge_labels = []  


        if self.feature_type != "traditional_only":
            model_path = "/home_nfs/qi.yang/PycharmProjects/pythonProject/chemBERTaModels"
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            self.chemberta_model = AutoModel.from_pretrained(model_path).to('cpu')
        else:
            self.tokenizer = None
            self.chemberta_model = None

    def build_from_mol(self, mol):
        self.mol = mol
        num_atoms = mol.GetNumAtoms()

        functional_groups, fg_labels = self.get_functional_groups(mol)
        rings, ring_labels = self.get_ring_structures(mol)
        special_structures, special_labels = self.get_special_structures(mol)
        hyperedges = []  
        hyperedge_labels = []  

        def add_valid_hyperedge(edge, label):

            valid_nodes = [node_id for node_id in edge if 0 <= node_id < num_atoms]
            if valid_nodes: 

                is_duplicate = False
                sorted_valid_nodes_tuple = tuple(sorted(valid_nodes))
                for existing_edge_tuple, existing_label in zip(map(lambda e: tuple(sorted(e)), hyperedges),
                                                               hyperedge_labels):
                    if existing_edge_tuple == sorted_valid_nodes_tuple and existing_label == label:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    hyperedges.append(valid_nodes)
                    hyperedge_labels.append(label)


        for edge, label in zip(functional_groups, fg_labels):
            add_valid_hyperedge(edge, label)
        for edge, label in zip(rings, ring_labels):
            add_valid_hyperedge(edge, label)
        for edge, label in zip(special_structures, special_labels):
            add_valid_hyperedge(edge, label)
        ring_atoms = set()
        atom_to_ring = {}

        for ring_id, ring_node_indices in enumerate(rings): 
            ring_atoms.update(ring_node_indices)
            for atom_idx in ring_node_indices:
                atom_to_ring.setdefault(atom_idx, set()).add(ring_id)
        for bond in mol.GetBonds():
            begin_idx = bond.GetBeginAtom().GetIdx()
            end_idx = bond.GetEndAtom().GetIdx()
            if begin_idx >= num_atoms or end_idx >= num_atoms:
                continue

            begin_in_ring = begin_idx in ring_atoms
            end_in_ring = end_idx in ring_atoms
            if begin_in_ring and end_in_ring:
                common_rings = atom_to_ring.get(begin_idx, set()) & atom_to_ring.get(end_idx, set())
                if not common_rings:  
                    add_valid_hyperedge([begin_idx, end_idx], 'RingConnector')

            elif begin_in_ring != end_in_ring:  
                add_valid_hyperedge([begin_idx, end_idx], 'RingSubstituent')
        all_atoms_in_hyperedges = set()
        for he in hyperedges:
            all_atoms_in_hyperedges.update(he)

        excluded_atoms = set(range(num_atoms)) - all_atoms_in_hyperedges
        for atom_idx in list(excluded_atoms):
            atom = mol.GetAtomWithIdx(atom_idx)
            neighbor_indices = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() < num_atoms]
            if neighbor_indices:
                current_group = sorted([atom_idx] + neighbor_indices)
                is_new_group = True
                for existing_edge, existing_label in zip(hyperedges, hyperedge_labels):
                    if existing_label == 'IsolatedAtomGroup' and sorted(existing_edge) == current_group:
                        is_new_group = False
                        break
                if is_new_group:
                    add_valid_hyperedge(current_group, 'IsolatedAtomGroup')
            else:  
                add_valid_hyperedge([atom_idx], 'SingleAtom')

        try:

            murcko_mol_full = MurckoScaffold.GetScaffoldForMol(mol)
            if murcko_mol_full.GetNumAtoms() > 0:
                match_indices_full = mol.GetSubstructMatch(murcko_mol_full)
                if match_indices_full:  
                    add_valid_hyperedge(list(match_indices_full), 'MurckoScaffoldFull')
        
            if murcko_mol_full.GetNumAtoms() > 0:
     
                murcko_mol_generic = MurckoScaffold.MakeScaffoldGeneric(murcko_mol_full)
                if murcko_mol_generic.GetNumAtoms() > 0:
                    match_indices_generic = mol.GetSubstructMatch(murcko_mol_generic)
                    if match_indices_generic:
        
                        add_valid_hyperedge(list(match_indices_generic), 'MurckoScaffoldCore')
   
        except ImportError:

            print(
                "Warning: RDKit MurckoScaffold module not available. Skipping Murcko scaffold hyperedges.")  
        except Exception as e:
   
            print(f"Warning: Error processing Murcko scaffold for molecule {Chem.MolToSmiles(mol)}: {e}")

        self.build_node_features(mol) 
        self.build_hyperedge_index(hyperedges, hyperedge_labels)  

        return self

    def build_node_features(self, mol):

        num_atoms = mol.GetNumAtoms()

        model_max_length = 512
        if hasattr(self.chemberta_model, 'config') and hasattr(self.chemberta_model.config, 'max_position_embeddings'):
            model_max_length = self.chemberta_model.config.max_position_embeddings

        mol_id = Chem.MolToSmiles(mol)
        cache_key = f"{self.feature_type}_{mol_id}"

        if hasattr(self, 'feature_cache') and cache_key in self.feature_cache:
            self.node_features = self.feature_cache[cache_key]
            return self.node_features

        if not hasattr(self, 'feature_cache'):
            self.feature_cache = {}

     
        if self.feature_type == "traditional_only":
       
            additional_features = []

      
            mol_weight = Descriptors.MolWt(mol) / 500.0 if hasattr(Descriptors, 'MolWt') else 0.0
            logp = Crippen.MolLogP(mol) / 10.0 if hasattr(Crippen, 'MolLogP') else 0.0
            tpsa = MolSurf.TPSA(mol) / 100.0 if hasattr(MolSurf, 'TPSA') else 0.0
            num_rings = len(mol.GetRingInfo().AtomRings()) / 10.0 if hasattr(mol, 'GetRingInfo') else 0.0

         
            charges = [0.0] * num_atoms
            try:
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp('_GasteigerCharge')
                           if atom.HasProp('_GasteigerCharge') else 0.0
                           for atom in mol.GetAtoms()]
       
                charges = [0.0 if (c is None or np.isnan(c) or np.isinf(c)) else c for c in charges]
            except:
                pass 

            ring_info = mol.GetRingInfo()
 
            for atom_idx in range(num_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]

        
                base_features = [
                    atom.GetAtomicNum() / 100.0,
                    atom.GetDegree() / 4.0,
                    int(atom.GetIsAromatic()),
                    atom.GetFormalCharge() / 8.0,
                    atom.GetNumRadicalElectrons() / 8.0,
                    atom.GetChiralTag() / 10.0,
                    atom.GetHybridization() / 6.0,
                    atom.GetImplicitValence() / 8.0,
                    atom.IsInRing() * 1.0,
                    len(neighbors) / 8.0,
                ]


                hybridization_features = [
                    int(str(atom.GetHybridization()) == "SP"),
                    int(str(atom.GetHybridization()) == "SP2"),
                    int(str(atom.GetHybridization()) == "SP3"),
                ]

  
                atom_type_features = [
                    int(atom.GetAtomicNum() == 1),  # H
                    int(atom.GetAtomicNum() == 6),  # C
                    int(atom.GetAtomicNum() == 7),  # N
                    int(atom.GetAtomicNum() == 8),  # O
                    int(atom.GetAtomicNum() == 9),  # F
                    int(atom.GetAtomicNum() == 15),  # P
                    int(atom.GetAtomicNum() == 16),  # S
                    int(atom.GetAtomicNum() == 17),  # Cl
                    int(atom.GetAtomicNum() == 35),  # Br
                    int(atom.GetAtomicNum() == 53),  # I
                ]

  
                bond_types = [0, 0, 0, 0] 
                for neighbor in neighbors:
                    bond = mol.GetBondBetweenAtoms(atom_idx, neighbor)
                    if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                        bond_types[0] += 1
                    elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        bond_types[1] += 1
                    elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                        bond_types[2] += 1
                    elif bond.GetBondType() == Chem.rdchem.BondType.AROMATIC:
                        bond_types[3] += 1

            
                if sum(bond_types) > 0:
                    bond_types = [b / sum(bond_types) for b in bond_types]

   
                neighbor_features = []
                if neighbors:
                    neighbor_atomic_nums = [mol.GetAtomWithIdx(n).GetAtomicNum() for n in neighbors]
                    neighbor_features = [
                        sum(1 for n in neighbor_atomic_nums if n == 6) / len(neighbors),  
                        sum(1 for n in neighbor_atomic_nums if n == 7) / len(neighbors),  
                        sum(1 for n in neighbor_atomic_nums if n == 8) / len(neighbors), 
                        sum(1 for n in neighbor_atomic_nums if n == 9 or n == 17 or n == 35 or n == 53) / len(neighbors)
               
                    ]
                else:
                    neighbor_features = [0.0] * 4

 
                additional_chemical_features = [
                    charges[atom_idx],  
                    atom.GetTotalNumHs() / 4.0,  
                    int(ring_info.IsAtomInRingOfSize(atom_idx, 3)),  
                    int(ring_info.IsAtomInRingOfSize(atom_idx, 5)), 
                    int(ring_info.IsAtomInRingOfSize(atom_idx, 6)),  
                ]


                global_mol_features = [mol_weight, logp, tpsa, num_rings]


                atom_features = (
                        base_features +
                        hybridization_features +
                        atom_type_features +
                        bond_types +
                        neighbor_features +
                        additional_chemical_features +
                        global_mol_features
                )

                additional_features.append(atom_features)

            self.node_features = np.array(additional_features, dtype=np.float32)

        elif self.feature_type == "chemberta_only":

            original_smiles = Chem.MolToSmiles(mol)


            atom_mapped_mol = Chem.AddHs(mol)
            for atom in atom_mapped_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)

            inputs = self.tokenizer(
                original_smiles,
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
                padding="max_length"
            )

            inputs = {k: v.to('cpu') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.chemberta_model(**inputs)

            hidden_states = outputs.last_hidden_state.squeeze(0)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            token_to_char = {}
            smiles_tokens = self.tokenizer.tokenize(original_smiles)
            current_pos = 0

            for i, token in enumerate(smiles_tokens):
                token_to_char[i] = current_pos

                clean_token = token.replace('#', '').replace('Ġ', '')
                current_pos += len(clean_token)


            token_to_atom_map = {}
            current_atom_idx = 0
            atom_to_token_map = {}  

            for token_idx, token in enumerate(tokens):
                if token == self.tokenizer.cls_token or \
                        token == self.tokenizer.sep_token or \
                        token == self.tokenizer.pad_token:
                    continue

                if token.startswith('##'):

                    if token_idx > 0 and token_idx - 1 in token_to_atom_map:
                        token_to_atom_map[token_idx] = token_to_atom_map[token_idx - 1]
                    continue


                atom_tokens = ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'Cl', 'Br', 'I',
                               '[cH]', '[nH]', '[oH]', '[sH]', '[CH]', '[CH2]', '[CH3]', '[NH]', '[NH2]',
                               '[OH]', '[SH]', '[PH]']

                if any(token == t or token.startswith(t) for t in atom_tokens):
                    if current_atom_idx < num_atoms:
                        token_to_atom_map[token_idx] = current_atom_idx


                        if current_atom_idx not in atom_to_token_map:
                            atom_to_token_map[current_atom_idx] = []
                        atom_to_token_map[current_atom_idx].append(token_idx)

                        current_atom_idx += 1
                    else:
                        break


            feature_dim = hidden_states.shape[1]
            atom_features = torch.zeros((num_atoms, feature_dim), device='cpu')


            mapped_atom_indices = set()


            for atom_idx, token_indices in atom_to_token_map.items():
                if atom_idx < num_atoms:
                    valid_tokens = [ti for ti in token_indices if ti < hidden_states.shape[0]]
                    if valid_tokens:

                        weights = torch.ones(len(valid_tokens))
                        for i, ti in enumerate(valid_tokens):
                            if not tokens[ti].startswith('##'): 
                                weights[i] = 2.0

                        weights = weights / weights.sum()  


                        for i, ti in enumerate(valid_tokens):
                            atom_features[atom_idx] += weights[i] * hidden_states[ti]

                        mapped_atom_indices.add(atom_idx)
                    else:
                        print(f"Warning: No valid token indices for atom {atom_idx}")


            for atom_idx in range(num_atoms):
                if atom_idx not in mapped_atom_indices:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in mapped_atom_indices]

                    if neighbors:

                        neighbor_features = atom_features[neighbors]
                        atom_features[atom_idx] = torch.mean(neighbor_features, dim=0)
                        mapped_atom_indices.add(atom_idx)


            unmapped_atoms = [i for i in range(num_atoms) if i not in mapped_atom_indices]
            if unmapped_atoms:

                valid_token_indices = [
                    idx for idx, t in enumerate(tokens)
                    if t != self.tokenizer.cls_token and \
                       t != self.tokenizer.sep_token and \
                       t != self.tokenizer.pad_token and \
                       idx < hidden_states.shape[0]
                ]

                if valid_token_indices:
                    global_features = torch.mean(hidden_states[valid_token_indices], dim=0)
                    for atom_idx in unmapped_atoms:
                        atom_features[atom_idx] = global_features
                else:
                    print(f"Warning: No valid token features for global average in SMILES: {original_smiles[:50]}...")

            self.node_features = atom_features.cpu().numpy()

        else:  

            original_smiles = Chem.MolToSmiles(mol)


            atom_mapped_mol = Chem.AddHs(mol)
            for atom in atom_mapped_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)

            inputs = self.tokenizer(
                original_smiles,
                return_tensors="pt",
                truncation=True,
                max_length=model_max_length,
                padding="max_length"
            )

            inputs = {k: v.to('cpu') for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.chemberta_model(**inputs)

            hidden_states = outputs.last_hidden_state.squeeze(0)
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

            token_to_atom_map = {}
            atom_to_token_map = {}
            current_atom_idx = 0

            for token_idx, token in enumerate(tokens):
                if token == self.tokenizer.cls_token or \
                        token == self.tokenizer.sep_token or \
                        token == self.tokenizer.pad_token:
                    continue

                if token.startswith('##'):

                    if token_idx > 0 and token_idx - 1 in token_to_atom_map:
                        token_to_atom_map[token_idx] = token_to_atom_map[token_idx - 1]
                    continue


                atom_tokens = ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'P', 'p', 'F', 'Cl', 'Br', 'I',
                               '[cH]', '[nH]', '[oH]', '[sH]']

                if any(token == t or token.startswith(t) for t in atom_tokens):
                    if current_atom_idx < num_atoms:
                        token_to_atom_map[token_idx] = current_atom_idx


                        if current_atom_idx not in atom_to_token_map:
                            atom_to_token_map[current_atom_idx] = []
                        atom_to_token_map[current_atom_idx].append(token_idx)

                        current_atom_idx += 1
                    else:
                        break


            feature_dim = hidden_states.shape[1]
            atom_features_tensor = torch.zeros((num_atoms, feature_dim), device='cpu')


            mapped_atom_indices = set()

            for atom_idx, token_indices in atom_to_token_map.items():
                if atom_idx < num_atoms:
                    valid_tokens = [ti for ti in token_indices if ti < hidden_states.shape[0]]
                    if valid_tokens:

                        for ti in valid_tokens:
                            atom_features_tensor[atom_idx] += hidden_states[ti]
                        atom_features_tensor[atom_idx] /= len(valid_tokens)
                        mapped_atom_indices.add(atom_idx)


            for atom_idx in range(num_atoms):
                if atom_idx not in mapped_atom_indices:
                    atom = mol.GetAtomWithIdx(atom_idx)
                    neighbors = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() in mapped_atom_indices]

                    if neighbors:

                        neighbor_features = atom_features_tensor[neighbors]
                        atom_features_tensor[atom_idx] = torch.mean(neighbor_features, dim=0)
                        mapped_atom_indices.add(atom_idx)
                    else:

                        valid_token_indices = [
                            idx for idx, t in enumerate(tokens)
                            if t != self.tokenizer.cls_token and \
                               t != self.tokenizer.sep_token and \
                               t != self.tokenizer.pad_token and \
                               idx < hidden_states.shape[0]
                        ]

                        if valid_token_indices:
                            atom_features_tensor[atom_idx] = torch.mean(hidden_states[valid_token_indices], dim=0)
                        else:
                            print(
                                f"Warning: No valid token features to average for fallback on atom {atom_idx} in SMILES: {original_smiles[:50]}...")

  
            mol_weight = Descriptors.MolWt(mol) / 500.0 if hasattr(Descriptors, 'MolWt') else 0.0
            logp = Crippen.MolLogP(mol) / 10.0 if hasattr(Crippen, 'MolLogP') else 0.0
            tpsa = MolSurf.TPSA(mol) / 100.0 if hasattr(MolSurf, 'TPSA') else 0.0


            charges = [0.0] * num_atoms
            try:
                AllChem.ComputeGasteigerCharges(mol)
                charges = [atom.GetDoubleProp('_GasteigerCharge')
                           if atom.HasProp('_GasteigerCharge') else 0.0
                           for atom in mol.GetAtoms()]
                charges = [0.0 if (c is None or np.isnan(c) or np.isinf(c)) else c for c in charges]
            except:
                pass


            additional_features_list = []

            for atom_idx in range(num_atoms):
                atom = mol.GetAtomWithIdx(atom_idx)
                neighbors = [n.GetIdx() for n in atom.GetNeighbors()]


                base_features = [
                    atom.GetAtomicNum() / 100.0,
                    atom.GetDegree() / 4.0,
                    int(atom.GetIsAromatic()),
                    atom.GetFormalCharge() / 8.0,
                    atom.GetNumRadicalElectrons() / 8.0
                ]

                enhanced_features = [
                    charges[atom_idx],  
                    int(atom.IsInRing()),  
                    atom.GetTotalNumHs() / 4.0,  
                    mol_weight, 
                    logp,  
                    tpsa  
                ]


                atom_additional_features = base_features + enhanced_features
                additional_features_list.append(atom_additional_features)


            additional_features_tensor = torch.tensor(additional_features_list, dtype=torch.float, device='cpu')
            bert_weight = 0.6
            trad_weight = 0.4
            bert_mean = torch.mean(atom_features_tensor, dim=0, keepdim=True)
            bert_std = torch.std(atom_features_tensor, dim=0, keepdim=True) + 1e-8
            atom_features_tensor = (atom_features_tensor - bert_mean) / bert_std

            trad_mean = torch.mean(additional_features_tensor, dim=0, keepdim=True)
            trad_std = torch.std(additional_features_tensor, dim=0, keepdim=True) + 1e-8
            additional_features_tensor = (additional_features_tensor - trad_mean) / trad_std


            atom_features_tensor = bert_weight * atom_features_tensor
            additional_features_tensor = trad_weight * additional_features_tensor


            final_features = torch.cat([atom_features_tensor, additional_features_tensor], dim=1)


            self.node_features = final_features.cpu().numpy()


        if hasattr(self, 'use_nonlinear_features') and self.use_nonlinear_features:
            self.node_features = self._add_nonlinear_interactions(self.node_features)


        self.feature_cache[cache_key] = self.node_features


        assert self.node_features.shape[0] == num_atoms, \
            f"Feature matrix shape {self.node_features.shape} does not match number of atoms {num_atoms}"

        return self.node_features

    def _add_nonlinear_interactions(self, features):


        if features.shape[1] > 30:
            main_features = features[:, :10]
        else:
            main_features = features

        num_samples, num_features = main_features.shape
        interactions = []

        for i in range(min(5, num_features)):
            interactions.append(np.square(main_features[:, i]).reshape(-1, 1))

        count = 0
        for i in range(min(5, num_features)):
            for j in range(i + 1, min(5, num_features)):
                if count < 10: 
                    interactions.append((main_features[:, i] * main_features[:, j]).reshape(-1, 1))
                    count += 1


        if interactions:
            interaction_features = np.hstack(interactions)

            return np.hstack([features, interaction_features])
        else:
            return features

    def build_hyperedge_index(self, hyperedges, hyperedge_labels):
        node_idx = []
        edge_idx = []
        num_nodes = self.mol.GetNumAtoms()
        valid_edges = []
        valid_labels = []

        for edge_id, (hyperedge, label) in enumerate(zip(hyperedges, hyperedge_labels)):

            valid_nodes = [node_id for node_id in hyperedge if 0 <= node_id < num_nodes]
            if len(valid_nodes) > 0:  
                if len(valid_nodes) != len(hyperedge):
                    print(
                        f"Warning: Hyperedge {edge_id} contains invalid nodes, keeping only valid ones: {valid_nodes}")
                valid_edges.append(valid_nodes)
                valid_labels.append(label)
            else:
                print(f"Warning: Hyperedge {edge_id} has no valid nodes, discarding")

        if not valid_edges:
            print("Warning: No valid hyperedges found! Creating a default edge.")

            if num_nodes > 0:
                valid_edges = [[0]]
                valid_labels = ["Default"]
            else:
                raise ValueError("No valid hyperedges could be created for molecule with no atoms")

        self.hyperedges = valid_edges
        self.hyperedge_labels = valid_labels

        for edge_id, hyperedge in enumerate(valid_edges):
            for node_id in hyperedge:
                if 0 <= node_id < num_nodes:
                    node_idx.append(node_id)
                    edge_idx.append(edge_id)


        if not node_idx or not edge_idx:
            print("Warning: Empty hyperedge index after processing. Creating fallback index.")

            if num_nodes > 0:
                node_idx = [0]
                edge_idx = [0]
            else:
                raise ValueError("Cannot create valid hyperedge index for empty molecule")

        self.hyperedge_index = np.array([node_idx, edge_idx])

    def get_functional_groups(self, mol):
        fg_smarts = {
            'Amine': '[NX3;H2,H1,H0;!$(NC=O)]',
            'QuaternaryAmmonium': '[NX4+]',
            'Alcohol': '[OX2H]',
            'Phenol': 'c[OH]',
            'CarboxylicAcid': 'C(=O)[OX1H0-,OX2H1]',
            'Ester': 'C(=O)O[C;!$(C=O)]',
            'Ketone': 'C(=O)[C;!$(C=O)]',
            'Aldehyde': '[CX3H1](=O)[#6]',
            'Ether': '[OD2]([#6])[#6]',
            'Amide': 'C(=O)N',
            'Halogen': '[F,Cl,Br,I]',
            'Sulfonamide': 'S(=O)(=O)N',
            'Thiol': '[#16X2H]',
            'Disulfide': '[#16X2]-[#16X2]',
            'Nitrate': '[N+](=O)[O-]',
            'Cyano': '[C]#N',
            'Azide': '[N]=[N+]=[N-]',
            'Alkyne': '[CX2]#C',
            'Alkene': 'C=C',
            'Phosphate': '[#15](=O)(O)(O)O',
            'Phosphonate': 'P(=O)(O)(O)[C]',
            'Sulfate': 'S(=O)(=O)(O)(O)',
            'Sulfonate': 'S(=O)(=O)O[C]',
            'Sulfoxide': '[#16X3+1][#6]',
            'SulfonicAcid': 'S(=O)(=O)[OH]',
            'Isocyanate': 'N=C=O',
            'Urea': 'N-C(=O)-N',
            'Carbamate': 'O=C(O)N',
            'Imine': '[CX2]=[NX2]',
            'Thioether': '[#16X2][#6]',
            'Epoxide': '[C@H1]1O[C@H1]1',
            'Peroxide': '[OX2][OX2]',
            'BoronicAcid': 'B(O)O',
            'Anhydride': 'C(=O)OC(=O)',
            'Thiocyanate': '[N-]=C=S',
            'Isothiocyanate': 'N=C=S',
            'Oxime': '[CX3](=NO)',
            'Hydrazone': '[CX3](=NN)',
            'Guanidine': 'N=C(N)N',
            'Pyridine': 'n1ccccc1',
            'Pyrazine': 'n1cnccn1',
            'Pyrrole': 'n1cccc1',
            'Imidazole': 'n1c[nH]cc1',
            'Thiazole': 'c1ncsc1',
            'NitroAromatic': '[N](=O)=O[c]',
            'Quinone': 'O=C1C=CC(=O)C=C1',
            'Piperidine': 'N1CCCCC1',
            'Pyrrolidine': 'N1CCCC1',
            'Morpholine': 'O1CCNCC1',
            'TertiaryButylEster': 'C(=O)OC(C)(C)C',  
            'IsopropylEster': 'C(=O)O[CH](C)C',
            'MethylEster': 'C(=O)OC',
            'EthylEster': 'C(=O)OCC',  
            'PropylEster': 'C(=O)OCCC',  
            'HexylEster': 'C(=O)OCCCCCC',  
            'OctylEster': 'C(=O)OCCCCCCCC', 
            'Fluorine': '[F]',  
            'Trifluoromethyl': 'C(F)(F)F',  
            'Adamantane': 'C1C2CC3CC(C1)CC(C2)C3', 
            'Cyclohexyl': 'C1CCCCC1', 
            'Sulfone': 'S(=O)(=O)[C,N,O]',  
            'PhosphineOxide': 'P(=O)[C,N,O]', 
            'Benzene': 'c1ccccc1',  
            'Naphthalene': 'c1cccc2ccccc12',  
            'NitroAromatic': '[$(c-[N+](=O)[O-]),$(c-[N+]-[O-])]',  
            'Aniline': 'c-[NX3H2]',
            'MichaelAcceptor_Enone': '[CX3]=[CX3]-[CX3](=[O,S,N])',
            'Epoxide_Alert': '[OD1r3]1[#6r3][#6r3]1', 
            'Tetrahydroisoquinoline_Core_Alert': 'c1ccc2c(c1)CCNCC2',  
            '4_Phenylpiperidine_Motif': 'c1ccccc1-C1CCNCC1', 
        }

        functional_groups = []
        fg_labels = []

        for fg_name, smarts in fg_smarts.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is None:
                    warnings.warn(f"Invalid SMARTS pattern: {fg_name} -> {smarts}")
                    continue
                matches = mol.GetSubstructMatches(pattern)
                for match in matches:
                    fg_atoms = set(match)
                    functional_groups.append(list(fg_atoms))
                    fg_labels.append(fg_name)
            except Exception as e:
                warnings.warn(f"Error in matching SMARTS pattern: {fg_name} -> {smarts}: {e}")
                continue

        return functional_groups, fg_labels

    def get_ring_structures(self, mol):
        rings = []
        labels = []
        ring_info = mol.GetRingInfo()
        for idxs in ring_info.AtomRings():
            is_aromatic = all([mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in idxs])
            ring_size = len(idxs)
            ring_label = f"AromaticRing_{ring_size}" if is_aromatic else f"Ring_{ring_size}"
            rings.append(list(idxs))
            labels.append(ring_label)
        fused_rings = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        if fused_rings > 0:
            labels.append("FusedRings")
            rings.append([atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()])
        return rings, labels

    def get_special_structures(self, mol):
        special_structures = []
        special_labels = []
        metal_atomic_numbers = [13, 12, 20, 26, 30, 29, 25, 24, 27, 28, 33, 80, 82, 50]
        metal_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in metal_atomic_numbers]
        if metal_atoms:
            special_structures.append(metal_atoms)
            special_labels.append("MetalAtoms")
        ssr = Chem.GetSymmSSSR(mol)
        atom_rings = [set(ring) for ring in ssr]
        spiro_atoms = set()
        for i in range(len(atom_rings)):
            for j in range(i + 1, len(atom_rings)):
                shared_atoms = atom_rings[i] & atom_rings[j]
                if len(shared_atoms) == 1:
                    spiro_atoms.update(shared_atoms)
        if spiro_atoms:
            special_structures.append(list(spiro_atoms))
            special_labels.append("SpiroAtoms")
        return special_structures, special_labels


    def generate_enhanced_hyperedge_attributes(self):

        num_hyperedges = len(self.hyperedges)


        hyperedge_features = np.zeros((num_hyperedges, 5), dtype=np.float32)


        if num_hyperedges == 0:
            print(f"Warning: No hyperedges found for molecule. Creating default attribute matrix with shape [1, 5].")
            return np.zeros((1, 5), dtype=np.float32)


        for i, (edge, label) in enumerate(zip(self.hyperedges, self.hyperedge_labels)):

            if i >= hyperedge_features.shape[0]:
                print(
                    f"Warning: Hyperedge index {i} exceeds feature matrix dimension {hyperedge_features.shape[0]}. Expanding matrix.")
   
                expanded_features = np.zeros((i + 1, 5), dtype=np.float32)
                expanded_features[:hyperedge_features.shape[0]] = hyperedge_features
                hyperedge_features = expanded_features

            current_label_handled = False
            if 'MurckoScaffoldCore' in label:
                hyperedge_features[i, 0] = 0.95  
                current_label_handled = True
            elif 'MurckoScaffoldFull' in label:
                hyperedge_features[i, 0] = 0.90 
                current_label_handled = True

            if not current_label_handled:
                if 'Ring' in label: 
                    if 'Aromatic' in label:
                        hyperedge_features[i, 0] = 0.85 
                    else:
                        hyperedge_features[i, 0] = 0.75 
                elif any(fg in label for fg in ['Amine', 'Alcohol', 'Acid', 'Amide', 'Carbonyl', 'Ester', 'Ketone', 'Aldehyde']): 
                    hyperedge_features[i, 0] = 0.80
                elif any(fg in label for fg in ['Halogen', 'Cyano', 'Nitro', 'Sulfonamide', 'Thiol', 'Sulfone']):
                    hyperedge_features[i, 0] = 0.70
                elif 'Connector' in label or 'Bridge' in label or 'Substituent' in label: 
                    hyperedge_features[i, 0] = 0.60
                elif 'IsolatedAtomGroup' in label or 'SingleAtom' in label:
                     hyperedge_features[i, 0] = 0.30
                else: 
                    hyperedge_features[i, 0] = 0.50 

            if 'Ring' in label:
                if 'Aromatic' in label:
                    hyperedge_features[i, 0] = 0.9  
                else:
                    hyperedge_features[i, 0] = 0.7 
            elif any(fg in label for fg in ['Amine', 'Alcohol', 'Acid', 'Amide', 'Carbonyl']):
                hyperedge_features[i, 0] = 0.8  
            elif any(fg in label for fg in ['Halogen', 'Cyano', 'Nitro']):
                hyperedge_features[i, 0] = 0.75  
            elif 'Connector' in label or 'Bridge' in label:
                hyperedge_features[i, 0] = 0.6  
            else:
                hyperedge_features[i, 0] = 0.5  

            hyperedge_size = len(edge)
            normalized_size = min(1.0, hyperedge_size / 10.0)
            hyperedge_features[i, 1] = normalized_size


            electron_feature = 0.0
            atom_count = 0

            for atom_idx in edge:
                if atom_idx < self.mol.GetNumAtoms():
                    atom = self.mol.GetAtomWithIdx(atom_idx)
                    atomic_num = atom.GetAtomicNum()
                    if atomic_num in [7, 8, 9, 17, 35, 53]: 
                        electron_feature += 0.8
                    elif atomic_num == 6 and atom.GetIsAromatic():  
                        electron_feature += 0.6
                    elif atomic_num == 6: 
                        electron_feature += 0.4
                    elif atomic_num == 1:  
                        electron_feature += 0.2
                    else:
                        electron_feature += 0.5
                    atom_count += 1

            hyperedge_features[i, 2] = electron_feature / max(1, atom_count)


            connectivity = 0.0
            for other_idx, other_edge in enumerate(self.hyperedges):
                if i != other_idx:
                    if set(edge).intersection(set(other_edge)):
                        connectivity += 1.0
            hyperedge_features[i, 3] = min(1.0, connectivity / max(1, len(self.hyperedges) / 2))


            pharmacophore_patterns = {
                'HBondDonor': ['[OH]', '[NH]', '[NH2]'],
                'HBondAcceptor': ['[O]', '[N;!$(N-*=O)]'],
                'Hydrophobic': ['[C;!$(C=O);!$(C#N)]', '[c]'],
                'Aromatic': ['c1ccccc1', 'c1ccncc1'],
                'Charged': ['[+]', '[-]', '[N+]', '[O-]']
            }

            is_pharmacophore = False
            for pattern_list in pharmacophore_patterns.values():
                for pattern in pattern_list:
                    patt = Chem.MolFromSmarts(pattern)
                    if patt and self.mol.HasSubstructMatch(patt):
                        matches = self.mol.GetSubstructMatches(patt)
                        for match in matches:
                            if any(atom_idx in edge for atom_idx in match):
                                is_pharmacophore = True
                                break
                        if is_pharmacophore:
                            break
                if is_pharmacophore:
                    break

            hyperedge_features[i, 4] = 0.9 if is_pharmacophore else 0.4

        if hyperedge_features.shape[0] != num_hyperedges:
            print(
                f"Warning: Final hyperedge feature matrix shape {hyperedge_features.shape} doesn't match expected size {num_hyperedges}. Fixing.")
            corrected_features = np.zeros((num_hyperedges, 5), dtype=np.float32)

            min_size = min(hyperedge_features.shape[0], num_hyperedges)
            corrected_features[:min_size] = hyperedge_features[:min_size]
            hyperedge_features = corrected_features


        return hyperedge_features

    def get_adjacency_matrix(self):
        num_nodes = self.mol.GetNumAtoms()
        num_hyperedges = len(self.hyperedges)
        adjacency_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)
        for hyperedge_idx, hyperedge in enumerate(self.hyperedges):
            for node in hyperedge:
                adjacency_matrix[node][hyperedge_idx] = 1
        return adjacency_matrix