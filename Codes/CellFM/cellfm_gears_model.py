"""
cellfm_gears_model.py - Modified GEARS model using cell-level embeddings (Original CellFM approach)

CellFM-GEARS uses cell-level control-cell embeddings: each control cell has a unique (n_genes, hidden_dim) embedding.

In this approach:
1. ctrl_emb has shape (n_ctrl_cells, n_genes, 1536)
2. data.z contains the control cell index for each sample
3. For each sample, we lookup ctrl_emb[data.z] to get cell-specific gene embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
import numpy as np
from torch_geometric.nn import SGConv
import os


class MLP(torch.nn.Module):
    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.network(x)


class GEARS_Model_CellEmb(torch.nn.Module):
    """
    GEARS with Cell-Level Embeddings (Original CellFM Approach)

    Uses pre-computed cell-level embeddings from CellFM encoder.
    Each control cell has its own embedding: (n_genes, hidden_dim)
    """

    def __init__(self, args, ctrl_emb=None):
        super(GEARS_Model_CellEmb, self).__init__()
        self.args = args
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.hidden_size = hidden_size
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.cell_fitness_pred = args['cell_fitness_pred']
        self.pert_emb_lambda = 0.2

        # Store cell-level embeddings
        # Shape: (n_ctrl_cells, n_genes, emb_dim)
        if ctrl_emb is not None:
            self.register_buffer('ctrl_emb', ctrl_emb)
            self.ctrl_emb_size = ctrl_emb.shape[-1]
            print(f"Loaded cell embeddings: {ctrl_emb.shape}")
        else:
            self.ctrl_emb = None
            self.ctrl_emb_size = 1536  # Default

        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)

        # gene/global perturbation embedding dictionary lookup
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)

        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()

        # Embedding transformation: from CellFM dim to hidden_size
        self.cellfm_emb_transform = MLP([self.ctrl_emb_size, 1024, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')

        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))

        # perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))

        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')

        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes, hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)

        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size, hidden_size])

        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes, hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)

        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)

        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')

        # cell fitness prediction
        self.cell_fitness_mlp = MLP([self.num_genes, hidden_size*2, hidden_size, 1], last_layer_act='linear')

        self.pretrained = ctrl_emb is not None
        print(f'Using cell-level embeddings: {self.pretrained}')

    def forward(self, data):
        x, pert_idx = data.x, data.pert_idx

        if self.no_perturb:
            out = x.reshape(-1, 1)
            out = torch.split(torch.flatten(out), self.num_genes)
            return torch.stack(out)

        num_graphs = len(data.batch.unique())

        # Get base gene embeddings using cell-level lookup
        x = x.reshape(num_graphs, self.num_genes + 1)[:, :-1]

        if self.pretrained and self.ctrl_emb is not None:
            # Cell-level embedding lookup using data.z
            # data.z contains the control cell index for each sample
            cell_indices = data.z.squeeze()  # (batch_size,)

            # Map indices to available embedding range (use modulo when we have fewer embeddings than cells)
            num_available_cells = self.ctrl_emb.shape[0]
            cell_indices = cell_indices % num_available_cells

            # Look up embeddings: ctrl_emb[cell_indices] -> (batch_size, n_genes, emb_dim)
            emb = self.ctrl_emb[cell_indices]  # (batch_size, n_genes, emb_dim)

            # Reshape for processing: (batch_size * n_genes, emb_dim)
            emb = emb.view(-1, self.ctrl_emb_size)

            # Position embeddings
            pos_emb = self.emb_pos(
                torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs).to(self.args['device'])
            )
        else:
            # Fallback: use learned gene embeddings (no pretrained)
            emb = self.gene_emb(
                torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs).to(self.args['device'])
            )
            pos_emb = self.emb_pos(
                torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs).to(self.args['device'])
            )

        # Transform embeddings
        if not self.pretrained:
            # No pretrained embeddings - use GNN to enhance
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.cellfm_emb_transform(base_emb)
        else:
            # Using CellFM pretrained cell-level embeddings
            base_emb = self.cellfm_emb_transform(emb)

        # Get perturbation index and embeddings
        pert_index = []
        for idx, i in enumerate(pert_idx):
            for j in i:
                if j != -1:
                    pert_index.append([idx, j])
        pert_index = torch.tensor(pert_index).T

        pert_global_emb = self.pert_emb(
            torch.LongTensor(list(range(self.num_perts))).to(self.args['device'])
        )

        # Augment global perturbation embedding with GNN
        for idx, layer in enumerate(self.sim_layers):
            pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
            if idx < self.num_layers - 1:
                pert_global_emb = pert_global_emb.relu()

        # Add global perturbation embedding to each gene in each cell in the batch
        base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)

        if pert_index.shape[0] != 0:
            # In case all samples in the batch are controls, there is no indexing for pert_index
            pert_track = {}
            for i, j in enumerate(pert_index[0]):
                if j.item() in pert_track:
                    pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                else:
                    pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

            if len(list(pert_track.values())) > 0:
                if len(list(pert_track.values())) == 1:
                    # Circumvent when batch size = 1 with single perturbation
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                else:
                    emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                for idx, j in enumerate(pert_track.keys()):
                    base_emb[j] = base_emb[j] + emb_total[idx]

        base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
        base_emb = self.bn_pert_base(base_emb)

        # Apply the first MLP
        base_emb = self.transform(base_emb)
        out = self.recovery_w(base_emb)
        out = out.reshape(num_graphs, self.num_genes, -1)
        out = out.unsqueeze(-1) * self.indv_w1
        w = torch.sum(out, axis=2)
        out = w + self.indv_b1

        # Cross gene
        cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
        cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

        cross_gene_embed = cross_gene_embed.reshape([num_graphs, self.num_genes, -1])
        cross_gene_out = torch.cat([out, cross_gene_embed], 2)

        cross_gene_out = cross_gene_out * self.indv_w2
        cross_gene_out = torch.sum(cross_gene_out, axis=2)
        out = cross_gene_out + self.indv_b2
        out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1, 1)
        out = torch.split(torch.flatten(out), self.num_genes)

        # Uncertainty head
        if self.uncertainty:
            out_logvar = self.uncertainty_w(base_emb)
            out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
            return torch.stack(out), torch.stack(out_logvar)

        if self.cell_fitness_pred:
            return torch.stack(out), self.cell_fitness_mlp(torch.stack(out))

        return torch.stack(out)


def load_cell_embeddings(embedding_path, device='cpu', mmap_mode=None, max_cells=None):
    """
    Load cell-level embeddings from file.

    Args:
        embedding_path: Path to the embedding file (.npy)
                       Expected shape: (n_ctrl_cells, n_genes, emb_dim)
        device: Device to load embeddings to
        mmap_mode: Memory-map mode. Options:
                   - None: Load entire array into RAM (default, required for GPU)
                   - 'r': Read-only memory-mapped (saves RAM but CPU only)
        max_cells: Maximum number of cells to load (default: None = load all)

    Returns:
        torch.Tensor: Cell-level embeddings

    Note:
        For GPU training, embeddings must be fully loaded into GPU memory.
        The ~1.1 TB embedding file requires sufficient GPU memory (A100 80GB works
        after loading subset or using gradient checkpointing).
    """
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    print(f"Loading cell-level embeddings from {embedding_path}")

    # Check file size
    file_size_gb = os.path.getsize(embedding_path) / (1024**3)
    print(f"  File size: {file_size_gb:.1f} GB")

    # Try to load metadata to get shape
    meta_path = embedding_path.replace('.npy', '_meta.npz')
    if os.path.exists(meta_path):
        meta = np.load(meta_path, allow_pickle=True)
        n_cells = int(meta['n_cells'])
        n_genes = int(meta['n_genes'])
        hidden_dim = int(meta['hidden_dim'])
        shape = (n_cells, n_genes, hidden_dim)
        print(f"  Shape from metadata: {shape}")
    else:
        # Default shape for nocap_perturbation dataset
        shape = (10000, 19424, 1536)
        print(f"  Using default shape: {shape}")

    # Load embeddings using memory-mapped file
    # The file was saved with np.memmap, so we load it the same way
    if mmap_mode:
        print(f"  Using memory-mapped mode: {mmap_mode}")
        emb = np.memmap(embedding_path, dtype='float32', mode=mmap_mode, shape=shape)
    else:
        print(f"  Loading full array into memory...")
        emb = np.memmap(embedding_path, dtype='float32', mode='r', shape=shape)

        # Apply max_cells limit before copying to save memory
        if max_cells is not None and max_cells < shape[0]:
            print(f"  Limiting to {max_cells} cells (from {shape[0]})")
            emb = np.array(emb[:max_cells])  # Copy only the subset
        else:
            emb = np.array(emb)  # Copy full array

    print(f"  Shape: {emb.shape}")
    print(f"  - {emb.shape[0]} control cells")
    print(f"  - {emb.shape[1]} genes")
    print(f"  - {emb.shape[2]} embedding dimensions")

    return torch.tensor(emb, dtype=torch.float32, device=device)
