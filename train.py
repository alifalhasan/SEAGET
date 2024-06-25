import os
import yaml
import torch
import pickle
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from pathlib import Path
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import OneHotEncoder
from param_parser import parameter_parser
from dataloader import load_graph_adj_mat, load_graph_node_features
from data_class import (
    TrajectoryDatasetTrain,
    TrajectoryDatasetVal,
    TrajectoryDatasetTest,
)
from helper_functions import (
    filter_inactive_pois,
    generate_active_hours,
    increment_path,
    input_traj_to_embeddings,
)
from model import (
    GCN,
    NodeAttnMap,
    UserEmbeddings,
    Time2Vec,
    CategoryEmbeddings,
    FuseEmbeddings,
    TransformerModel,
)
from utils import (
    calculate_laplacian_matrix,
    top_k_acc_last_timestep,
    mAP_metric_last_timestep,
    MRR_metric_last_timestep,
    masked_mse_loss,
)


def train(args):
    args.save_dir = increment_path(
        Path(args.project) / args.name, exist_ok=args.exist_ok, sep="-"
    )
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Clear any existing handlers to start fresh
    for handler in logging.root.handlers[:]:  # Iterate through existing handlers
        logging.root.removeHandler(handler)  # Remove each handler

    # Configure the main logger
    logging.basicConfig(
        level=logging.DEBUG,  # Set minimum logging level to DEBUG (most detailed)
        format="%(asctime)s %(message)s",  # Specify log message format
        datefmt="%Y-%m-%d %H:%M:%S",  # Set date and time format for log messages
        filename=os.path.join(args.save_dir, f"log_training.txt"),  # Set log file path
        filemode="w",  # Overwrite existing log file (create a new one)
    )

    # Add a console handler for printing logs to the terminal
    console = logging.StreamHandler()  # Create a handler for console output
    console.setLevel(logging.INFO)  # Set minimum logging level for console to INFO
    formatter = logging.Formatter(
        "%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )  # Set message format
    console.setFormatter(formatter)  # Apply the formatter to the console handler
    logging.getLogger("").addHandler(
        console
    )  # Add the console handler to the root logger

    # Disable logging from matplotlib's font manager to avoid excessive output
    logging.getLogger("matplotlib.font_manager").disabled = True

    # Section for saving configuration information

    # Log the arguments for reference
    logging.info(args)

    # Save arguments to a YAML file for reproducibility
    with open(os.path.join(args.save_dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f, sort_keys=False)  # Dump arguments as YAML

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train)
    val_df = pd.read_csv(args.data_val)
    test_df = pd.read_csv(args.data_test)

    # Build POI graph (built from train_df)
    print("Loading POI graph...")
    raw_adj_mat = load_graph_adj_mat(args.data_adj_mat)
    raw_node_features = load_graph_node_features(
        args.data_node_feats, args.feature1, args.feature2, args.feature3, args.feature4
    )
    logging.info(
        f"raw_node_features.shape: {raw_node_features.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}."
    )
    logging.info(
        f"raw_adj_mat.shape: {raw_adj_mat.shape}; Edge from row_index to col_index with weight (frequency)."
    )
    num_pois = raw_node_features.shape[0]

    # One-hot encoding poi categories
    logging.info("One-hot encoding poi categories id")  # Log a message for context

    one_hot_encoder = OneHotEncoder()  # Create a one-hot encoder object

    cat_list = list(raw_node_features[:, 1])  # Extract category values as a list

    # Fit the encoder to the categories (ensure compatibility with single-element lists)
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))

    # Transform the categories into one-hot encoded representation
    one_hot_rlt = one_hot_encoder.transform(
        list(map(lambda x: [x], cat_list))  # Reshape for transformation
    ).toarray()  # Convert to NumPy array

    num_cats = one_hot_rlt.shape[-1]  # Get the number of categories

    # Create a new feature matrix to accommodate one-hot encoding
    X = np.zeros(
        (num_pois, raw_node_features.shape[-1] - 1 + num_cats), dtype=np.float32
    )  # Allocate space for encoded features

    # Populate the new feature matrix
    X[:, 0] = raw_node_features[:, 0]  # Copy first column (original feature)
    X[:, 1 : num_cats + 1] = one_hot_rlt  # Insert one-hot encoded features
    X[:, num_cats + 1 :] = raw_node_features[
        :, 2:
    ]  # Copy remaining columns (original features)

    # Log information about the encoded features
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f"POI categories: {list(one_hot_encoder.categories_[0])}")

    # Save the one-hot encoder for later use
    with open(os.path.join(args.save_dir, "one-hot-encoder.pkl"), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print("Laplician matrix...")
    A = calculate_laplacian_matrix(raw_adj_mat, mat_type="hat_rw_normd_lap_mat")

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats, encoding="latin-1")
    poi_ids = list(nodes_df["node_name/poi_id"].unique())
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = list(nodes_df[args.feature2].unique())
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row["node_name/poi_id"]]] = (
            cat_id2idx_dict[row[args.feature2]]
        )

    # User id to index
    user_ids = [str(each) for each in list(train_df["user_id"].unique())]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # %% ====================== Define dataloader ======================
    print("Prepare dataloader...")
    train_dataset = TrajectoryDatasetTrain(args, train_df, poi_id2idx_dict)
    val_dataset = TrajectoryDatasetVal(args, val_df, poi_id2idx_dict, user_id2idx_dict)
    test_dataset = TrajectoryDatasetTest(
        args, test_df, poi_id2idx_dict, user_id2idx_dict
    )

    # Create a DataLoader for the training dataset
    train_loader = DataLoader(
        train_dataset,  # Specify the training dataset
        batch_size=args.batch,  # Set the batch size (likely from command-line arguments)
        shuffle=True,  # Shuffle the training data before each epoch
        drop_last=False,  # Keep the last batch even if it's smaller than the batch size
        pin_memory=True,  # Pin memory for faster GPU transfer
        num_workers=args.workers,  # Use multiple subprocesses to load data in parallel
        collate_fn=lambda x: x,  # Custom function to collate data (here, it keeps data as-is)
    )

    # Create a DataLoader for the validation dataset
    val_loader = DataLoader(
        val_dataset,  # Specify the validation dataset
        batch_size=args.batch,  # Use the same batch size as for training
        shuffle=False,  # Don't shuffle the validation data
        drop_last=False,  # Keep the last validation batch
        pin_memory=True,  # Pin memory for faster GPU transfer
        num_workers=args.workers,  # Use multiple subprocesses for validation data loading
        collate_fn=lambda x: x,  # Same collation function as for training
    )

    test_loader = DataLoader(
        test_dataset,  # Specify the testing dataset
        batch_size=args.batch,  # Use the same batch size as for training
        shuffle=False,  # Don't shuffle the testing data
        drop_last=False,  # Keep the last testing batch
        pin_memory=True,  # Pin memory for faster GPU transfer
        num_workers=args.workers,  # Use multiple subprocesses for testing data loading
        collate_fn=lambda x: x,  # Same collation function as for training
    )

    # %% ====================== Get Active Hours ======================
    active_hours = generate_active_hours(poi_id2idx_dict, args.device)

    # %% ====================== Build Models ======================
    # Model-1: POI embedding model
    if isinstance(X, np.ndarray):
        # Convertion to a PyTorch tensor from NumPy array
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    # Set the number of input features for the GCN model:
    args.gcn_nfeat = X.shape[
        1
    ]  # Determine the number of features from the input data X

    # Create an instance of the GCN model:
    poi_embed_model = GCN(
        ninput=args.gcn_nfeat,  # Number of input features
        nhid=args.gcn_nhid,  # Number of hidden units in each hidden layer
        noutput=args.poi_embed_dim,  # Number of output features (embedding dimension)
        dropout=args.gcn_dropout,  # Dropout probability for regularization
    )

    # Node Attn Model
    node_attn_model = NodeAttnMap(
        in_features=X.shape[1],  # Number of input features
        nhid=args.node_attn_nhid,  # Number of hidden units
        use_mask=False,  # Disable masked attention for now
    )

    # %% Model-2: User embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, args.user_embed_dim)

    # %% Model-3: Time Model
    time_embed_model = Time2Vec("sin", out_dim=args.time_embed_dim)

    # %% Model-4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # %% Model-5: Embedding fusion models
    embed_fuse_model1 = FuseEmbeddings(
        args.user_embed_dim, args.poi_embed_dim
    )  # Combine user and POI embeddings
    embed_fuse_model2 = FuseEmbeddings(
        args.time_embed_dim, args.cat_embed_dim
    )  # Combine time and category embeddings
    embed_fuse_model3 = FuseEmbeddings(
        args.season_embed_dim, args.poi_embed_dim
    )  # Combine season and POI embeddings

    # %% Model-6: Sequence model
    # Calculate total embedding dimension for input sequences:
    args.seq_input_embed = (
        args.poi_embed_dim  # Dimension for POI embeddings
        + args.user_embed_dim  # Dimension for user embeddings
        + args.time_embed_dim  # Dimension for time embeddings
        + args.cat_embed_dim  # Dimension for category embeddings
        + args.season_embed_dim  # Dimension for season embeddings
        + args.poi_embed_dim  # Dimension for POI embeddings
    )

    # Create a TransformerModel instance:
    seq_model = TransformerModel(
        # Decoder output dimensions:
        num_pois=num_pois,  # Number of possible POIs
        num_cats=num_cats,  # Number of possible categories
        # Transformer architecture parameters:
        embed_size=args.seq_input_embed,  # Input embedding dimension
        nhead=args.transformer_nhead,  # Number of attention heads
        nhid=args.transformer_nhid,  # Hidden layer dimension
        nlayers=args.transformer_nlayers,  # Number of encoder layers
        dropout=args.transformer_dropout,  # Dropout probability
    )

    # Define overall loss and optimizer
    optimizer = optim.Adam(
        params=list(poi_embed_model.parameters())
        + list(node_attn_model.parameters())
        + list(user_embed_model.parameters())
        + list(time_embed_model.parameters())
        + list(cat_embed_model.parameters())
        + list(embed_fuse_model1.parameters())
        + list(embed_fuse_model2.parameters())
        + list(embed_fuse_model3.parameters())
        + list(seq_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # CrossEntropyLoss for POI prediction, ignoring padding:
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 represents padding

    # CrossEntropyLoss for category prediction, also ignoring padding:
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 represents padding

    # Masked MSE loss for time prediction:
    criterion_time = masked_mse_loss

    # Learning rate scheduler:
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,  # The optimizer to adjust learning rates for
        "min",  # Monitor the validation loss (minimize it)
        verbose=True,  # Print messages when learning rate is reduced
        factor=args.lr_scheduler_factor,  # Factor by which to reduce learning rate
    )

    # %% Helper function for training

    def adjust_pred_prob_by_graph(y_pred_poi):
        """
        Adjusts predicted POI probabilities based on information from a graph.

        Args:
            y_pred_poi: A tensor of predicted POI probabilities (batch, seq_len, num_pois).

        Returns:
            A tensor of adjusted POI probabilities with the same shape.
        """
        # 1. Initialize adjusted probabilities
        y_pred_poi_adjusted = torch.zeros_like(
            y_pred_poi
        )  # Create a tensor to store adjusted probabilities

        # 2. Calculate attention scores from graph
        attn_map = node_attn_model(
            X, A
        )  # Get attention scores from a graph attention model

        # 3. Adjust probabilities for each trajectory
        for i in range(len(batch_seq_lens)):  # Loop through trajectories in the batch
            traj_i_input = batch_input_seqs[
                i
            ]  # Extract input POIs for the current trajectory
            for j in range(len(traj_i_input)):  # Loop through POIs in the trajectory
                # Adjust probability by adding attention score:
                y_pred_poi_adjusted[i, j, :] = (
                    attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]
                )

        return y_pred_poi_adjusted

    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    embed_fuse_model1 = embed_fuse_model1.to(device=args.device)
    embed_fuse_model2 = embed_fuse_model2.to(device=args.device)
    embed_fuse_model3 = embed_fuse_model3.to(device=args.device)
    seq_model = seq_model.to(device=args.device)

    # For plotting

    # Top-k accuracy for POI prediction at k=1, 5, 10, and 20
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []

    # Mean Average Precision@20 (mAP@20) for POI prediction
    train_epochs_mAP20_list = []

    # Mean Reciprocal Rank (MRR) for POI prediction
    train_epochs_mrr_list = []

    # Total loss and individual losses for different tasks
    train_epochs_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []

    # Top-k accuracy for POI prediction at k=1, 5, 10, and 20
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []

    # Mean Average Precision@20 (mAP@20) for POI prediction
    val_epochs_mAP20_list = []

    # Mean Reciprocal Rank (MRR) for POI prediction
    val_epochs_mrr_list = []

    # Total loss and individual losses for different tasks
    val_epochs_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []

    # Checkpoint management
    # Best validation score for saving checkpoints
    max_val_score = -np.inf

    # %% ============================================ Train starts ============================================
    # %% Loop epoch
    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        embed_fuse_model1.train()
        embed_fuse_model2.train()
        embed_fuse_model3.train()
        seq_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(
            args.device
        )  # Create a mask to prevent attending to future positions in sequences

        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(
                    args.device
                )

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[2]]
                label_seq = [each[0] for each in sample[3]]
                label_seq_time = [each[1] for each in sample[3]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(
                    input_traj_to_embeddings(
                        sample=sample,
                        poi_embeddings=poi_embeddings,
                        args=args,
                        poi_idx2cat_idx_dict=poi_idx2cat_idx_dict,
                        user_id2idx_dict=user_id2idx_dict,
                        user_embed_model=user_embed_model,
                        time_embed_model=time_embed_model,
                        cat_embed_model=cat_embed_model,
                        embed_fuse_model1=embed_fuse_model1,
                        embed_fuse_model2=embed_fuse_model2,
                        embed_fuse_model3=embed_fuse_model3,
                    )
                )
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(
                batch_seq_embeds, batch_first=True, padding_value=-1
            )
            label_padded_poi = pad_sequence(
                batch_seq_labels_poi, batch_first=True, padding_value=-1
            )
            label_padded_time = pad_sequence(
                batch_seq_labels_time, batch_first=True, padding_value=-1
            )
            label_padded_cat = pad_sequence(
                batch_seq_labels_cat, batch_first=True, padding_value=-1
            )

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            # Perform a forward pass through the sequence model
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            # Adjust predicted POI probabilities using graph attention
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            # Filter out inactive POIs
            y_pred_poi_adjusted = filter_inactive_pois(
                y_pred_poi_adjusted, y_pred_time, active_hours, args.device
            )

            # Calculate loss for POI predictions
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)

            # Calculate loss for time predictions
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)

            # Calculate loss for category predictions
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            # Combine losses with weights
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

            # Zero gradients
            optimizer.zero_grad()

            # Backpropagate gradients
            loss.backward(retain_graph=True)  # Keep graph for later analysis

            # Update model parameters
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0

            # Move tensors to CPU for subsequent calculations
            batch_label_pois = y_poi.detach().cpu().numpy()  # Ground truth POI labels
            batch_pred_pois = (
                y_pred_poi_adjusted.detach().cpu().numpy()
            )  # Predicted POIs
            batch_pred_times = y_pred_time.detach().cpu().numpy()  # Predicted times
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()  # Predicted categories

            # Iterate through samples in the batch
            for label_pois, pred_pois, seq_len in zip(
                batch_label_pois, batch_pred_pois, batch_seq_lens
            ):
                # Truncate tensors to actual sequence length
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)

                # Calculate various accuracy and ranking metrics for the last timestep
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))
            train_batches_loss_list.append(loss.detach().cpu().numpy())
            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(
                    f"Epoch:{epoch}, batch:{b_idx}, "
                    f"train_batch_loss:{loss.item():.2f}, "
                    f"train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, "
                    f"train_move_loss:{np.mean(train_batches_loss_list):.2f}\n"
                    f"train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n"
                    f"train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n"
                    f"train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n"
                    f"train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n"
                    f"train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n"
                    f"train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n"
                    f"train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n"
                    f"train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n"
                    f"traj_id:{batch[sample_idx][0]}\n"
                    f"input_seq: {batch[sample_idx][2]}\n"
                    f"label_seq:{batch[sample_idx][3]}\n"
                    f"pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][3]]}\n"
                    f"pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n"
                    f"pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    + "=" * 100
                )

        # %% ============================================ Train ends ============================================

        # %% ============================================ Validation starts ============================================
        # Set models to evaluation mode (disables dropout and batch normalization)
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        embed_fuse_model3.eval()
        seq_model.eval()

        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []

        # Create a source mask for attention in the sequence model
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)

        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(
                    args.device
                )

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[2]]
                label_seq = [each[0] for each in sample[3]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(
                    input_traj_to_embeddings(
                        sample=sample,
                        poi_embeddings=poi_embeddings,
                        args=args,
                        poi_idx2cat_idx_dict=poi_idx2cat_idx_dict,
                        user_id2idx_dict=user_id2idx_dict,
                        user_embed_model=user_embed_model,
                        time_embed_model=time_embed_model,
                        cat_embed_model=cat_embed_model,
                        embed_fuse_model1=embed_fuse_model1,
                        embed_fuse_model2=embed_fuse_model2,
                        embed_fuse_model3=embed_fuse_model3,
                    )
                )
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(
                batch_seq_embeds, batch_first=True, padding_value=-1
            )
            label_padded_poi = pad_sequence(
                batch_seq_labels_poi, batch_first=True, padding_value=-1
            )
            label_padded_time = pad_sequence(
                batch_seq_labels_time, batch_first=True, padding_value=-1
            )
            label_padded_cat = pad_sequence(
                batch_seq_labels_cat, batch_first=True, padding_value=-1
            )

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            # Filter out inactive POIs
            y_pred_poi_adjusted = filter_inactive_pois(
                y_pred_poi_adjusted, y_pred_time, active_hours, args.device
            )

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(
                batch_label_pois, batch_pred_pois, batch_seq_lens
            ):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))
            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(
                    f"Epoch:{epoch}, batch:{vb_idx}, "
                    f"val_batch_loss:{loss.item():.2f}, "
                    f"val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, "
                    f"val_move_loss:{np.mean(val_batches_loss_list):.2f} \n"
                    f"val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n"
                    f"val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n"
                    f"val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n"
                    f"val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n"
                    f"val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n"
                    f"val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n"
                    f"val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n"
                    f"val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n"
                    f"traj_id:{batch[sample_idx][0]}\n"
                    f"input_seq:{batch[sample_idx][2]}\n"
                    f"label_seq:{batch[sample_idx][3]}\n"
                    f"pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"label_seq_cat:{[poi_idx2cat_idx_dict[each[0]] for each in batch[sample_idx][3]]}\n"
                    f"pred_seq_cat:{list(np.argmax(batch_pred_cats, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    f"label_seq_time:{list(batch_seq_labels_time[sample_idx].numpy()[:batch_seq_lens[sample_idx]])}\n"
                    f"pred_seq_time:{list(np.squeeze(batch_pred_times)[sample_idx][:batch_seq_lens[sample_idx]])} \n"
                    + "=" * 100
                )
        # %% ============================================ Validation ends ============================================

        # Calculate average metrics for the training set
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)
        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)

        # Calculate average metrics for the validation set
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)
        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save train metrics to lists
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)

        # Save validation metrics to lists
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)

        # Monitor loss and score
        monitor_loss = epoch_val_loss  # Use validation loss for monitoring
        monitor_score = np.mean(epoch_val_top1_acc * 4 + epoch_val_top20_acc)

        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(
            f"Epoch {epoch}/{args.epochs}\n"
            f"train_loss:{epoch_train_loss:.4f}, "
            f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
            f"train_time_loss:{epoch_train_time_loss:.4f}, "
            f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
            f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
            f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
            f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
            f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
            f"train_mAP20:{epoch_train_mAP20:.4f}, "
            f"train_mrr:{epoch_train_mrr:.4f}\n"
            f"val_loss: {epoch_val_loss:.4f}, "
            f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
            f"val_time_loss: {epoch_val_time_loss:.4f}, "
            f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
            f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
            f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
            f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
            f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
            f"val_mAP20:{epoch_val_mAP20:.4f}, "
            f"val_mrr:{epoch_val_mrr:.4f}"
        )

        # Save poi and user embeddings
        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, "embeddings")
            if not os.path.exists(embeddings_save_dir):
                os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy()
                poi_embedding_list = []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(
                    os.path.join(embeddings_save_dir, "saved_poi_embeddings"),
                    save_poi_embeddings,
                )
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=args.device)
                    user_embedding = (
                        user_embed_model(input).detach().cpu().numpy().flatten()
                    )
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(
                    os.path.join(embeddings_save_dir, "saved_user_embeddings"),
                    user_embeddings,
                )
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=args.device)
                    cat_embedding = (
                        cat_embed_model(input).detach().cpu().numpy().flatten()
                    )
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(
                    os.path.join(embeddings_save_dir, "saved_cat_embeddings"),
                    cat_embeddings,
                )
                # Save time embeddings
                time_embedding_list = []
                for time_idx in range(args.time_units):
                    input = torch.FloatTensor([time_idx]).to(device=args.device)
                    time_embedding = (
                        time_embed_model(input).detach().cpu().numpy().flatten()
                    )
                    time_embedding_list.append(time_embedding)
                time_embeddings = np.array(time_embedding_list)
                np.save(
                    os.path.join(embeddings_save_dir, "saved_time_embeddings"),
                    time_embeddings,
                )

        # Save model state dict
        if args.save_weights:
            # Create a dictionary to store model state information
            state_dict = {
                "epoch": epoch,  # Current training epoch
                # State dictionaries of various model components:
                "poi_embed_state_dict": poi_embed_model.state_dict(),
                "node_attn_state_dict": node_attn_model.state_dict(),
                "user_embed_state_dict": user_embed_model.state_dict(),
                "time_embed_state_dict": time_embed_model.state_dict(),
                "cat_embed_state_dict": cat_embed_model.state_dict(),
                "embed_fuse1_state_dict": embed_fuse_model1.state_dict(),
                "embed_fuse2_state_dict": embed_fuse_model2.state_dict(),
                "embed_fuse3_state_dict": embed_fuse_model3.state_dict(),
                "seq_model_state_dict": seq_model.state_dict(),
                # Optimizer state and various mapping dictionaries:
                "optimizer_state_dict": optimizer.state_dict(),
                "user_id2idx_dict": user_id2idx_dict,
                "poi_id2idx_dict": poi_id2idx_dict,
                "cat_id2idx_dict": cat_id2idx_dict,
                "poi_idx2cat_idx_dict": poi_idx2cat_idx_dict,
                # Node attention map calculated using input X and A:
                "node_attn_map": node_attn_model(X, A),
                # Arguments used for training and evaluation metrics:
                "args": args,
                "epoch_train_metrics": {
                    # Training metrics (losses and accuracies)
                    "epoch_train_loss": epoch_train_loss,
                    "epoch_train_poi_loss": epoch_train_poi_loss,
                    "epoch_train_time_loss": epoch_train_time_loss,
                    "epoch_train_cat_loss": epoch_train_cat_loss,
                    "epoch_train_top1_acc": epoch_train_top1_acc,
                    "epoch_train_top5_acc": epoch_train_top5_acc,
                    "epoch_train_top10_acc": epoch_train_top10_acc,
                    "epoch_train_top20_acc": epoch_train_top20_acc,
                    "epoch_train_mAP20": epoch_train_mAP20,
                    "epoch_train_mrr": epoch_train_mrr,
                },
                "epoch_val_metrics": {
                    # Validation metrics (losses and accuracies)
                    "epoch_val_loss": epoch_val_loss,
                    "epoch_val_poi_loss": epoch_val_poi_loss,
                    "epoch_val_time_loss": epoch_val_time_loss,
                    "epoch_val_cat_loss": epoch_val_cat_loss,
                    "epoch_val_top1_acc": epoch_val_top1_acc,
                    "epoch_val_top5_acc": epoch_val_top5_acc,
                    "epoch_val_top10_acc": epoch_val_top10_acc,
                    "epoch_val_top20_acc": epoch_val_top20_acc,
                    "epoch_val_mAP20": epoch_val_mAP20,
                    "epoch_val_mrr": epoch_val_mrr,
                },
            }
            # Construct the model checkpoint directory path
            model_save_dir = os.path.join(args.save_dir, "checkpoints")

            # Save the model state if the current validation score is the best so far
            if monitor_score >= max_val_score:
                # Create the directory if it doesn't exist
                if not os.path.exists(model_save_dir):
                    os.makedirs(model_save_dir)

                # Save the state dictionary to a file
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")

                # Save validation metrics to a text file
                with open(rf"{model_save_dir}/best_epoch.txt", "w") as f:
                    print(state_dict["epoch_val_metrics"], file=f)

                # Update the maximum validation score
                max_val_score = monitor_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, "metrics-train.txt"), "w") as f:
            print(
                f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}',
                file=f,
            )
            print(
                f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}',
                file=f,
            )
            print(
                f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                file=f,
            )
            print(
                f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}',
                file=f,
            )
            print(
                f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}',
                file=f,
            )
            print(
                f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}',
                file=f,
            )
            print(
                f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                file=f,
            )
            print(
                f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                file=f,
            )
            print(
                f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}',
                file=f,
            )
            print(
                f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}',
                file=f,
            )
        with open(os.path.join(args.save_dir, "metrics-val.txt"), "w") as f:
            print(
                f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}',
                file=f,
            )
            print(
                f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}',
                file=f,
            )
            print(
                f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}',
                file=f,
            )
            print(
                f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}',
                file=f,
            )
            print(
                f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}',
                file=f,
            )
            print(
                f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}',
                file=f,
            )
            print(
                f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}',
                file=f,
            )
            print(
                f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}',
                file=f,
            )
            print(
                f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}',
                file=f,
            )
            print(
                f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}',
                file=f,
            )

    # %% ============================================ Testing starts ============================================
    with torch.inference_mode():
        # Set models to evaluation mode (disables dropout and batch normalization)
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        embed_fuse_model1.eval()
        embed_fuse_model2.eval()
        embed_fuse_model3.eval()
        seq_model.eval()

        test_batches_top1_acc_list = []
        test_batches_top5_acc_list = []
        test_batches_top10_acc_list = []
        test_batches_top20_acc_list = []
        test_batches_mAP20_list = []
        test_batches_mrr_list = []
        test_batches_loss_list = []
        test_batches_poi_loss_list = []
        test_batches_time_loss_list = []
        test_batches_cat_loss_list = []

        # Create a source mask for attention in the sequence model
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)

        for vb_idx, batch in enumerate(test_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(
                    args.device
                )

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_seq_embeds = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A)

            # Convert input seq to embeddings
            for sample in batch:
                input_seq = [each[0] for each in sample[2]]
                label_seq = [each[0] for each in sample[3]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                input_seq_embed = torch.stack(
                    input_traj_to_embeddings(
                        sample=sample,
                        poi_embeddings=poi_embeddings,
                        args=args,
                        poi_idx2cat_idx_dict=poi_idx2cat_idx_dict,
                        user_id2idx_dict=user_id2idx_dict,
                        user_embed_model=user_embed_model,
                        time_embed_model=time_embed_model,
                        cat_embed_model=cat_embed_model,
                        embed_fuse_model1=embed_fuse_model1,
                        embed_fuse_model2=embed_fuse_model2,
                        embed_fuse_model3=embed_fuse_model3,
                    )
                )
                batch_seq_embeds.append(input_seq_embed)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            batch_padded = pad_sequence(
                batch_seq_embeds, batch_first=True, padding_value=-1
            )
            label_padded_poi = pad_sequence(
                batch_seq_labels_poi, batch_first=True, padding_value=-1
            )
            label_padded_time = pad_sequence(
                batch_seq_labels_time, batch_first=True, padding_value=-1
            )
            label_padded_cat = pad_sequence(
                batch_seq_labels_cat, batch_first=True, padding_value=-1
            )

            # Feedforward
            x = batch_padded.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x, src_mask)

            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)

            # Filter out inactive POIs
            y_pred_poi_adjusted = filter_inactive_pois(
                y_pred_poi_adjusted, y_pred_time, active_hours, args.device
            )

            # Calculate loss
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss = loss_poi + loss_time * args.time_loss_weight + loss_cat

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for label_pois, pred_pois, seq_len in zip(
                batch_label_pois, batch_pred_pois, batch_seq_lens
            ):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)
            test_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            test_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            test_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            test_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            test_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            test_batches_mrr_list.append(mrr / len(batch_label_pois))
            test_batches_loss_list.append(loss.detach().cpu().numpy())
            test_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            test_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            test_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

        with open(os.path.join(args.save_dir, "metrics-test.txt"), "w") as f:
            print(
                f"test_loss={np.mean(test_batches_loss_list)}",
                file=f,
            )
            print(
                f"test_poi_loss={np.mean(test_batches_poi_loss_list)}",
                file=f,
            )
            print(
                f"test_time_loss={np.mean(test_batches_time_loss_list)}",
                file=f,
            )
            print(
                f"test_cat_loss={np.mean(test_batches_cat_loss_list)}",
                file=f,
            )
            print(
                f"test_top1_acc={np.mean(test_batches_top1_acc_list)}",
                file=f,
            )
            print(
                f"test_top5_acc={np.mean(test_batches_top5_acc_list)}",
                file=f,
            )
            print(
                f"test_top10_acc={np.mean(test_batches_top10_acc_list)}",
                file=f,
            )
            print(
                f"test_top20_acc={np.mean(test_batches_top20_acc_list)}",
                file=f,
            )
            print(
                f"test_mAP20={np.mean(test_batches_mAP20_list)}",
                file=f,
            )
            print(
                f"test_mrr={np.mean(test_batches_mrr_list)}",
                file=f,
            )
        # %% ============================================ Testing ends ============================================


if __name__ == "__main__":
    args = parameter_parser()
    args.feature1 = "weight"
    args.feature2 = "poi_catid"
    args.feature3 = "latitude"
    args.feature4 = "longitude"
    train(args)
