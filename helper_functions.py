import re
import glob
import torch
import pandas as pd
import torch.nn as nn

from pathlib import Path


def increment_path(path, exist_ok=True, sep=""):
    """
    This function takes a path and sep argument.
    It increments the path by adding a number to the end of the path, separated by the sep argument
    """
    # Increment path, i.e. runs/iter --> runs/iter{sep}0, runs/iter{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def generate_active_hours(poi_id2idx_dict, device):
    # Load the main dataset
    df = pd.read_csv(r"data/processed/NYC_train.csv")
    # Load the category data with opening and closing times
    category_data = pd.read_csv(r"data/processed/category_active_hours.csv")

    # Convert POI_id to index
    df["POI_idx"] = df["POI_id"].map(poi_id2idx_dict)

    # Extract hour from local_time
    df["hour"] = pd.to_datetime(df["local_time"]).dt.hour

    # Create a dictionary for category active hours
    category_active_hours = category_data.set_index("Category").to_dict(orient="index")

    # Map each POI to its category
    poi_to_category = (
        df[["POI_id", "POI_catname"]]
        .drop_duplicates()
        .set_index("POI_id")
        .to_dict()["POI_catname"]
    )

    # Initialize active_hours dictionary
    active_hours_dict = {}

    # Set active hours based on category data
    for poi_id, category in poi_to_category.items():
        idx = poi_id2idx_dict[poi_id]
        opening_time = category_active_hours[category]["Opening time"]
        closing_time = category_active_hours[category]["Closing time"]
        active_hours_dict[idx] = {
            "min": torch.tensor(opening_time).to(device),
            "max": torch.tensor(closing_time).to(device),
        }

    return active_hours_dict


def filter_inactive_pois(y_pred_poi_adjusted, y_pred_time, active_hours_dict, device):
    """
    Filters out inactive POIs from predicted probabilities based on active hours.

    Args:
      y_pred_poi_adjusted: Tensor of predicted POI probabilities (batch_size, sequence_length, num_of_pois).
      y_pred_time: Tensor of predicted time values (batch_size, sequence_length).
      active_hours_dict: Dictionary mapping POI index to its min and max active hours.
      device: The device to move the tensors to.

    Returns:
      Tensor of filtered predicted POI probabilities with inactive POIs set to 0.
    """

    batch_size, seq_len, num_pois = y_pred_poi_adjusted.shape

    # Convert y_pred_time to corresponding hours by multiplying with 24
    predicted_hours = (torch.sigmoid(y_pred_time) * 24).long()

    # Create a mask of active POIs at each predicted time (broadcasted across batch and sequence)
    active_mask = torch.zeros(
        (batch_size, seq_len, num_pois), dtype=y_pred_poi_adjusted.dtype, device=device
    )

    for b in range(batch_size):
        for t in range(seq_len):
            predicted_hour = predicted_hours[b, t].item()
            if predicted_hour in active_hours_dict:
                min_hour = active_hours_dict[predicted_hour]["min"].item()
                max_hour = active_hours_dict[predicted_hour]["max"].item()
                active_mask[b, t, min_hour : max_hour + 1] = 1.0

    # Apply the mask to filter out inactive POIs
    return y_pred_poi_adjusted + (-1000.0) * (1 - active_mask)


def input_traj_to_embeddings(
    sample,
    poi_embeddings,
    args,
    poi_idx2cat_idx_dict,
    user_id2idx_dict,
    user_embed_model,
    time_embed_model,
    cat_embed_model,
    embed_fuse_model1,
    embed_fuse_model2,
    embed_fuse_model3,
):
    """
    Converts a trajectory sample into a sequence of fused embeddings.

    Args:
        sample: A trajectory sample, containing a trajectory ID and a list of (POI, time) pairs.
        poi_embeddings: A dictionary of pre-computed POI embeddings.

    Returns:
        A list of fused embeddings for each POI in the trajectory.
    """

    # ----------------------------------
    # 1. Parse the trajectory sample
    # ----------------------------------

    traj_id = sample[0]  # Extract the trajectory ID
    traj_season = sample[1]  # Extract the trajectory season
    input_seq = [each[0] for each in sample[2]]  # Extract POI indices
    input_seq_time = [each[1] for each in sample[2]]  # Extract time values
    input_seq_cat = [
        poi_idx2cat_idx_dict[each] for each in input_seq
    ]  # Get category indices for POIs

    # ----------------------------------
    # 2. Generate season embedding
    # ----------------------------------

    season_embedding = nn.Linear(args.season_embed_dim, 128)(
        torch.randn(12, args.season_embed_dim)[traj_season - 1]
    ).to(device=args.device)

    # ----------------------------------
    # 3. Generate user embedding
    # ----------------------------------

    user_id = traj_id.split("_")[0]  # Extract user ID from trajectory ID
    user_idx = user_id2idx_dict[user_id]  # Get user index
    input = torch.LongTensor([user_idx]).to(
        device=args.device
    )  # Create tensor for user index
    user_embedding = user_embed_model(input)  # Get user embedding from model
    user_embedding = torch.squeeze(user_embedding)  # Remove unnecessary dimensions

    # ----------------------------------
    # 4. Generate fused embeddings for each POI
    # ----------------------------------

    input_seq_embed = []  # List to store fused embeddings
    for idx in range(len(input_seq)):
        # Get individual embeddings
        poi_embedding = poi_embeddings[input_seq[idx]]  # Retrieve POI embedding
        poi_embedding = torch.squeeze(poi_embedding).to(
            device=args.device
        )  # Prepare for fusion

        time_embedding = time_embed_model(
            torch.tensor([input_seq_time[idx]], dtype=torch.float).to(
                device=args.device
            )
        )  # Generate time embedding
        time_embedding = torch.squeeze(time_embedding).to(device=args.device)

        cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(
            device=args.device
        )  # Create tensor for category index
        cat_embedding = cat_embed_model(cat_idx)  # Get category embedding from model
        cat_embedding = torch.squeeze(cat_embedding)

        # Fuse embeddings
        fused_embedding1 = embed_fuse_model1(
            user_embedding, poi_embedding
        )  # Fuse user and POI embeddings
        fused_embedding2 = embed_fuse_model2(
            time_embedding, cat_embedding
        )  # Fuse time and category embeddings
        fused_embedding3 = embed_fuse_model3(
            season_embedding, poi_embedding
        )  # Fuse season and POI embeddings

        # Concatenate fused embeddings
        concat_embedding = torch.cat(
            (fused_embedding1, fused_embedding2, fused_embedding3), dim=-1
        )

        # Store the final fused embedding
        input_seq_embed.append(concat_embedding)

    return input_seq_embed
