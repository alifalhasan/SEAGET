import numpy as np

from scipy.sparse.linalg import eigsh


def calculate_laplacian_matrix(adj_mat, mat_type):
    n_vertex = adj_mat.shape[0]

    # Calculate the degree matrix (diagonal matrix with node degrees)
    deg_mat = np.asmatrix(np.diag(np.sum(adj_mat, axis=1)))

    # Convert input matrices to NumPy matrices for efficient operations
    adj_mat = np.asmatrix(adj_mat)
    id_mat = np.asmatrix(np.identity(n_vertex))

    if mat_type == "com_lap_mat":
        # Combinatorial Laplacian
        com_lap_mat = deg_mat - adj_mat
        return com_lap_mat
    elif mat_type == "wid_rw_normd_lap_mat":
        # Random walk normalized Laplacian (for ChebConv)
        rw_lap_mat = np.matmul(
            np.linalg.matrix_power(deg_mat, -1), adj_mat
        )  # Calculate random walk Laplacian
        rw_normd_lap_mat = id_mat - rw_lap_mat  # Normalize using identity matrix
        lambda_max_rw = eigsh(rw_lap_mat, k=1, which="LM", return_eigenvectors=False)[
            0
        ]  # Find largest eigenvalue
        wid_rw_normd_lap_mat = (
            2 * rw_normd_lap_mat / lambda_max_rw - id_mat
        )  # Rescale for ChebConv
        return wid_rw_normd_lap_mat

    elif mat_type == "hat_rw_normd_lap_mat":
        # Rescaled random walk normalized Laplacian (for GCNConv)
        wid_deg_mat = deg_mat + id_mat  # Add self-loops to degree matrix
        wid_adj_mat = adj_mat + id_mat  # Add self-loops to adjacency matrix
        hat_rw_normd_lap_mat = np.matmul(
            np.linalg.matrix_power(wid_deg_mat, -1), wid_adj_mat
        )  # Calculate Laplacian
        return hat_rw_normd_lap_mat
    else:
        raise ValueError(f"ERROR: {mat_type} is unknown.")


def masked_mse_loss(input, target, mask_value=-1):
    """
    Calculates the mean squared error loss between input and target,
    ignoring elements where the target is equal to the mask value.

    Args:
        input: A tensor of predicted values.
        target: A tensor of target values.
        mask_value: The value in the target tensor to be ignored (default: -1).

    Returns:
        The masked mean squared error loss.
    """

    # Create a mask indicating elements to exclude from the loss calculation
    mask = target == mask_value

    # Invert the mask to select valid elements
    inverted_mask = ~mask

    # Apply the mask to both input and target tensors
    # This ensures only corresponding elements are compared
    valid_input = input[inverted_mask]
    valid_target = target[inverted_mask]

    # Calculate the squared difference between valid elements
    squared_diff = (valid_input - valid_target) ** 2

    # Calculate the mean of the squared differences (the MSE loss)
    loss = squared_diff.mean()

    return loss


def top_k_acc(y_true_seq, y_pred_seq, k):
    """
    Calculates the top-k accuracy.

    Args:
        y_true_seq: A list of true labels (binary vectors).
        y_pred_seq: A list of predicted scores for each label.
        k: The number of top predictions to consider.

    Returns:
        The top-k accuracy (fraction of samples where the true label is in the top-k predictions).
    """

    hit = 0  # Initialize hit counter

    # Iterate through each example (true label and predicted scores)
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        # Get the indices of the top-k predicted scores (in descending order)
        top_k_rec = y_pred.argsort()[-k:][::-1]

        # Check if the true label is within the top-k predicted indices
        idx = np.where(top_k_rec == y_true)[0]
        if len(idx) != 0:  # True label is in the top-k
            hit += 1  # Increment hit counter

    # Calculate and return the overall top-k accuracy
    return hit / len(y_true_seq)


def mAP_metric(y_true_seq, y_pred_seq, k):
    """
    Calculates the mean Average Precision (mAP) metric for next POI recommendation.
    There's typically only one correct POI to recommend for a given user at a given time. This means that there's only one positive sample, making it difficult to define precision meaningfully.
    The definition of mAP being used in the code is based on a research paper titled "Personalized Long- and Short-term Preference Learning for Next POI Recommendation."

    Args:
        y_true_seq: A list of true labels for each sequence.
        y_pred_seq: A list of predicted scores for each sequence.
        k: The number of top recommendations to consider.

    Returns:
        The mAP score.
    """

    rlt = 0  # Initialize the result variable

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        # Get the top k recommended POIs based on predicted scores
        rec_list = y_pred.argsort()[-k:][
            ::-1
        ]  # Sort scores in descending order and get top k indices

        # Find the rank of the true POI in the recommended list
        r_idx = np.where(rec_list == y_true)[
            0
        ]  # Get indices where the true label matches recommendations

        if len(r_idx) != 0:  # True POI found in the top k recommendations
            # Calculate the precision at the rank of the true POI
            precision_at_rank = 1 / (r_idx[0] + 1)  # 1 divided by the rank + 1
            rlt += precision_at_rank  # Add to the accumulated result

    # Calculate the mean Average Precision (mAP)
    mAP = rlt / len(y_true_seq)  # Average precision across all sequences

    return mAP


def MRR_metric(y_true_seq, y_pred_seq):
    """
    Calculates the Mean Reciprocal Rank (MRR) metric.

    Args:
        y_true_seq: A list of true labels for each sequence.
        y_pred_seq: A list of predicted scores for each sequence.

    Returns:
        The MRR score.
    """

    rlt = 0  # Initialize the result variable to accumulate reciprocal ranks

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        # Get the recommended list in descending order of predicted scores
        rec_list = y_pred.argsort()[-len(y_pred) :][::-1]  # Sort and get all indices

        # Find the rank of the true label in the recommended list
        r_idx = np.where(rec_list == y_true)[0][
            0
        ]  # Get the first index where it matches

        # Calculate the reciprocal rank (1 divided by the rank + 1)
        reciprocal_rank = 1 / (r_idx + 1)

        # Add the reciprocal rank to the accumulated result
        rlt += reciprocal_rank

    # Calculate the Mean Reciprocal Rank (MRR)
    MRR = rlt / len(y_true_seq)  # Average of reciprocal ranks across all sequences

    return MRR


def top_k_acc_last_timestep(y_true_seq, y_pred_seq, k):
    """
    Calculates top-k accuracy at the last timestep for next POI recommendation.

    Args:
        y_true_seq: A list of true labels for each sequence.
        y_pred_seq: A list of predicted scores for each sequence.
        k: The number of top recommendations to consider.

    Returns:
        1 if the true POI is in the top k recommendations at the last timestep,
        0 otherwise.
    """

    # Get the true label and predicted scores at the last timestep
    y_true = y_true_seq[-1]  # True label for the last POI
    y_pred = y_pred_seq[-1]  # Predicted scores for the last POI

    # Get the top k recommended POIs based on predicted scores
    top_k_rec = y_pred.argsort()[-k:][
        ::-1
    ]  # Sort scores in descending order and get top k indices

    # Check if the true POI is in the top k recommendations
    idx = np.where(top_k_rec == y_true)[
        0
    ]  # Find indices where the true label matches recommendations

    # Return 1 if the true POI is found in the top k, 0 otherwise
    if len(idx) != 0:  # True POI found in the top k recommendations
        return 1
    else:
        return 0


def mAP_metric_last_timestep(y_true_seq, y_pred_seq, k):
    """
    Calculates the mean Average Precision (mAP) metric for next POI recommendation
    based on the last timestep predictions.

    Args:
        y_true_seq: A list of true labels for each sequence.
        y_pred_seq: A list of predicted scores for each sequence.
        k: The number of top recommendations to consider.

    Returns:
        The mAP score for the last timestep.
    """

    # Extract true label and predicted scores for the last timestep
    y_true = y_true_seq[-1]  # True label for the last timestep
    y_pred = y_pred_seq[-1]  # Predicted scores for the last timestep

    # Get the top k recommended POIs based on predicted scores
    rec_list = y_pred.argsort()[-k:][
        ::-1
    ]  # Sort scores in descending order and get top k indices

    # Find the rank of the true POI in the recommended list
    r_idx = np.where(rec_list == y_true)[
        0
    ]  # Get indices where the true label matches recommendations

    # Calculate the precision at the rank of the true POI
    if len(r_idx) != 0:  # True POI found in the top k recommendations
        return 1 / (r_idx[0] + 1)  # 1 divided by the rank + 1
    else:
        return 0  # True POI not found in the top k


def MRR_metric_last_timestep(y_true_seq, y_pred_seq):
    """
    Calculates the Mean Reciprocal Rank (MRR) metric for the last timestep in next POI recommendation.

    Args:
        y_true_seq: A list of true labels for each sequence.
        y_pred_seq: A list of predicted scores for each sequence.

    Returns:
        The MRR score for the last timestep.
    """

    # Extract true label and predicted scores for the last timestep
    y_true = y_true_seq[-1]  # True label for the last item
    y_pred = y_pred_seq[-1]  # Predicted scores for the last item

    # Get the recommended list based on predicted scores (in descending order)
    rec_list = y_pred.argsort()[-len(y_pred) :][
        ::-1
    ]  # Sort indices in descending order of scores

    # Find the rank of the true POI in the recommended list
    r_idx = np.where(rec_list == y_true)[0][
        0
    ]  # Get the first index where the true label matches

    # Calculate the reciprocal of the rank (MRR)
    mrr = 1 / (r_idx + 1)  # 1 divided by the rank + 1

    return mrr
