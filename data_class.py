from torch.utils.data import Dataset
from tqdm.auto import tqdm


class TrajectoryDatasetTrain(Dataset):
    def __init__(self, args, train_df, poi_id2idx_dict):
        self.df = train_df
        self.traj_seqs = []  # traj id: user id + traj no.
        self.season_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(train_df["trajectory_id"].tolist())):
            traj_df = train_df[train_df["trajectory_id"] == traj_id]
            poi_ids = traj_df["POI_id"].to_list()
            poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
            time_feature = traj_df[args.time_feature].to_list()
            season_feature = traj_df[args.season_feature].to_list()

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            if len(input_seq) < args.short_traj_thres:
                continue

            self.traj_seqs.append(traj_id)
            self.season_seqs.append(season_feature[0])
            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (
            self.traj_seqs[index],
            self.season_seqs[index],
            self.input_seqs[index],
            self.label_seqs[index],
        )


class TrajectoryDatasetVal(Dataset):
    def __init__(self, args, df, poi_id2idx_dict, user_id2idx_dict):
        self.df = df
        self.traj_seqs = []
        self.season_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(df["trajectory_id"].tolist())):
            user_id = traj_id.split("_")[0]

            # Ignore user if not in training set
            if user_id not in user_id2idx_dict.keys():
                continue

            # Get POIs idx in this trajectory
            traj_df = df[df["trajectory_id"] == traj_id]
            poi_ids = traj_df["POI_id"].to_list()
            poi_idxs = []
            time_feature = traj_df[args.time_feature].to_list()
            season_feature = traj_df[args.season_feature].to_list()

            for each in poi_ids:
                if each in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[each])
                else:
                    # Ignore poi if not in training set
                    continue

            # Construct input seq and label seq
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            # Ignore seq if too short
            if len(input_seq) < args.short_traj_thres:
                continue

            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(traj_id)
            self.season_seqs.append(season_feature[0])

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (
            self.traj_seqs[index],
            self.season_seqs[index],
            self.input_seqs[index],
            self.label_seqs[index],
        )


class TrajectoryDatasetTest(Dataset):
    def __init__(self, args, df, poi_id2idx_dict, user_id2idx_dict):
        self.df = df
        self.traj_seqs = []
        self.season_seqs = []
        self.input_seqs = []
        self.label_seqs = []

        for traj_id in tqdm(set(df["trajectory_id"].tolist())):
            user_id = traj_id.split("_")[0]

            # Ignore user if not in training set
            if user_id not in user_id2idx_dict.keys():
                continue

            # Get POIs idx in this trajectory
            traj_df = df[df["trajectory_id"] == traj_id]
            poi_ids = traj_df["POI_id"].to_list()
            poi_idxs = []
            time_feature = traj_df[args.time_feature].to_list()
            season_feature = traj_df[args.season_feature].to_list()

            for each in poi_ids:
                if each in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[each])
                else:
                    # Ignore poi if not in training set
                    continue

            # Construct input seq and label seq
            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature[i]))
                label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

            # Ignore seq if too short
            if len(input_seq) < args.short_traj_thres:
                continue

            self.input_seqs.append(input_seq)
            self.label_seqs.append(label_seq)
            self.traj_seqs.append(traj_id)
            self.season_seqs.append(season_feature[0])

    def __len__(self):
        assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
        return len(self.traj_seqs)

    def __getitem__(self, index):
        return (
            self.traj_seqs[index],
            self.season_seqs[index],
            self.input_seqs[index],
            self.label_seqs[index],
        )
