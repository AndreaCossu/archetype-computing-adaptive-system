import torch
from torch.utils.data import Dataset
from aeon.datasets import load_from_ts_file

class PenDigitsDataset(Dataset):
    """Torch dataset wrapper for PenDigits ``.ts`` files.

    :param ts_file: Path to an aeon-compatible PenDigits time-series file.
    """

    def __init__(self, ts_file):
        """Load PenDigits samples from an aeon ``.ts`` file.

        :param ts_file: Path to the ``.ts`` file.
        """
        # Load the .ts file into two data frames:
        # X_df with the series and y_df with the labels.
        # In some datasets the label is integrated into X_df.
        X_df, y_df = load_from_ts_file(ts_file)

        # If y_df is not a DataFrame, it may be a label series while X_df may
        # contain one or more series with different dimensions for each sample.
        # Each X_df column can represent one temporal dimension.
        #
        # For PenDigits, a sample can have this shape:
        # X_df.iloc[i, 0] -> the X-coordinate series
        # X_df.iloc[i, 1] -> the Y-coordinate series
        # y_df[i]         -> the label
        # Check the actual structure of the .ts file before adapting this code.
        #
        # Example: if i = 0, X_df.iloc[0, 0] -> pd.Series (for example, 8
        # points), and X_df.iloc[0, 1] -> pd.Series with the same length.
        
        self.data = []
        self.labels = []

        for i in range(len(X_df)):
            # Assume there are exactly two columns: X and Y.
            coords_x = X_df[i, 0]  # array of length 8
            coords_y = X_df[i, 1]  # array of length 8

            # Create an [8, 2] tensor.
            coords = torch.tensor(list(zip(coords_x, coords_y)), dtype=torch.float)

            # Associated label.
            label = y_df[i]
            
            self.data.append(coords)
            self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_seq = self.data[idx]   # shape [8, 2]
        y = self.labels[idx]
        return x_seq, y
