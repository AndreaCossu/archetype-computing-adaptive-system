import argparse
import torch
import numpy as np
from typing import Optional
from acds.archetypes import RandomizedOscillatorsNetwork
from att_dim_experiments.utils import  load_results
import pandas as pd
from pandas import DataFrame
from skdim.id import lPCA, CorrInt
def compute_corr_dim(trajectory: np.ndarray, k1=10,  k2=20, transient=1000) -> Optional[list[float]]:
    print(f"Computing correlation dimension for trajectory of shape {trajectory.shape}")
    corr_dim_values = []
    for i in range(trajectory.shape[1]): # for each module in the network
        corr_dim_estimator = CorrInt(k1=k1, k2=k2)
        try:
            traj_i = trajectory[:, i]
            corr_dim = corr_dim_estimator.fit_transform(traj_i)
            corr_dim_values.append(corr_dim)
        except Exception as e:
            print(f"Error computing correlation dimension: {e}")
            return None
    return corr_dim_values



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trajectory_path",
        type=str,
        default='trajectories/cycle_trajectories_collection.pkl',
        help="Path to retrieve the generated trajectory.",
    )

    args = parser.parse_args()  

    data = load_results(args.trajectory_path)
    print(type(data))

    #df = DataFrame(data)
    meta = dict(data['metadata'])
    df = DataFrame([
    {**t, **h}
    for t, h in zip(data["trajectories"], data["hyperparameters"])
    ])
    corr_dim = lambda traj: compute_corr_dim(traj)
    print(df['h'][0].shape)
    df['corr_dim'] = df['h'].map(corr_dim)
    print(df.head())

    df.to_pickle(args.trajectory_path.replace('.pkl', '_with_corrdim.pkl'))