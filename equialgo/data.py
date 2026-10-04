"""Loading, group definition and the held-out split of the historical data."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "figures"
RUNS_DIR = ROOT / "runs"

ID = "id_candidat"
LABEL = "decision_octroi"  # the past committee's decision, not ground truth
REGION = "region_administrative"
GROUP = "groupe"
R_SCORE = "cote_r_equivalent"
WEIGHT = "poids_entrainement"  # training weight of a row, set by some targets

CENTRE = "Centre"
REMOTE = "Eloignee"
GROUPS = [CENTRE, REMOTE]
REMOTE_REGIONS = ["Bas-Saint-Laurent", "Cote-Nord", "Gaspesie-Iles-de-la-Madeleine"]
REGIONS = ["Montreal", "Capitale-Nationale"] + REMOTE_REGIONS

CATEGORICAL = ["programme_etudes", REGION, "code_postal_3"]


def add_group(df):
    """Return a copy with the two-block `groupe` column the scorer uses."""
    out = df.copy()
    out[GROUP] = np.where(out[REGION].isin(REMOTE_REGIONS), REMOTE, CENTRE)
    return out


def load_history():
    """The 10,000 historical applications, with the committee's decision."""
    return add_group(pd.read_csv(DATA_DIR / "donnees_demandes.csv"))


def load_candidates():
    """The 4,000 unlabeled applicants that get graded."""
    return add_group(pd.read_csv(DATA_DIR / "candidats_evaluation.csv"))


def split_history(history, seed=42, test_size=0.3):
    """Train/held-out split of the history, the same one the baseline notebook uses."""
    train, test = train_test_split(
        history, test_size=test_size, random_state=seed, stratify=history[LABEL]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
