import pandas as pd
from typing import Dict, Any


def get_simulation_data(
    db, lamb: float, scale: float, K_range: tuple[int, int]
) -> pd.DataFrame:
    """
    Get simulation data from MongoDB and convert to DataFrame.

    Args:
        db: MongoDB database object
        lamb: Lambda parameter value
        scale: Scale parameter value

    Returns:
        pd.DataFrame: DataFrame containing simulation results
    """
    query = {
        "lamb": lamb,
        "scale": scale,
        "K": {"$gte": K_range[0], "$lte": K_range[1]},
    }
    results = list(db["simulation_results"].find(query))

    if not results:
        return pd.DataFrame()

    # Convert to DataFrame and exclude _id field
    df = pd.DataFrame(results).drop("_id", axis=1)

    # Select and sort by K
    df = df.sort_values("K")

    return df
