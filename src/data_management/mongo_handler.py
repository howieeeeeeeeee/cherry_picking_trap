def get_simulation_results_filters(db):
    collection = db["simulation_results"]
    dic = {
        "K": collection.distinct("K"),
        "lamb": collection.distinct("lamb"),
        "scale": collection.distinct("scale"),
    }
    return dic


def insert_simulation_result(db, simulation_result):
    col = db["simulation_results"]
    col.insert_one(simulation_result)


def get_simulation_results(db, query):
    col = db["simulation_results"]
    return list(col.find(query))
