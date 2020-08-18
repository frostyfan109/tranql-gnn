import requests
from tqdm import tqdm

def make_queries(queries, api_url, verbose=False):
    results = []
    for i, query in enumerate(queries, 1):
        print(f"Making TranQL query {i}")
        res = requests.post(
            f"{api_url}/tranql/query?asynchronous=true",
            headers={
                "accept": "application/json",
                "content-type": "text/plain"
            },
            data=query
        )
        if res.ok:
            results.append({
                "query": query,
                "response": res.json()
            })
        else:
            print(f"Query \"{query}\" failed with HTTP {res.status_code}: {res.reason}")

    return results


class QuerySet:
    def __init__(self, data):
        # Metadata (optional)
        metadata = data.get("info", {})
        self.name = metadata.get("name", "<untitled>")
        self.description = metadata.get("description", "<none>")

        def get_required_key(key):
            try:
                return data[key]
            except KeyError:
                raise Exception(f'Query set "{self.name}" missing required key "{key}"')

        # Required
        self.queries = get_required_key("queries")

if __name__ == "__main__":
    import os
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description='Creates response sets for every query set in "query_sets"')
    parser.add_argument(
        "-r",
        "--remake",
        help="Runs every query set regardless of if its response set has already been created",
        action="store_true"
    )
    parser.add_argument(
        "-a",
        "--api",
        help="Specify the url for the TranQL API.",
        default="http://localhost:8001"
    )

    args = parser.parse_args()
    remake = args.remake
    api_base = args.api

    query_set_path = os.path.join("data", "query_sets")
    response_set_path = os.path.join("data", "response_sets")
    def check_response_set_exists(name):
        return os.path.isfile(os.path.join(response_set_path, name))
    for file_name in os.listdir(query_set_path):
        response_set_exists = check_response_set_exists(file_name)
        # If remake option is passed in, remake the response set regardless of if it exists or not
        # Otherwise, only generate the response set if it hasn't already been made
        if remake or not response_set_exists:
            with open(os.path.join(query_set_path, file_name), "r") as file:
                query_set = QuerySet(yaml.safe_load(file))
                print(f"""Processing query set "{query_set.name}"
Description: {query_set.description}""")

                queries = query_set.queries
                results = make_queries(queries, api_base)
                # print("Completed queries on query set.", "\n")
                with open(os.path.join(response_set_path, file_name), "w+") as output_file:
                    yaml.safe_dump(
                        results,
                        output_file,
                        default_flow_style=False
                    )
