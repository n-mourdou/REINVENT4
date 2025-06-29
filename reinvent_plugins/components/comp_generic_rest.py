"""Generic REST interface from Syngenta"""

from __future__ import annotations

__all__ = ["REST"]
from typing import List, Dict, Any

import requests
import numpy as np
import logging

from pydantic.dataclasses import dataclass

from reinvent_plugins.components.component_results import ComponentResults
from reinvent_plugins.components.add_tag import add_tag
from reinvent_plugins.normalize import normalize_smiles


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    server_url: List[str]
    server_port: List[int]
    server_endpoint: List[str]
    predictor_id: List[str]
    predictor_version: List[str]
    header: List[Dict[str, str]] = None
    reference_smiles: Any = None


DEFAULT_HEADER = {
    "accept": "application/json",
    "Content-Type": "application/json",
}


@add_tag("__component")
class REST:
    def __init__(self, params: Parameters):
        if isinstance(params.reference_smiles, list):
            self.reference_smile = params.reference_smiles[0]  # Convert to string if it's a list
        else:
            self.reference_smile = params.reference_smiles
        self.server_urls = params.server_url
        self.server_ports = params.server_port
        self.server_endpoints = params.server_endpoint
        self.predictor_ids = params.predictor_id
        self.predictor_versions = params.predictor_version
        self.headers = params.header
        # self.reference_smile = params.reference_smiles  # Store the dynamic reference smile

        # needed in the normalize_smiles decorator
        self.smiles_type = "rdkit_smiles"
        self.number_of_endpoints = len(params.server_url)

    @normalize_smiles
    def __call__(self, smilies: List[str]) -> np.array:
        if not self.headers:
            self.headers = DEFAULT_HEADER

        scores = []
        for url, port, endpoint, pred_id, pred_vers, header in zip(
            self.server_urls,
            self.server_ports,
            self.server_endpoints,
            self.predictor_ids,
            self.predictor_versions,
            self.headers,
        ):
            full_url = f"{url}:{port}/{endpoint}"
            # For tanimoto_similarity, inject the dynamic reference_smiles
            if endpoint.lower() == "tanimoto_similarity" and self.reference_smile:
                json_data = [
                    {
                        "input_string": smiles,
                        "query_id": str(i),
                        "params": {"reference_smiles": self.reference_smile}
                    }
                    for i, smiles in enumerate(smilies)
            ]
            else:
                json_data = [
                    {"input_string": smiles, "query_id": str(i)} for i, smiles in enumerate(smilies)
                ]
            params = {"predictor_id": pred_id, "predictor_version": pred_vers, "inp_fmt": "smiles"}
            # Log the full URL and JSON payload before making the request

            response = execute_request(full_url, json_data, header, params)
            results = parse_response(response, len(smilies))
            scores.append(results)

        return ComponentResults(scores)


def execute_request(url, data, header, params) -> dict:
    request = requests.post(url, json=data, headers=header, params=params)

    if request.status_code != 200:
        raise ValueError(
            f"REST component request failed.\n"
            f"Status Code: {request.status_code}\n"
            f"Reason: ({request.reason})\n"
            f"Response content: {request.content}\n"
            f"Response text: {request.text}"
        )

    return request.json()


def parse_response(response_json: dict, data_size: int) -> np.ndarray:
    compounds = response_json["output"]["successes_list"]
    results = np.empty(data_size, dtype=np.float32)
    results[:] = np.nan

    try:
        for compound in compounds:
            try:
                index = int(compound["query_id"])
                results[index] = float(compound["output_value"])
            except (ValueError, TypeError, KeyError):
                pass  # If parsing failed, keep value NaN for this compound and continue.
    finally:
        return results
