"""
Modified from ASKCOSv2 src
https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance/-/blob/main/templ_rel_preprocessor.py
"""

import argparse
import csv
import json
import logging
import misc
import multiprocessing
import numpy as np
import os
import random
import templ_rel_parser
import time
from datetime import datetime
from utils import canonicalize_smarts, canonicalize_smiles
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from rdchiral.template_extractor import extract_from_reaction
from generate_retro_templates import process_an_example
from rdkit import Chem, RDLogger
from scipy import sparse
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Tuple
from utils import load_templates_as_dict, save_templates_from_dict, mol_smi_to_count_fp
from multiprocessing import Pool


def _gen_product_fp(task: Tuple[str, int, int]) -> Tuple[sparse.csr_matrix, str]:
    line, radius, fp_size = task
    rxn_with_template = json.loads(line.strip())
    p_smi = rxn_with_template["rxn_smiles"].split(">")[-1]

    p_smi = canonicalize_smiles(p_smi, remove_atom_number=True)
    try:
        product_fp = mol_smi_to_count_fp(mol_smi=p_smi, radius=radius, fp_size=fp_size)
    except:
        logging.info(
            "Error when converting smi to count fingerprint. "
            "Setting it to zero vector."
        )
        count_fp = np.zeros((1, fp_size), dtype=np.int32)
        product_fp = sparse.csr_matrix(count_fp, dtype="int32")

    canon_reaction_smarts = rxn_with_template["canon_reaction_smarts"]

    return product_fp, canon_reaction_smarts


def get_tpl(task: Tuple[int, Dict[str, str]]) -> Tuple[int, Dict[str, str]]:
    i, rxn = task
    rxn_id = rxn["id"]
    r_smi, _, p_smi = rxn["rxn_smiles"].strip().split(">")

    reaction = {"_id": rxn_id, "reactants": r_smi, "products": p_smi}
    try:
        with misc.BlockPrint():
            # template = extract_from_reaction(reaction)
            template = process_an_example(rxn["rxn_smiles"], super_general=True)
        # p_templ = canonicalize_smarts(template["products"])
        # r_templ = canonicalize_smarts(template["reactants"])
        p_templ = canonicalize_smarts(template.split(">>")[0])
        r_templ = canonicalize_smarts(template.split(">>")[1])

        # Note: "reaction_smarts" is actually: p_temp >> r_temp!
        canon_templ = p_templ + ">>" + r_templ
    except:
        canon_templ = ""
    rxn_with_template = rxn
    rxn_with_template["canon_reaction_smarts"] = canon_templ

    return i, rxn_with_template


def dep(args):
    i, rxn = args
    if i % 10_000 == 0:
        print("{} templates processed.".format(i))
    r_smi, _, p_smi = rxn["rxn_smiles"].strip().split(">")
    canon_r_smi = canonicalize_smiles(r_smi, remove_atom_number=True)
    canon_p_smi = canonicalize_smiles(p_smi, remove_atom_number=True)
    canon_rxn_smi = f"{canon_r_smi}>>{canon_p_smi}"
    return rxn, canon_rxn_smi


def _deduplicate(
    rxns: Iterable[Dict[str, str]], num_workers: int
) -> List[Dict[str, str]]:
    p = Pool(num_workers)

    canon_rxn_smis = set()
    dedupped_rxns = []
    for rxn, canon_rxn_smi in tqdm(p.imap(dep, ((args) for args in enumerate(rxns)))):
        if canon_rxn_smi in canon_rxn_smis:
            continue
        else:
            canon_rxn_smis.add(canon_rxn_smi)
            dedupped_rxns.append(rxn)

    return dedupped_rxns


def _extract_templates(
    rxns: List[Dict[str, str]], max_workers: int
) -> Tuple[List[Dict[str, str]], Dict[str, Dict[str, Any]], int]:
    _start = time.time()
    rxns_with_template = []
    templates = {}
    failed_count = 0

    with ProcessPool(max_workers=max_workers) as pool:
        # Using pebble to add timeout, as rdchiral could hang
        future = pool.map(get_tpl, enumerate(rxns), timeout=10)
        iterator = future.result()

        # The while True - try/except/StopIteration is just pebble signature
        while True:
            try:
                i, rxn_with_template = next(iterator)
                if i > 0 and i % 10000 == 0:
                    logging.info(
                        f"Processing {i}th reaction, "
                        f"elapsed time: {time.time() - _start: .0f} s"
                    )

                rxn_id = rxn_with_template["id"]
                canon_reaction_smarts = rxn_with_template["canon_reaction_smarts"]
                if canon_reaction_smarts:
                    if canon_reaction_smarts in templates:
                        templates[canon_reaction_smarts]["count"] += 1
                        templates[canon_reaction_smarts]["references"].append(rxn_id)
                    else:
                        templates[canon_reaction_smarts] = {
                            "index": -1,  # placeholder, to be reset after sorting
                            "reaction_smarts": canon_reaction_smarts,
                            "count": 1,
                            "necessary_reagent": "",
                            "intra_only": True,
                            "dimer_only": False,
                            "template_set": "",
                            "references": [rxn_id],
                            "attributes": {"ring_delta": 1.0, "chiral_delta": 0},
                            "_id": "-1",  # placeholder, to be reset after sorting
                        }
                else:
                    failed_count += 1
            except StopIteration:
                break
            except TimeoutError as error:
                logging.info(f"get_tpl() call took more than {error.args} seconds.")
                failed_count += 1
                rxn_with_template = rxns[i]
                rxn_with_template["canon_reaction_smarts"] = ""
            except:
                logging.info("Unknown error for getting template.")
                failed_count += 1
                rxn_with_template = rxns[i]
                rxn_with_template["canon_reaction_smarts"] = ""

            rxns_with_template.append(rxn_with_template)

    # pool.close()
    # pool.join()

    return rxns_with_template, templates, failed_count


def _sort_and_filter_templates(
    templates: Dict[str, Dict[str, Any]], min_freq: int
) -> Dict[str, Dict[str, Any]]:
    sorted_templates = sorted(
        templates.items(), key=lambda _tup: _tup[1]["count"], reverse=True
    )
    filtered_templates = {}
    for i, (canon_templ, metadata) in enumerate(sorted_templates):
        if metadata["count"] < min_freq:
            break
        metadata["index"] = i
        metadata["_id"] = str(i)
        filtered_templates[canon_templ] = metadata

    return filtered_templates


class TemplRelProcessor:
    """Class for Template Relevance Preprocessing"""

    def __init__(self, args):
        self.args = args

        self.model_name = args.model_name
        self.data_name = args.data_name
        self.log_file = args.log_file
        self.all_reaction_file = args.all_reaction_file
        self.train_file = args.train_file
        self.val_file = args.val_file
        self.test_file = args.test_file
        self.processed_data_path = args.processed_data_path
        self.num_cores = args.num_cores
        self.min_freq = args.min_freq

        os.makedirs(self.processed_data_path, exist_ok=True)

        self.is_data_presplit = None

    def preprocess(self) -> None:
        self.check_data_format()
        if self.is_data_presplit:
            self.extract_templates_for_all_split()
        else:
            self.extract_templates_and_split()
        assert all(
            os.path.exists(file)
            for file in [
                os.path.join(
                    self.processed_data_path, "train_rxns_with_template_gen.jsonl"
                ),
                os.path.join(
                    self.processed_data_path, "val_rxns_with_template_gen.jsonl"
                ),
                os.path.join(
                    self.processed_data_path, "test_rxns_with_template_gen.jsonl"
                ),
                os.path.join(self.processed_data_path, "templates_gen.jsonl"),
            ]
        )
        self.featurize()

    def check_data_format(self) -> None:
        """
        Check that all files exists and the data format is correct for the
        first few lines
        """
        check_count = 100

        logging.info(f"Checking the first {check_count} entries for each file")
        assert os.path.exists(self.all_reaction_file) or os.path.exists(
            self.train_file
        ), (
            f"Either the train file ({self.train_file}) "
            f"or the file with all reactions ({self.all_reaction_file}) "
            f"needs to be supplied!"
        )

        for fn in [
            self.train_file,
            self.val_file,
            self.test_file,
            self.all_reaction_file,
        ]:
            if not os.path.exists(fn):
                logging.info(f"{fn} does not exist, skipping format check")
                continue

            with open(fn, "r") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for i, row in enumerate(csv_reader):
                    if i > check_count:
                        break

                    assert (c in row for c in ["id", "rxn_smiles"]), (
                        f"Error processing file {fn} line {i}, ensure columns 'id' "
                        f"and 'rxn_smiles' are included!"
                    )

                    reactants, reagents, products = row["rxn_smiles"].split(">")
                    # simply ensures that SMILES can be parsed
                    Chem.MolFromSmiles(reactants)
                    Chem.MolFromSmiles(products)

        logging.info("Data format check passed")

        self.is_data_presplit = os.path.isfile(self.train_file)

    def extract_templates_and_split(self):
        _start = time.time()
        logging.info(
            f"Data is not presplit. Extracting templates from "
            f"{self.all_reaction_file}.."
        )

        with open(self.all_reaction_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dedupped_rxns = _deduplicate(csv_reader, self.num_cores)

            # extract templates from reactions
            logging.info(
                f"Loaded all reaction SMILES and deduplicated into {len(dedupped_rxns)} reactions."
                f"Parallelizing extraction over {self.num_cores} cores"
            )
            rxns_with_template, templates, failed_count = _extract_templates(
                dedupped_rxns, max_workers=self.num_cores
            )
            logging.info(f"No of rxn where template extraction failed: {failed_count}")

            # filter templates by min_freq and save
            filtered_templates = _sort_and_filter_templates(
                templates, min_freq=self.min_freq
            )
            template_file = os.path.join(
                self.processed_data_path, "templates_gen.jsonl"
            )
            save_templates_from_dict(filtered_templates, template_file)

            # filter reactions by templates
            rxns_with_template = [
                rxn_with_template
                for rxn_with_template in rxns_with_template
                if rxn_with_template["canon_reaction_smarts"]
                and rxn_with_template["canon_reaction_smarts"] in filtered_templates
            ]

            # split
            split_ratio = [float(r) for r in self.args.split_ratio.split(":")]
            split_ratio = [val / sum(split_ratio) for val in split_ratio]
            assert len(split_ratio) == 3
            random.shuffle(rxns_with_template)

            train_count = int(len(rxns_with_template) * split_ratio[0])
            val_count = int(len(rxns_with_template) * split_ratio[1])
            train_rxns = rxns_with_template[:train_count]
            val_rxns = rxns_with_template[train_count : train_count + val_count]
            test_rxns = rxns_with_template[train_count + val_count :]

            for rxns, phase in [
                (train_rxns, "train"),
                (val_rxns, "val"),
                (test_rxns, "test"),
            ]:
                ofn = os.path.join(
                    self.processed_data_path, f"{phase}_rxns_with_template_gen.jsonl"
                )
                with open(ofn, "w") as of:
                    for rxn in rxns:
                        of.write(f"{json.dumps(rxn)}\n")

            logging.info(
                f"Done template extraction, filtering and splitting, "
                f"time: {time.time() - _start: .2f} s"
            )

    def extract_templates_for_all_split(self):
        _start = time.time()
        logging.info(
            f"Data is presplit. Extracting templates from " f"{self.train_file}.."
        )

        with open(self.train_file, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            dedupped_rxns = _deduplicate(csv_reader, args.num_cores)

            # extract templates from train reactions
            logging.info(
                f"Loaded all train reaction SMILES and deduplicated into {len(dedupped_rxns)} reactions."
                f"Parallelizing extraction over {self.num_cores} cores"
            )
            train_rxns_with_template, templates, failed_count = _extract_templates(
                dedupped_rxns, max_workers=self.num_cores
            )
            logging.info(
                f"No of rxn where template extraction failed: {failed_count}"
                f"Total number of train rxns with template: {len(train_rxns_with_template)}"
            )

            # filter templates by min_freq and save
            filtered_templates = _sort_and_filter_templates(
                templates, min_freq=self.min_freq
            )
            template_file = os.path.join(
                self.processed_data_path, "templates_gen.jsonl"
            )
            save_templates_from_dict(filtered_templates, template_file)

            # filter reactions by templates
            train_rxns_with_template = [
                rxn_with_template
                for rxn_with_template in train_rxns_with_template
                if rxn_with_template["canon_reaction_smarts"]
                and rxn_with_template["canon_reaction_smarts"] in filtered_templates
            ]

            # save filtered train reactions
            ofn = os.path.join(
                self.processed_data_path, "train_rxns_with_template_gen.jsonl"
            )
            with open(ofn, "w") as of:
                for rxn in train_rxns_with_template:
                    of.write(f"{json.dumps(rxn)}\n")

            # for val and test we'll keep all reactions
            for file, phase in [(self.val_file, "val"), (self.test_file, "test")]:
                with open(file, "r") as csv_file:
                    csv_reader = csv.DictReader(csv_file)

                    # extract templates from train reactions
                    logging.info(
                        f"Loaded all reaction SMILES from {file}."
                        f"Parallelizing extraction over {self.num_cores} cores"
                    )
                    rxns_with_template, _, failed_count = _extract_templates(
                        list(csv_reader), max_workers=self.num_cores
                    )
                    logging.info(
                        f"No of rxn where template extraction failed: {failed_count}"
                    )

                    ofn = os.path.join(
                        self.processed_data_path,
                        f"{phase}_rxns_with_template_gen.jsonl",
                    )
                    with open(ofn, "w") as of:
                        for rxn in rxns_with_template:
                            of.write(f"{json.dumps(rxn)}\n")

            logging.info(
                f"Done template extraction, time: {time.time() - _start: .2f} s"
            )

    def featurize(self):
        logging.info("(Re-)loading templates for featurization")
        templates = load_templates_as_dict(
            template_file=os.path.join(self.processed_data_path, "templates_gen.jsonl")
        )
        for phase in ["train", "val", "test"]:
            fn = os.path.join(
                self.processed_data_path, f"{phase}_rxns_with_template_gen.jsonl"
            )
            logging.info(
                f"Loading rxns_with_template from {fn} "
                f"and featurizing over {self.num_cores} cores"
            )
            pool = multiprocessing.Pool(self.num_cores)

            product_fps = []
            labels = []

            with open(fn, "r") as f:
                lines = f.readlines()
            tasks = [(line, self.args.radius, self.args.fp_size) for line in lines]
            for result in tqdm(
                pool.imap(_gen_product_fp, tasks),
                total=len(tasks),
                desc="Processing line ",
            ):
                product_fp, canon_reaction_smarts = result
                product_fps.append(product_fp)

                if canon_reaction_smarts and canon_reaction_smarts in templates:
                    label = templates[canon_reaction_smarts]["index"]
                else:
                    label = -1
                labels.append(label)

            pool.close()
            pool.join()

            product_fps = sparse.vstack(product_fps)
            sparse.save_npz(
                os.path.join(self.processed_data_path, f"product_fps_{phase}.npz"),
                product_fps,
            )
            np.save(
                os.path.join(self.processed_data_path, f"labels_{phase}.npy"),
                np.asarray(labels),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("template_relevance")
    templ_rel_parser.add_model_opts(parser)
    templ_rel_parser.add_preprocess_opts(parser)
    args, unknown = parser.parse_known_args()

    # logger setup
    RDLogger.DisableLog("rdApp.warning")
    os.makedirs("./logs/preprocess", exist_ok=True)
    dt = datetime.strftime(datetime.now(), "%y%m%d-%H%Mh")
    args.log_file = f"./logs/preprocess/{args.log_file}.{dt}"
    logger = misc.setup_logger(args.log_file)
    misc.log_args(args, message="Logging arguments")

    start = time.time()
    random.seed(args.seed)

    processor = TemplRelProcessor(args)
    processor.check_data_format()
    processor.preprocess()

    logging.info(f"Preprocessing done, total time: {time.time() - start: .2f} s")
