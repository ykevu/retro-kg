import json
import numpy as np
import torch
from rdkit import Chem
from retro_sim import RetroSim
from tqdm import tqdm
from transE_MLP import TransE


def clear_atom_map(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


def test_rxn_emb(rxn, retro_sim, top_k=1):
    rxn_smi = rxn["rxn_smiles"]
    reactants, _, products = rxn_smi.split(">")
    target = clear_atom_map(products)
    reactants = {clear_atom_map(reactant) for reactant in reactants.split(".")}
    preds = retro_sim.predict_emb(target, top_k=top_k)
    for pred in preds:
        precursors = set(pred[1].split("."))
        if precursors == reactants:
            return True
    return False


def test_rxn_tanimoto(rxn, retro_sim, top_k=1):
    rxn_smi = rxn["rxn_smiles"]
    reactants, _, products = rxn_smi.split(">")
    target = clear_atom_map(products)
    reactants = {clear_atom_map(reactant) for reactant in reactants.split(".")}
    preds = retro_sim.predict_tanimoto(target, top_k=top_k)
    for pred in preds:
        precursors = set(pred[1].split("."))
        if precursors == reactants:
            return True
    return False


def test_rxn_random(rxn, retro_sim, top_k=1):
    rxn_smi = rxn["rxn_smiles"]
    reactants, _, products = rxn_smi.split(">")
    reactants = {clear_atom_map(reactant) for reactant in reactants.split(".")}
    product = clear_atom_map(products)
    preds = retro_sim.predict_random(product, top_k=top_k)
    for pred in preds:
        precursors = set(pred[1].split("."))
        if precursors == reactants:
            return True
    return False


if __name__ == "__main__":
    # Load test set
    with open("data/test_rxns_with_template.jsonl", "r") as f:
        test_rxns = [json.loads(line) for line in f]

    # Load TransE model
    model = TransE(
        n_templates=2990,
        device="cpu",
        hidden_sizes=[1024, 1024],
        output_dim=64,
    ).to("cpu")
    model.load_state_dict(torch.load("output2/model.pth", map_location="cpu"))
    model.eval()

    # Load RetroSim model
    retro_sim = RetroSim(
        emb_index="output/product_embs.npy",
        fp_index="output/product_bvs.pkl",
        template_map="output/pdt2rxn.pkl",
        model=model,
    )

    top_1_emb = 0
    top_1_tan = 0
    top_1_rand = 0
    top_5_emb = 0
    top_5_tan = 0
    top_5_rand = 0
    top_10_emb = 0
    top_10_tan = 0
    top_10_rand = 0
    top_25_emb = 0
    top_25_tan = 0
    top_25_rand = 0
    top_50_emb = 0
    top_50_tan = 0
    top_50_rand = 0
    for rxn in tqdm(test_rxns):
        # if test_rxn_emb(rxn, retro_sim, top_k=1):
        #     top_1_emb += 1
        # if test_rxn_tanimoto(rxn, retro_sim, top_k=1):
        #     top_1_tan += 1
        if test_rxn_random(rxn, retro_sim, top_k=1):
            top_1_rand += 1
        # if test_rxn_emb(rxn, retro_sim, top_k=5):
        #     top_5_emb += 1
        # if test_rxn_tanimoto(rxn, retro_sim, top_k=5):
        #     top_5_tan += 1
        if test_rxn_random(rxn, retro_sim, top_k=5):
            top_5_rand += 1
        # if test_rxn_emb(rxn, retro_sim, top_k=10):
        #     top_10_emb += 1
        # if test_rxn_tanimoto(rxn, retro_sim, top_k=10):
        #     top_10_tan += 1
        if test_rxn_random(rxn, retro_sim, top_k=10):
            top_10_rand += 1
        # if test_rxn_emb(rxn, retro_sim, top_k=25):
        #     top_25_emb += 1
        # if test_rxn_tanimoto(rxn, retro_sim, top_k=25):
        #     top_25_tan += 1
        if test_rxn_random(rxn, retro_sim, top_k=25):
            top_25_rand += 1
        # if test_rxn_emb(rxn, retro_sim, top_k=50):
        #     top_50_emb += 1
        # if test_rxn_tanimoto(rxn, retro_sim, top_k=50):
        #     top_50_tan += 1
        if test_rxn_random(rxn, retro_sim, top_k=50):
            top_50_rand += 1
    print(f"Top-1 Accuracy Emb: {top_1_emb / len(test_rxns)}")
    print(f"Top-1 Accuracy Tanimoto: {top_1_tan / len(test_rxns)}")
    print(f"Top-1 Accuracy Random: {top_1_rand / len(test_rxns)}")
    print(f"Top-5 Accuracy Emb: {top_5_emb / len(test_rxns)}")
    print(f"Top-5 Accuracy Tanimoto: {top_5_tan / len(test_rxns)}")
    print(f"Top-5 Accuracy Random: {top_5_rand / len(test_rxns)}")
    print(f"Top-10 Accuracy Emb: {top_10_emb / len(test_rxns)}")
    print(f"Top-10 Accuracy Tanimoto: {top_10_tan / len(test_rxns)}")
    print(f"Top-10 Accuracy Random: {top_10_rand / len(test_rxns)}")
    print(f"Top-25 Accuracy Emb: {top_25_emb / len(test_rxns)}")
    print(f"Top-25 Accuracy Tanimoto: {top_25_tan / len(test_rxns)}")
    print(f"Top-25 Accuracy Random: {top_25_rand / len(test_rxns)}")
    print(f"Top-50 Accuracy Emb: {top_50_emb / len(test_rxns)}")
    print(f"Top-50 Accuracy Tanimoto: {top_50_tan / len(test_rxns)}")
    print(f"Top-50 Accuracy Random: {top_50_rand / len(test_rxns)}")
