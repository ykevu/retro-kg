import numpy as np
import pickle
import torch
from rdchiral.main import rdchiralRunText
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.metrics.pairwise import cosine_similarity
from transE_MLP import TransE


def clear_atom_map(smi):
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.CanonSmiles(Chem.MolToSmiles(mol))


class RetroSim:
    def __init__(self, emb_index, fp_index, template_map, model):
        """
        Initialize the RetroSim model

        args:
            fp_index: path to the file containing product embeddings
            template_map: path to the file mapping product index to templates
        """
        self.emb_index = np.load(emb_index)
        with open(fp_index, "rb") as f:
            self.fp_index = pickle.load(f)
        with open(template_map, "rb") as f:
            self.template_map = pickle.load(f)
        self.model = model

    def _get_fp(self, smi):
        """
        Get the fingerprint of a molecule

        args:
            smi: SMILES representation of the molecule

        returns:
            fingerprint of the molecule
        """
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return fp

    def _get_emb(self, smi):
        """
        Get the embedding of a molecule

        args:
            smi: SMILES representation of the molecule

        returns:
            embedding of the molecule
        """
        fp = self._get_fp(smi)
        emb = (
            self.model.run_layers(torch.tensor(fp).float().unsqueeze(0))
            .detach()
            .numpy()
        )
        return emb

    def _cosine_sim_search(self, fp, top_k=10):
        """
        Search for the k most similar products to the target product
        using cosine similarity

        args:
            fp: fingerprint of the target product
            top_k: number of similar products to return

        returns:
            list of similar products
        """
        cos_sim = cosine_similarity(fp, self.emb_index)
        scores = np.sort(cos_sim[0])[-top_k:][::-1]
        idx = np.argsort(cos_sim[0])[-top_k:][::-1]
        return scores, idx

    def _tanimoto_sim_search(self, fp, top_k=10):
        """
        Search for the k most similar products to the target product
        using Tanimoto similarity

        args:
            fp: fingerprint of the target product
            top_k: number of similar products to return

        returns:
            list of similar products
        """
        tanimoto_sim = DataStructs.BulkTanimotoSimilarity(fp, self.fp_index)
        scores = np.sort(tanimoto_sim)[-top_k:][::-1]
        idx = np.argsort(tanimoto_sim)[-top_k:][::-1]
        return scores, idx

    def _random_search(self, top_k=10):
        """
        Sample k random products

        args:
            top_k: number of random products to return

        returns:
            list of random products
        """
        scores = np.ones(top_k)
        idx = np.random.choice(len(self.emb_index), top_k, replace=False)
        return scores, idx

    def predict_emb(self, target_smi, top_k=10, threshold=0):
        """ """
        emb = self._get_emb(target_smi)
        scores, top_k_idx = self._cosine_sim_search(emb, top_k)
        top_k_rxns = [self.template_map[idx] for idx in top_k_idx]
        flt_rxns = []
        flt_scores = []
        for i, rxns in enumerate(top_k_rxns):
            for rxn in rxns:
                flt_rxns.append(rxn)
                flt_scores.append(scores[i])

        precursors = []
        for i, rxn in enumerate(flt_rxns):
            template = rxn["canon_reaction_smarts"]
            ref_rxn = rxn["rxn_smiles"]
            try:
                template = f"({template.replace('>>', ')>>')}"
                results = rdchiralRunText(template, target_smi)
            except Exception as e:
                print(e)
                results = []
            if not results:
                continue

            scored_results = []
            for reactants in results:
                ref_reactants = ref_rxn.split(">")[0].split(".")
                try:
                    ref_emb = np.mean(
                        [self._get_emb(reactant) for reactant in ref_reactants], axis=0
                    )
                    target_emb = np.mean(
                        [self._get_emb(reactant) for reactant in reactants.split(".")],
                        axis=0,
                    )
                    score = cosine_similarity(ref_emb, target_emb)[0][0]
                    scored_results.append((reactants, score))
                except Exception as e:
                    print(e)
                    continue
            precursors.append((i, scored_results))

        output = []
        precursors_set = set()
        for i, scored_results in precursors:
            rxn = flt_rxns[i]
            prod_score = flt_scores[i]

            for reactants, prec_score in scored_results:
                if reactants in precursors_set:
                    continue
                score = prod_score * prec_score
                output.append((score, reactants, rxn["id"]))
                precursors_set.add(reactants)

        output = sorted(output, key=lambda x: x[0], reverse=True)
        output = [o for o in output if o[0] >= threshold]
        output = output[:top_k]

        return output

    def predict_random(self, target_smi, top_k=10, threshold=0):
        """ """
        scores, top_k_idx = self._random_search(top_k)
        top_k_rxns = [self.template_map[idx] for idx in top_k_idx]
        flt_rxns = []
        flt_scores = []
        for i, rxns in enumerate(top_k_rxns):
            for rxn in rxns:
                flt_rxns.append(rxn)
                flt_scores.append(scores[i])

        precursors = []
        for i, rxn in enumerate(flt_rxns):
            template = rxn["canon_reaction_smarts"]
            try:
                template = f"({template.replace('>>', ')>>')}"
                results = rdchiralRunText(template, target_smi)
            except Exception as e:
                print(e)
                results = []
            if not results:
                continue

            scored_results = []
            for reactants in results:
                scored_results.append((reactants, 1))
            precursors.append((i, scored_results))

        output = []
        precursors_set = set()
        for i, scored_results in precursors:
            rxn = flt_rxns[i]
            prod_score = flt_scores[i]

            for reactants, prec_score in scored_results:
                if reactants in precursors_set:
                    continue
                score = prod_score * prec_score
                output.append((score, reactants, rxn["id"]))
                precursors_set.add(reactants)

        output = sorted(output, key=lambda x: x[0], reverse=True)
        output = [o for o in output if o[0] >= threshold]
        output = output[:top_k]

        return output

    def predict_tanimoto(self, target_smi, top_k=10, threshold=0):
        """ """
        fp = self._get_fp(target_smi)
        scores, top_k_idx = self._tanimoto_sim_search(fp, top_k)
        top_k_rxns = [self.template_map[idx] for idx in top_k_idx]
        flt_rxns = []
        flt_scores = []
        for i, rxns in enumerate(top_k_rxns):
            for rxn in rxns:
                flt_rxns.append(rxn)
                flt_scores.append(scores[i])

        precursors = []
        for i, rxn in enumerate(flt_rxns):
            template = rxn["canon_reaction_smarts"]
            ref_rxn = rxn["rxn_smiles"]
            try:
                template = f"({template.replace('>>', ')>>')}"
                results = rdchiralRunText(template, target_smi)
            except Exception as e:
                print(e)
                results = []
            if not results:
                continue

            scored_results = []
            for reactants in results:
                ref_reactants = ref_rxn.split(">")[0]
                try:
                    ref_emb = self._get_fp(ref_reactants)
                    target_emb = self._get_fp(reactants)
                    score = DataStructs.TanimotoSimilarity(ref_emb, target_emb)
                    scored_results.append((reactants, score))
                except Exception as e:
                    print(e)
                    continue
            precursors.append((i, scored_results))

        output = []
        precursors_set = set()
        for i, scored_results in precursors:
            rxn = flt_rxns[i]
            prod_score = flt_scores[i]

            for reactants, prec_score in scored_results:
                if reactants in precursors_set:
                    continue
                score = prod_score * prec_score
                output.append((score, reactants, rxn["id"]))
                precursors_set.add(reactants)

        output = sorted(output, key=lambda x: x[0], reverse=True)
        output = [o for o in output if o[0] >= threshold]
        output = output[:top_k]

        return output


if __name__ == "__main__":
    # Load TransE model
    model = TransE(
        n_templates=10225, device="cpu", hidden_sizes=[512, 512, 512, 512]
    ).to("cpu")
    model.load_state_dict(torch.load("output/model.pth", map_location="cpu"))
    model.eval()

    # Initialize RetroSim model
    retro_sim = RetroSim(
        emb_index="output/product_embs.npy",
        fp_index="output/product_bvs.pkl",
        template_map="output/pdt2rxn.pkl",
        model=model,
    )

    smi = "[O:2]=[C:1]([NH:7][c:8]1[cH:9][cH:10][c:11]([N+:12](=[O:13])[O-:14])[cH:15][cH:16]1)[C:3]([F:4])([F:5])[F:6]"
    mol = Chem.MolFromSmiles(clear_atom_map(smi))
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    emb = model.run_layers(torch.tensor(fp).float().unsqueeze(0)).detach().numpy()

    print(f"{retro_sim.predict_emb(smi, top_k=10)} \n\n\n")
    print(retro_sim.predict_tanimoto(smi, top_k=10))
