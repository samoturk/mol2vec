"""
Features - Main Mol2vec Module
==============================


"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
from gensim.models import word2vec
import timeit
from joblib import Parallel, delayed


class DfVec(object):
    """
    Helper class to store vectors in a pandas DataFrame
    
    Parameters  
    ---------- 
    vec: np.array
    """
    def __init__(self, vec):
        self.vec = vec
        if type(self.vec) != np.ndarray:
            raise TypeError('numpy.ndarray expected, got %s' % type(self.vec))

    def __str__(self):
        return "%s dimensional vector" % str(self.vec.shape)

    __repr__ = __str__

    def __len__(self):
        return len(self.vec)

    _repr_html_ = __str__


class MolSentence:
    """Class for storing mol sentences in pandas DataFrame
    """
    def __init__(self, sentence):
        self.sentence = sentence
        if type(self.sentence[0]) != str:
            raise TypeError('List with strings expected')

    def __len__(self):
        return len(self.sentence)

    def __str__(self):  # String representation
        return 'MolSentence with %i words' % len(self.sentence)

    __repr__ = __str__  # Default representation

    def contains(self, word):
        """Contains (and __contains__) method enables usage of "'Word' in MolSentence"""
        if word in self.sentence:
            return True
        else:
            return False

    __contains__ = contains  # MolSentence.contains('word')

    def __iter__(self):  # Iterate over words (for word in MolSentence:...)
        for x in self.sentence:
            yield x

    _repr_html_ = __str__



def mol2sentence(mol, radius):

    """Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float 
        Fingerprint radius

    Returns
    -------
    identifier sentence
        List with sentences for each radius
    alternating sentence
        Sentence (list) with identifiers from all radii combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # iterate over all atoms and radii
    identifier_sentences = []
    
    for r in radii:  # iterate over radii to get one sentence per radius
        identifiers = []
        for atom in dict_atoms:  # iterate over atoms
            # get one sentence per radius
            identifiers.append(dict_atoms[atom][r])
        identifier_sentences.append(list(map(str, [x for x in identifiers if x])))
    
    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(identifier_sentences), list(alternating_sentence)


def mol2alt_sentence(mol, radius):
    """Same as mol2sentence() expect it only returns the alternating sentence
    Calculates ECFP (Morgan fingerprint) and returns identifiers of substructures as 'sentence' (string).
    Returns a tuple with 1) a list with sentence for each radius and 2) a sentence with identifiers from all radii
    combined.
    NOTE: Words are ALWAYS reordered according to atom order in the input mol object.
    NOTE: Due to the way how Morgan FPs are generated, number of identifiers at each radius is smaller
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float 
        Fingerprint radius
    
    Returns
    -------
    list
        alternating sentence
    combined
    """
    radii = list(range(int(radius) + 1))
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)  # info: dictionary identifier, atom_idx, radius

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: {r: None for r in radii} for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            dict_atoms[atom_idx][radius_at] = element  # {atom number: {fp radius: identifier}}

    # merge identifiers alternating radius to sentence: atom 0 radius0, atom 0 radius 1, etc.
    identifiers_alt = []
    for atom in dict_atoms:  # iterate over atoms
        for r in radii:  # iterate over radii
            identifiers_alt.append(dict_atoms[atom][r])

    alternating_sentence = map(str, [x for x in identifiers_alt if x])

    return list(alternating_sentence)


def _parallel_job(mol, r):
    """Helper function for joblib jobs
    """
    if mol is not None:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
        sentence = mol2alt_sentence(mol, r)
        return " ".join(sentence)


def _read_smi(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield Chem.MolFromSmiles(line.split('\t')[0])


def generate_corpus(in_file, out_file, r, sentence_type='alt', n_jobs=1):

    """Generates corpus file from sdf
    
    Parameters
    ----------
    in_file : str
        Input sdf
    out_file : str
        Outfile name prefix, suffix is either _r0, _r1, etc. or _alt_r1 (max radius in alt sentence)
    r : int
        Radius of morgan fingerprint
    sentence_type : str
        Options:    'all' - generates all corpus files for all types of sentences, 
                    'alt' - generates a corpus file with only combined alternating sentence, 
                    'individual' - generates corpus files for each radius
    n_jobs : int
        Number of cores to use (only 'alt' sentence type is parallelized)

    Returns
    -------
    """

    # File type detection
    in_split = in_file.split('.')
    if in_split[-1].lower() not in ['sdf', 'smi', 'ism', 'gz']:
        raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    gzipped = False
    if in_split[-1].lower() == 'gz':
        gzipped = True
        if in_split[-2].lower() not in ['sdf', 'smi', 'ism']:
            raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')

    file_handles = []
    
    # write only files which contain corpus
    if (sentence_type == 'individual') or (sentence_type == 'all'):
        
        f1 = open(out_file+'_r0.corpus', "w")
        f2 = open(out_file+'_r1.corpus', "w")
        file_handles.append(f1)
        file_handles.append(f2)

    if (sentence_type == 'alt') or (sentence_type == 'all'):
        f3 = open(out_file, "w")
        file_handles.append(f3)
    
    if gzipped:
        import gzip
        if in_split[-2].lower() == 'sdf':
            mols_file = gzip.open(in_file, mode='r')
            suppl = Chem.ForwardSDMolSupplier(mols_file)
        else:
            mols_file = gzip.open(in_file, mode='rt')
            suppl = _read_smi(mols_file)
    else:
        if in_split[-1].lower() == 'sdf':
            suppl = Chem.ForwardSDMolSupplier(in_file)
        else:
            mols_file = open(in_file, mode='rt')
            suppl = _read_smi(mols_file)

    if sentence_type == 'alt':  # This can run parallelized
        result = Parallel(n_jobs=n_jobs, verbose=1)(delayed(_parallel_job)(mol, r) for mol in suppl)
        for i, line in enumerate(result):
            f3.write(str(line) + '\n')
        print('% molecules successfully processed.')

    else:
        for mol in suppl:
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                mol = Chem.MolFromSmiles(smiles)
                identifier_sentences, alternating_sentence = mol2sentence(mol, r)

                identifier_sentence_r0 = " ".join(identifier_sentences[0])
                identifier_sentence_r1 = " ".join(identifier_sentences[1])
                alternating_sentence_r0r1 = " ".join(alternating_sentence)

                if len(smiles) != 0:
                    if (sentence_type == 'individual') or (sentence_type == 'all'):
                        f1.write(str(identifier_sentence_r0)+'\n')
                        f2.write(str(identifier_sentence_r1)+'\n')

                    if (sentence_type == 'alt') or (sentence_type == 'all'):
                        f3.write(str(alternating_sentence_r0r1)+'\n')

    for fh in file_handles:
        fh.close()


def _read_corpus(file_name):
    while True:
        line = file_name.readline()
        if not line:
            break
        yield line.split()


def insert_unk(corpus, out_corpus, threshold=3, uncommon='UNK'):
    """Handling of uncommon "words" (i.e. identifiers). It finds all least common identifiers (defined by threshold) and
    replaces them by 'uncommon' string.

    Parameters
    ----------
    corpus : str
        Input corpus file
    out_corpus : str
        Outfile corpus file
    threshold : int
        Number of identifier occurrences to consider it uncommon
    uncommon : str
        String to use to replace uncommon words/identifiers

    Returns
    -------
    """
    # Find least common identifiers in corpus
    f = open(corpus)
    unique = {}
    for i, x in tqdm(enumerate(_read_corpus(f)), desc='Counting identifiers in corpus'):
        for identifier in x:
            if identifier not in unique:
                unique[identifier] = 1
            else:
                unique[identifier] += 1
    n_lines = i + 1
    least_common = set([x for x in unique if unique[x] <= threshold])
    f.close()

    f = open(corpus)
    fw = open(out_corpus, mode='w')
    for line in tqdm(_read_corpus(f), total=n_lines, desc='Inserting %s' % uncommon):
        intersection = set(line) & least_common
        if len(intersection) > 0:
            new_line = []
            for item in line:
                if item in least_common:
                    new_line.append(uncommon)
                else:
                    new_line.append(item)
            fw.write(" ".join(new_line) + '\n')
        else:
            fw.write(" ".join(line) + '\n')
    f.close()
    fw.close()


def train_word2vec_model(infile_name, outfile_name=None, vector_size=100, window=10, min_count=3, n_jobs=1,
                         method='skip-gram', **kwargs):
    """Trains word2vec (Mol2vec, ProtVec) model on corpus file extracted from molecule/protein sequences.
    The corpus file is treated as LineSentence corpus (one sentence = one line, words separated by whitespaces)
    
    Parameters
    ----------
    infile_name : str
        Corpus file, e.g. proteins split in n-grams or compound identifier
    outfile_name : str
        Name of output file where word2vec model should be saved
    vector_size : int
        Number of dimensions of vector
    window : int
        Number of words considered as context
    min_count : int
        Number of occurrences a word should have to be considered in training
    n_jobs : int
        Number of cpu cores used for calculation
    method : str
        Method to use in model training. Options cbow and skip-gram, default: skip-gram)
    
    Returns
    -------
    word2vec.Word2Vec
    """
    if method.lower() == 'skip-gram':
        sg = 1
    elif method.lower() == 'cbow':
        sg = 0
    else:
        raise ValueError('skip-gram or cbow are only valid options')
  
    start = timeit.default_timer()
    corpus = word2vec.LineSentence(infile_name)
    model = word2vec.Word2Vec(corpus, size=vector_size, window=window, min_count=min_count, workers=n_jobs, sg=sg,
                              **kwargs)
    if outfile_name:
        model.save(outfile_name)
    
    stop = timeit.default_timer()
    print('Runtime: ', round((stop - start)/60, 2), ' minutes')
    return model
    
    
def remove_salts_solvents(smiles, hac=3):
    """Remove solvents and ions have max 'hac' heavy atoms. This function removes any fragment in molecule that has
    number of heavy atoms <= "hac" and it might not be an actual solvent or salt
    
    Parameters
    ----------
    smiles : str
        SMILES
    hac : int
        Max number of heavy atoms

    Returns
    -------
    str
        smiles
    """
    save = []
    for el in smiles.split("."):
        mol = Chem.MolFromSmiles(str(el))
        if mol.GetNumHeavyAtoms() <= hac:
            save.append(mol)
        
    return ".".join([Chem.MolToSmiles(x) for x in save])


def sentences2vec(sentences, model, unseen=None):
    """Generate vectors for each sentence (list) in a list of sentences. Vector is simply a
    sum of vectors for individual words.
    
    Parameters
    ----------
    sentences : list, array
        List with sentences
    model : word2vec.Word2Vec
        Gensim word2vec model
    unseen : None, str
        Keyword for unseen words. If None, those words are skipped.
        https://stats.stackexchange.com/questions/163005/how-to-set-the-dictionary-for-text-analysis-using-neural-networks/163032#163032

    Returns
    -------
    np.array
    """
    keys = set(model.wv.vocab.keys())
    vec = []
    if unseen:
        unseen_vec = model.wv.word_vec(unseen)

    for sentence in sentences:
        if unseen:
            vec.append(sum([model.wv.word_vec(y) if y in set(sentence) & keys
                       else unseen_vec for y in sentence]))
        else:
            vec.append(sum([model.wv.word_vec(y) for y in sentence 
                            if y in set(sentence) & keys]))
    return np.array(vec)


def featurize(in_file, out_file, model_path, r, uncommon=None):
    """Featurize mols in SDF, SMI.
    SMILES are regenerated with RDKit to get canonical SMILES without chirality information.

    Parameters
    ----------
    in_file : str
        Input SDF, SMI, ISM (or GZ)
    out_file : str
        Output csv
    model_path : str
        File path to pre-trained Gensim word2vec model
    r : int
        Radius of morgan fingerprint
    uncommon : str
        String to used to replace uncommon words/identifiers while training. Vector obtained for 'uncommon' will be used
        to encode new (unseen) identifiers

    Returns
    -------
    """
    # Load the model
    word2vec_model = word2vec.Word2Vec.load(model_path)
    if uncommon:
        try:
            word2vec_model[uncommon]
        except KeyError:
            raise KeyError('Selected word for uncommon: %s not in vocabulary' % uncommon)

    # File type detection
    in_split = in_file.split('.')
    f_type = in_split[-1].lower()
    if f_type not in ['sdf', 'smi', 'ism', 'gz']:
        raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
    if f_type == 'gz':
        if in_split[-2].lower() not in ['sdf', 'smi', 'ism']:
            raise ValueError('File extension not supported (sdf, smi, ism, sdf.gz, smi.gz)')
        else:
            f_type = in_split[-2].lower()

    print('Loading molecules.')
    if f_type == 'sdf':
        df = PandasTools.LoadSDF(in_file)
        print("Keeping only molecules that can be processed by RDKit.")
        df = df[df['ROMol'].notnull()]
        df['Smiles'] = df['ROMol'].map(Chem.MolToSmiles)
    else:
        df = pd.read_csv(in_file, delimiter='\t', usecols=[0, 1], names=['Smiles', 'ID'])  # Assume <tab> separated
        PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Smiles')
        print("Keeping only molecules that can be processed by RDKit.")
        df = df[df['ROMol'].notnull()]
        df['Smiles'] = df['ROMol'].map(Chem.MolToSmiles)  # Recreate SMILES

    print('Featurizing molecules.')
    df['mol-sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], r)), axis=1)
    vectors = sentences2vec(df['mol-sentence'], word2vec_model, unseen=uncommon)
    df_vec = pd.DataFrame(vectors, columns=['mol2vec-%03i' % x for x in range(vectors.shape[1])])
    df_vec.index = df.index
    df = df.join(df_vec)

    df.drop(['ROMol', 'mol-sentence'], axis=1).to_csv(out_file)
