import argparse
import sys

from mol2vec import features

def do_corpus(args):
    print('Generating corpus file.')
    features.generate_corpus(args.in_file, args.out_file, args.radius, sentence_type='alt', n_jobs=args.n_jobs)
    if args.uncommon:
        print('Finding uncommon identifiers and replacing them with %s.' % args.uncommon)
        features.insert_unk(args.out_file, args.out_file + '_' + args.uncommon, uncommon=args.uncommon,
                            threshold=args.threshold)
    print('Done!')


def do_train(args):
    print('Training word2vec/Mol2vec model.')
    features.train_word2vec_model(args.in_file, args.model, vector_size=args.dimensions, window=args.window,
                                  min_count=args.threshold, n_jobs=args.n_jobs, method=args.method)
    print('Done!')


def do_featurize(args):
    print('Featurizing molecules.')
    features.featurize(args.in_file, args.out_file, args.model, args.radius, args.uncommon)
    print('Done!')


def get_parser():
    parser = argparse.ArgumentParser(
        description="""\
Mol2vec is an unsupervised machine learning approach to learn vector 
representations of molecular substructures. Command line application 
has subcommands to prepare a corpus from molecular data (SDF or SMILES), 
train Mol2vec model and featurize new samples.

    --- Subcommand 'corpus' --- 

    Generates corpus to train Mol2vec model. It generates morgan identifiers 
    (up to selected radius) which represent words (molecules are sentences). 
    Words are ordered in the sentence according to atom order in canonical 
    SMILES (generated when generating corpus) and at each atom starting by 
    identifier at radius 0. Corpus subcommand also optionally replaces rare 
    identifiers with selected string (e.g. UNK) which can be later used to 
    represent completely new substructures (i.e. at featurization step). NOTE: 
    It saves the corpus with replaced uncommon identifiers in separate file 
    with ending "_{selected string to replace uncommon}". Since this is 
    unsupervised method we recommend using as much molecules as possible (e.g. 
    complete ZINC database).
    
    Performance: 
        Corpus generation using 20M compounds with replacement of uncommon 
        identifiers takes 6 hours on 4 cores.
    
    Example:
        To prepare a corpus using radius 1, 4 cores, replace uncommon 
        identifiers that appear <= 3 times with 'UNK' run:
        mol2vec corpus -i mols.smi -o mols.cp -r 1 -j 4 --uncommon UNK --threshold 3
          

    --- Subcommand 'train' ---

    Trains Mol2vec model using previously prepared corpus.
    
    Performance:
        Training the model on 20M sentences takes ~2 hours on 4 cores.
    
    Example:
        To train a Mol2vec model on corpus with replaced uncommon identifiers 
        using Skip-gram, window size 10, generating 300 dimensional vectors and 
        using 4 cores run:
        mol2vec train -i mols.cp_UNK -o model.pkl -d 300 -w 10 -m skip-gram --threshold 3 -j 4
    
    
    --- Subcommand 'featurize' ---

    Featurizes new samples using pre-trained Mol2vec model. It saves the 
    result in CSV file with columns for molecule identifiers, canonical 
    SMILES (generated during featurization) and all potential SD fields from
    input SDF file and finally followed by mol2vec-{0 to n-1} where n is 
    dimensionality of embeddings in the model.  
    
    Example:
        To featurize new samples using pre-trained embeddings and using 
        vector trained on uncommon samples to represent new substructures:
        mol2vec featurize -i new.smi -o new.csv -m model.pkl -r 1 --uncommon UNK


For more detail on individual subcommand run:
    mol2vec $sub-command --help

""", formatter_class=argparse.RawDescriptionHelpFormatter)

    subparsers = parser.add_subparsers(
        title="subcommands",
        help="mol2vec $subcommand --help for details on sub-commands")

    corpus = subparsers.add_parser("corpus",
                                description="""\
Generates corpus to train Mol2vec model.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    corpus.add_argument("--in-file", "-i", metavar="FILE", help="Input SDF, SMI, ISM (can be gzipped).", type=str,
                        required=True)
    corpus.add_argument("--out-file", "-o", metavar="FILE", help="Output corpus", type=str, required=True)
    corpus.add_argument("--radius", "-r", metavar="RADIUS", help="Max radius of Morgan substructures (recommended: 1).",
                        type=int, required=True)
    corpus.add_argument("--n-jobs", "-j", metavar="INT", help="Number of CPU cores to use (optional, default: 1).",
                        default=1, type=int,)
    corpus.add_argument("--uncommon", metavar="UNK", help="String to replace uncommon identifiers (optional). Will \
                        generate a second corpus file with suffix '_{selected string}'",
                        type=str)
    corpus.add_argument("--threshold", metavar="INT", help="Number of identifier occurrences to consider it uncommon \
                        (optional, default: 3).", type=int, default=3)
    corpus.set_defaults(func=do_corpus)

    train = subparsers.add_parser("train",
                                  description="""\
Trains Mol2vec model.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    train.add_argument("--in-file", "-i", metavar="FILE", help="Input corpus (from corpus subcommand).", type=str,
                       required=True)
    train.add_argument("--model", "-o", metavar="PICKLE", help="Output PICKLE to save Mol2vec model.", type=str,
                       required=True)
    train.add_argument("--dimensions", "-d", metavar="INT", help="Dimensions of final vector embeddings (default: 300).",
                       type=int, default=300)
    train.add_argument("--window", "-w", metavar="INT", help="Window size (default: 10).", type=int,
                       default=10)
    train.add_argument("--method", "-m", metavar="STR", help="Word2vec method to use (options: cbow or skip-gram, \
                        default:skip-gram.", choices=['cbow', 'skip-gram'], default='skip-gram')
    train.add_argument("--threshold", metavar="INT", help="Number of identifier occurrences to consider it uncommon \
                            (optional, default: 3). Does not generate embeddings for words that occur less than \
                            'threshold' times", type=int, default=3)
    train.add_argument("--n-jobs", "-j", metavar="INT", help="Number of CPU cores to use (optional, default: 1).",
                       default=1, type=int,)
    train.set_defaults(func=do_train)

    featurize = subparsers.add_parser("featurize",
                                    description="""\
Featurizes new samples.
""", formatter_class=argparse.RawDescriptionHelpFormatter)
    featurize.add_argument("--in-file", "-i", metavar="FILE", help="Input SDF, SMI, ISM (can be gzipped).", type=str,
                           required=True)
    featurize.add_argument("--out-file", "-o", metavar="CSV", help="Output CSV with features as columns", type=str,
                           required=True)
    featurize.add_argument("--model", "-m", metavar="PICKLE", help="PICKLE containing Mol2vec model", type=str,
                           required=True)
    featurize.add_argument("--radius", "-r", metavar="RADIUS", help="Max radius of Morgan substructures", type=int,
                           required=True)
    featurize.add_argument("--uncommon", metavar="UNK", help="String to used to represent uncommon identifiers when \
                            Mol2vec model was trained (optional). Vector trained on uncommon identifiers will be used \
                            to represent new identifiers (substructures). If it is not provided then new identifiers \
                            are skipped", type=str)
    featurize.set_defaults(func=do_featurize)

    return parser


def run():
    parser = get_parser()
    if len(sys.argv) == 1:
        argv = ['-h']
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    run()