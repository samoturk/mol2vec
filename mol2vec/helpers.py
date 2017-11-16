"""
Helpers - Mostly plotting functions
===================================
"""

from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import rdDepictor
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_mol(mol, kekulize):
    """Prepare mol for SVG depiction (embed 2D coords)
    """
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc


def mol_to_svg(mol, molSize=(300, 300), kekulize=True, drawer=None, font_size=0.8, **kwargs):
    """Generates a SVG from mol structure.
    
    Inspired by: http://rdkit.blogspot.ch/2016/02/morgan-fingerprint-bit-statistics.html
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    molSize : tuple
    kekulize : bool 
    drawer : funct
        Specify which drawing function to use (default: rdMolDraw2D.MolDraw2DSVG)
    font_size : float
        Atom font size

    Returns
    -------
    IPython.display.SVG
    """
    from IPython.display import SVG    
    
    mc = _prepare_mol(mol, kekulize)
    mol_atoms = [a.GetIdx() for a in mc.GetAtoms()]
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(*molSize)
    drawer.SetFontSize(font_size)
    drawer.DrawMolecule(mc, highlightAtomRadii={x: 0.5 for x in mol_atoms}, **kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:', ''))


def depict_atoms(mol, atom_ids, radii, molSize=(300, 300), atm_color=(0, 1, 0), oth_color=(0.8, 1, 0)):
    """Get a depiction of molecular substructure. Useful for depicting bits in fingerprints.
    
    Inspired by: http://rdkit.blogspot.ch/2016/02/morgan-fingerprint-bit-statistics.html
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    atom_ids : list
        List of atoms to depict
    radii : list
        List of radii - how many atoms around each atom with atom_id to highlight
    molSize : tuple
    atm_color, oth_color : tuple
        Colors of central atoms and surrounding atoms and bonds
    
    Returns
    -------
    IPython.display.SVG
    """
    atoms_to_use = []
    bonds = []
    for atom_id, radius in zip(atom_ids, radii):    
        if radius > 0:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_id)
            bonds += [x for x in env if x not in bonds]
            for b in env:
                atoms_to_use.append(mol.GetBondWithIdx(b).GetBeginAtomIdx())
                atoms_to_use.append(mol.GetBondWithIdx(b).GetEndAtomIdx())
            atoms_to_use = list(set(atoms_to_use))       
        else:
            atoms_to_use.append(atom_id)
            env = None
    if sum(radii) == 0:
        return mol_to_svg(mol, molSize=molSize, highlightBonds=False, highlightAtoms=atoms_to_use,
                          highlightAtomColors={x: atm_color for x in atom_ids})
    else:
        colors = {x: atm_color for x in atom_ids}
        for x in atoms_to_use:
            if x not in atom_ids:
                colors[x] = oth_color
        bond_colors = {b: oth_color for b in bonds}
        return mol_to_svg(mol, molSize=molSize, highlightAtoms=atoms_to_use, highlightAtomColors=colors,
                          highlightBonds=bonds, highlightBondColors=bond_colors)


def depict_identifier(mol, identifier, radius, useFeatures=False, **kwargs):
    """Depict an identifier in Morgan fingerprint.
    
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule
    identifier : int or str
        Feature identifier from Morgan fingerprint
    radius : int
        Radius of Morgan FP
    useFeatures : bool
        Use feature-based Morgan FP
    
    Returns
    -------
    IPython.display.SVG
    """
    identifier = int(identifier)
    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info, useFeatures=useFeatures)
    if identifier in info.keys():
        atoms, radii = zip(*info[identifier])
        return depict_atoms(mol, atoms, radii, **kwargs)
    else:
        return mol_to_svg(mol, **kwargs)


def plot_class_distribution(df, x_col, y_col, c_col, ratio=0.1, n=1, marker='o', alpha=1, x_label='auto', 
                            y_label='auto', cmap=plt.cm.viridis, size=(8,8), share_axes=False):
    """Scatter + histogram plots of x and y, e.g. after t-SNE dimensionality reduction.
    Colors are wrong in scatter plot if len(class) < 4. Open issue in matplotlib.
    (See: https://github.com/pandas-dev/pandas/issues/9724)
    
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with our data
    {x,y}_col : str
        Name of a column with {x,y} values
    c_col : str
        Name of a column with classes (basis for hue)
    ratio : float
        Ratio to determine empty space of limits of x/y-axis
    marker : str
        Marker in scatter plot
    n : float
        Number of columns of legend
    alpha : float
        Alpha for scatter plot
    x_label : str
        Label of x-axis, default auto: x_col name
    y_label : str
        Label of y-axis, default auto: y_col name
    cmap : matplotlib.colors.ListedColormap
    size : tuple
    
    """
    if y_label is 'auto':
        y_label = y_col
    if x_label is 'auto':
        x_label = x_col    
    
    f, ((h1, xx), (sc, h2)) = plt.subplots(2,2, squeeze=True, sharex=share_axes, sharey=share_axes, figsize=size,
                                           gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})
    f.subplots_adjust(hspace=0.1, wspace=0.1)
    xx.axis('off')

    ratio_xaxis = (max(df[x_col]) - min(df[x_col])) * ratio
    ratio_yaxis = (max(df[y_col]) - min(df[y_col])) * ratio
                  
    x_max = max(df[x_col])+ratio_xaxis
    x_min = min(df[x_col])-ratio_xaxis

    y_max = max(df[y_col])+ratio_yaxis
    y_min = min(df[y_col])-ratio_yaxis

    h1.set_xlim(x_min, x_max)
    h1.xaxis.set_visible(False)
    sc.set_xlim(x_min, x_max)
    sc.set_xlabel(x_label)
    
    h2.set_ylim(y_min, y_max)
    h2.yaxis.set_visible(False)
    sc.set_ylim(y_min, y_max)
    sc.set_ylabel(y_label)
    
    c_unique = np.sort(df[c_col].unique(),)

    h, bins = np.histogram(range(len(cmap.colors)), bins=len(c_unique))  # get equally spaced colors from cmap
    colors = [cmap.colors[int(x)] for x in bins[1:]]
    
    for cl, color in zip(c_unique, colors):
        if len(df[df[c_col] == cl]) > 1:
            sns.kdeplot(df[df[c_col] == cl][x_col], ax=h1, c=color, label=cl, legend=False)  # hist1
            sns.kdeplot(df[df[c_col] == cl][y_col], ax=h2, c=color, vertical=True, label=cl, legend=False)  # hist2
        handles, labels = h1.get_legend_handles_labels()
        h1.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=n)
        sc.scatter(df[df[c_col] == cl][x_col], df[df[c_col] == cl][y_col], c=color, marker=marker, alpha=alpha)
        
    return f


def plot_2D_vectors(vectors, sumup=True, min_max_x=None, min_max_y=None, 
                    cmap=plt.cm.viridis_r, colors=None, vector_labels=None,
                    ax=None):
    """Plots 2d vectors by adding them in sequence and transposing them.
    
    Parameters
    ----------
    vectors : list 
        2D vectors eg: [[0,1], [3,4]]
    sumup : bool
        Show a vector that represents a sum of vectors
    min_max_{x,y} : tuple
        min and max of {x,y} axis
    cmap : plt.cm
        Default: plt.cm.viridis_r
    colors : list
        List of matplotlib colors. Number of colors has to match number of vecors
        (including sum vector if sumup=True). Default=None selects colors from cmap
    vector_labels : list
        Has to match number of vecors (including sum vector if sumup=True)
    ax : plt.ax
        Name of axis to plot to
    Returns
    -------
    plt.figure()
    """
    # Transform the vectors
    soa = []  # vectors with x,y of start point and x,y of end point
    for x in vectors:
        if len(soa) == 0:
            soa.append([0, 0]+list(x))
        else:
            last = soa[-1]
            soa.append([last[0]+last[2]]+[last[1]+last[3]]+list(x))
    if sumup:
        soa.append([0, 0]+list(sum(vectors)))
    X, Y, U, V = zip(*soa)
    if not ax:
        f = plt.figure()
        ax = plt.gca()
    if not colors and sumup:
        colors = [[cmap.colors[120]]*(len(soa)-1)][0] + [cmap.colors[-1]]
    if not colors and not sumup:
        colors = [[cmap.colors[120]]*(len(soa))][0]
    if vector_labels:
        if (len(vector_labels) != len(vectors)) and sumup is False:
            raise Exception('Number of vectors does not match the number of labels')
        if (len(vector_labels) != len(vectors) + 1) and sumup is True:
            raise Exception('Number of vectors does not match the number of labels')
        for x, y, u, v, c, vl in zip(X, Y, U, V, colors, vector_labels):
            Q = ax.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1)
            ax.quiverkey(Q, x, y, u, vl, coordinates='data', color=[0, 0, 0, 0], labelpos='N')
    else:
        ax.quiver(X, Y, U, V, color=colors, angles='xy', scale_units='xy', scale=1)
    # set plot limits based on positions of vectors
    if not min_max_x:
        min_max_x = min([x[0] + x[2] for x in soa]), max([x[0] for x in soa])
    if not min_max_y:
        min_max_y = min([x[1] + x[3] for x in soa]), max([x[1] for x in soa])
    # margins on each side
    margin_x, margin_y = sum(min_max_x)/10., sum(min_max_y)/10.
    ax.set_xlim(min_max_x[0]+margin_x, min_max_x[1]-margin_x)
    ax.set_ylim(min_max_y[0]-margin_y, min_max_y[1]+margin_y)
    return ax


class IdentifierTable(object):
    def _get_depictions(self):
        """Depicts an identifier on the first molecules that contains that identifier"""
        for idx in self.identifiers:
            for mol, sentence in zip(self.mols, self.sentences):
                if idx in sentence:
                    self.depictions.append(depict_identifier(mol, idx, self.radius, molSize=self.size).data)
                    break

    def __init__(self, identifiers, mols, sentences, cols, radius, size=(150, 150)):
        self.mols = mols
        self.sentences = sentences
        self.identifiers = identifiers
        self.cols = cols
        self.radius = radius
        self.depictions = []
        self.size = size
        self._get_depictions()

    def _repr_html_(self):
        table = '<table style="width:100%">'
        c = 1
        for depict, idx in zip(self.depictions, self.identifiers):
            if c == 1:
                table += '<tr>'
            table += '<td><div align="center">%s</div>\n<div align="center">%s</div></td>' % (depict, idx)
            if c == self.cols:
                table += '</tr>'
                c = 0
            c += 1
        table += '</table>'
        return table
