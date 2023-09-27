import click

from rmsd_qcp import get_rmsd_matrix
from parse_proteins import get_n_proteins
import numpy as np


@click.command()
@click.option('-n', '--number-of-proteins', type=click.INT,
              help="Number of proteins to get the RMSD distance matrix for", required=True)
@click.option('-o', '--output-type', type=click.Choice(['npy', 'csv', 'both']), default="npy",
              help="Output type: npy, csv, both")
def main(number_of_proteins, output_type):
    """
        CLI function to generate an RMSD distance matrix for a given number of proteins and output it in either an npy or csv file format.

        Usage:
            python3 build_features.py -n 10 -o npy

        Returns:
            None
    """

    proteins = get_n_proteins(number_of_proteins)
    distances = get_rmsd_matrix(proteins)

    name = "distances_"+str(number_of_proteins)+"prots"
    path = "../../data/processed/"

    if output_type != 'csv':
        np.save(path+name+".npy", distances)
    if output_type != 'npy':
        np.savetxt(path+name+".csv", distances, delimiter=',')


if __name__ == '__main__':
    main()
