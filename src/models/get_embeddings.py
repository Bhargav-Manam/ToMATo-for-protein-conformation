import click

from mds import get_mds_embeddings
import numpy as np


@click.command()
@click.option('-i', '--input-path', type=click.Path(exists=True),
              help="Input path for the rmsd distance matrix, should be an npy file", required=True)
@click.option('-o', '--output-type', type=click.Choice(['npy', 'csv', 'both']), default="npy",
              help="Output type: npy, csv, both")
def main(input_path, output_type):
    """
    CLI function to generate embeddings for a given rmsd distance matrix and save to a file in either .npy,
    .csv or both formats.

    Usage:
        python3 get_embeddings,py -i <input_path> -o <output_type>

    Returns:
        None
    """

    distances = np.load(input_path)
    embeddings = get_mds_embeddings(distances)

    name = "embeddings_" + str(embeddings.shape[0]) + "prots"
    path = "../../data/processed/"

    if output_type != 'csv':
        np.save(path+name + ".npy", embeddings)

    if output_type != 'npy':
        np.savetxt(path+name+".csv", embeddings, delimiter=',')


if __name__ == '__main__':
    main()

    # n_aladips = 2
    # aladips = get_n_proteins(n_aladips)
    # distances = get_rmsd_matrix(aladips)
    # np.savetxt("foo.csv", distances)

    # mds = MDS(n_components=2, dissimilarity='precomputed', normalized_stress='auto')
    # embedding = mds.fit_transform(distances)
    # # Create a scatter plot of the embedding
    # plt.scatter(embedding[:, 0], embedding[:, 1])
    #
    # # Add labels to the plot
    # plt.title('MDS Embedding')
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    #
    # # Show the plot
    # plt.show()
