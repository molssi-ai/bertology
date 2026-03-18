###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: August 2025                                                           #
# Description:                                                                #
# This module will use the 'std_effect.xls' raw data file to plot the         #
# standardization effect on model performance.                                #
###############################################################################

# import the necessary modules
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def custom_aggfunc(x):
    # critical values from NIST STAT Handbook
    # https://www.itl.nist.gov/div898/handbook/eda/section3/eda3672.htm
    if len(x) == 1:
        ci = np.nan
    elif len(x) == 2:
        ci = 12.706 * x.std() / np.sqrt(len(x))
    elif len(x) == 3:
        ci = 4.303 * x.std() / np.sqrt(len(x))
    elif len(x) == 4:
        ci = 3.182 * x.std() / np.sqrt(len(x))
    elif len(x) == 5:
        ci = 2.776 * x.std() / np.sqrt(len(x))
    return f"{x.mean():.2f}\n$\\pm$\n{ci:.2f}"


def main():
    # input and output file paths
    input_file = "./std_effect.xls"

    # load the dataset
    ds = pd.read_excel(input_file)

    # remove the diverged trainings with high perplexity (outliers)
    ds = ds[ds["perplexity"] < 10.0]

    # create heatmaps for t-loss as score and bin_idx and corrupt as x and y axes
    # set the figure size to 3.5 x 3.5 inches in dpi=150
    model_dict = {"tiny": "Tiny", "small": "Small", "base": "Base"}
    metric_dict = {
        "t-loss": "T-Loss",
        "v-loss": "V-Loss",
        "accuracy": "V-Acc",
        "w-f1": "V-wF1",
        "perplexity": "V-PPPL",
    }

    for model in model_dict.keys():
        for metric in metric_dict.keys():
            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
            annot = ds[ds["Name"] == model].pivot_table(
                index="corrupt",
                columns="bin_idx",
                values=metric,
                aggfunc=custom_aggfunc,
            )
            sns.heatmap(
                data=ds[ds["Name"] == model].pivot_table(
                    index="corrupt",
                    columns="bin_idx",
                    values=metric,
                    aggfunc="mean",
                ),
                ax=ax,
                cmap="viridis",
                annot=annot,
                fmt="",
            )
            ax.set_title(f"{model_dict[model]}\n({metric_dict[metric]})")
            ax.set_xlabel("$ \\tau $")
            # invert the y axis
            ax.invert_yaxis()
            ax.set_ylabel("$ \\nu $")

            # set the format of the labels in the color bar to 2 decimal places
            # make sure all models have the same color bar range for the same metric
            cbar = ax.collections[0].colorbar
            cbar.ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, _: f"{x:.2f}")
            )

            # adjust the layout
            fig.tight_layout()

            # save the figure
            fig.savefig(f"./{model}_{metric}_heatmap.pdf", bbox_inches="tight", dpi=150)


if __name__ == "__main__":
    main()
