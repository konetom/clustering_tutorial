import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from shutil import rmtree
from os.path import join as opj
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.cluster import AgglomerativeClustering

# configurations
st.set_page_config(layout="wide")
warnings.filterwarnings("ignore")


@st.cache
def load_data(outp, file):
    if outp not in os.listdir():
        os.mkdir(outp)
    else:
        rmtree(outp)
        os.mkdir(outp)
    data = pd.read_excel(file, index_col=0)
    cols = [i.split(' ')[0] if "Anis" not in i else "A. "+i.split(' ')[1] for i in data.columns.values.tolist()]
    return data, cols


@st.cache
def main(outp, data, cols, metrics, methods):
    # metrics = distance.__all__
    # methods = hierarchy.__all__
    train_data = data.T.values
    enlarge = 4
    for cluster_n in range(1, len(cols)+1):
        if len(methods) > 1 and len(metrics) > 1:
            fig, ax = plt.subplots(nrows=len(metrics), ncols=len(methods), figsize=(len(methods)*enlarge, len(metrics)*enlarge), sharex=True, sharey=True, squeeze=True)
            for ic, metric in enumerate(metrics):
                for od, method in enumerate(methods):
                    try:
                        model = AgglomerativeClustering(n_clusters=cluster_n, linkage=method, affinity=metric)
                        result = model.fit_predict(train_data)
                        clusters = np.unique(result)
                        for cluster in clusters:
                            index = np.where(result == cluster)
                            ax[ic, od].scatter(train_data[index, 1], train_data[index, 0], s=400)
                            ax[-1, od].set_xlabel(method, fontsize=16)
                            ax[ic, 0].set_ylabel(metric, fontsize=16)
                            for e, label in enumerate(cols):
                                ax[ic, od].annotate(label, (train_data[e, 1], train_data[e, 0]), fontsize=10)
                    except ValueError:
                        pass
            fig.suptitle(f"N clusters = {cluster_n}", color=f"C{cluster_n-1}", fontsize=18)
        elif len(methods) > 1 and len(metrics) == 1:
            fig, ax = plt.subplots(nrows=1, ncols=len(methods), figsize=(len(methods)*enlarge, len(metrics)*enlarge), sharex=True, sharey=True, squeeze=True)
            for ic, metric in enumerate(metrics):
                for od, method in enumerate(methods):
                    try:
                        model = AgglomerativeClustering(n_clusters=cluster_n, linkage=method, affinity=metric)
                        result = model.fit_predict(train_data)
                        clusters = np.unique(result)
                        for cluster in clusters:
                            index = np.where(result == cluster)
                            ax[od].scatter(train_data[index, 1], train_data[index, 0], s=400)
                            ax[od].set_xlabel(method, fontsize=16)
                            ax[0].set_ylabel(metric, fontsize=16)
                            for e, label in enumerate(cols):
                                ax[od].annotate(label, (train_data[e, 1], train_data[e, 0]), fontsize=10)
                    except ValueError:
                        pass
            fig.suptitle(f"N clusters = {cluster_n}", color=f"C{cluster_n-1}", fontsize=18)
        elif len(methods) == 1 and len(metrics) > 1:
            fig, ax = plt.subplots(nrows=1, ncols=len(metrics), figsize=(len(metrics)*enlarge, len(methods)*enlarge), sharex=True, sharey=True, squeeze=True)
            for ic, metric in enumerate(metrics):
                for od, method in enumerate(methods):
                    try:
                        model = AgglomerativeClustering(n_clusters=cluster_n, linkage=method, affinity=metric)
                        result = model.fit_predict(train_data)
                        clusters = np.unique(result)
                        for cluster in clusters:
                            index = np.where(result == cluster)
                            ax[ic].scatter(train_data[index, 1], train_data[index, 0], s=400)
                            ax[ic].set_xlabel(metric, fontsize=16)
                            ax[0].set_ylabel(method, fontsize=16)
                            for e, label in enumerate(cols):
                                ax[ic].annotate(label, (train_data[e, 1], train_data[e, 0]), fontsize=10)
                    except ValueError:
                        pass
            fig.suptitle(f"N clusters = {cluster_n}", color=f"C{cluster_n-1}", fontsize=18)
        else:
            try:
                model = AgglomerativeClustering(n_clusters=cluster_n, linkage=methods[0], affinity=metrics[0])
                result = model.fit_predict(train_data)
                clusters = np.unique(result)
                plt.figure(figsize=(4, 4))
                for cluster in clusters:
                    index = np.where(result == cluster)
                    plt.scatter(train_data[index, 1], train_data[index, 0], s=300)
                    plt.xlabel(methods[0], fontsize=10)
                    plt.ylabel(metrics[0], fontsize=10)
                    for e, label in enumerate(cols):
                        plt.annotate(label, (train_data[e, 1], train_data[e, 0]), fontsize=8)
            except ValueError:
                pass
            plt.title(f"N clusters = {cluster_n}", color=f"C{cluster_n-1}", fontsize=12)
        plt.tight_layout()
        plt.savefig(opj(outp, f"{cluster_n}.png"), dpi=100)
    return len(cols)


st.header('How does the clustering work?')
st.subheader('...by hierarchical agglomeration...')
file = st.file_uploader("Upload data", type=["xlsx"], accept_multiple_files=False)
if file is not None:
    output_folder = "data"
    data, cols = load_data(output_folder, file)
    left, right = st.columns(2)
    metrics = [i for i in left.multiselect("Select metric(s):", ["canberra", "chebyshev", "correlation", "euclidean"])]
    methods = [i for i in left.multiselect("Select method(s):", ["average", "complete", "single", "ward"])]
    if len(metrics) == 0 or len(methods) == 0:
        left.write("Please select at least one distance metric and clustering method")
    elif len(metrics) > 1 and len(methods) > 1:
        cluster_n = left.slider('Choose the number of clusters:', 1, len(cols))
        if left.button("Generate the plot"):
            main(output_folder, data, cols, metrics, methods)
        if len(os.listdir(output_folder)) == len(cols):
            right.image(opj(output_folder, f"{cluster_n}.png"))
    else:
        cluster_n = st.slider('Choose the number of clusters:', 1, len(cols))
        if st.button("Generate the plot"):
            main(output_folder, data, cols, metrics, methods)
        if len(os.listdir(output_folder)) == len(cols):
            st.image(opj(output_folder, f"{cluster_n}.png"))
