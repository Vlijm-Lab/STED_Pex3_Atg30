import os
import pickle
import math
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from circles.data_handling import folder_structure, init_dataframe, load_updated_pickle, add_empty_row
from circles.misc import unit_to_list, moving_average
from circles.circle_math import outer_circles, inside_outside_line_profile
from progressbar import progressbar


def obtain_statistics(folder, store_path, lw, k_size=20):
    """Obtain statistics from the given images"""

    missing_list = []

    folder_structure(store_path)
    folder_list = os.listdir(folder)

    df = init_dataframe()

    p_profs = []
    a_profs = []

    for num, fname in zip(progressbar(range(len(folder_list))), folder_list):
        path = f'{folder}/{fname}'
        line_profile_path = f'{path}/line_profiles.p'

        if os.path.isfile(line_profile_path):
            line_profile_pex, line_profile_atg = pickle.load(open(line_profile_path, "rb"))
            centers, rads = load_updated_pickle(path)
            rads = unit_to_list(rads)

            circle_list = outer_circles(centers, rads, lw)
            for j in range(len(line_profile_pex)):

                df, idx = add_empty_row(df)
                df.at[idx, "File name"] = fname
                df.at[idx, "Circle number"] = j + 1
                df.at[idx, "#Circles in image"] = len(rads)
                df.at[idx, "#Data points"] = len(line_profile_pex[j])
                df.at[idx, "Circumference (nm)"] = 2 * math.pi * rads[j] * 23

                pex_prof = np.mean(line_profile_pex[j], axis=1)
                pex_ma = moving_average(np.concatenate((pex_prof, pex_prof, pex_prof)), k_size)[
                         len(pex_prof):-len(pex_prof)]
                p_profs.append(pex_prof)

                atg_prof = np.mean(line_profile_atg[j], axis=1)
                atg_ma = moving_average(np.concatenate((atg_prof, atg_prof, atg_prof)), k_size)[
                         len(atg_prof):-len(atg_prof)]
                a_profs.append(atg_prof)

                r, p = stats.pearsonr(pex_prof, atg_prof)
                df.at[idx, "R (raw)"] = r
                df.at[idx, "p (raw)"] = p

                rma, pma = stats.pearsonr(pex_ma, atg_ma)
                df.at[idx, "R (ma)"] = rma
                df.at[idx, "p (ma)"] = pma

                df.at[idx, "mean Pex3"] = np.mean(pex_prof)
                df.at[idx, "mean ATG30"] = np.mean(atg_prof)

                fig, ax = plt.subplots(1)
                ax.plot(pex_prof, c="Magenta", alpha=0.5, label='_nolegend_')
                ax.plot(pex_ma, c="Magenta")
                ax.plot(atg_prof, c="Green", alpha=0.5, label='_nolegend_')
                ax.plot(atg_ma, c="Green")
                ax.legend(['Pex3', 'Atg30'])
                plt.title(f'{fname} Circle {j + 1} (of {len(line_profile_pex)}) - R = {r:.3f} with P = {p:.3f}')
                plt.savefig(f'{store_path}/line_figs/{fname}_circle{j + 1}.png')

                fig, ax1 = plt.subplots(1)
                ax2 = ax1.twinx()
                ax1.plot(pex_prof, c="Magenta", alpha=0.5, label='_nolegend_')
                ax1.plot(pex_ma, c="Magenta")
                ax2.plot(atg_prof, c="Green", alpha=0.5, label='_nolegend_')
                ax2.plot(atg_ma, c="Green")
                ax1.set_ylabel('Pex3', color='Magenta')
                ax2.set_ylabel('Atg30', color='Green')
                ax1.set_xlabel('Circle steps (x 10nm)')
                plt.title(f'{fname} Circle {j + 1} (of {len(line_profile_pex)}) - R = {r:.3f} with P = {p:.3f}')
                plt.savefig(f'{store_path}/line_figs_n/{fname}_circle{j + 1}.png')

                if len(line_profile_pex) > 1:
                    in_lp = inside_outside_line_profile(circle_list, j, len(atg_prof), path=f'{folder}/{fname}')
                    out_lp = list(1 - np.array(in_lp))

                    if np.sum(in_lp) != 0:
                        df.at[idx, "inside mean"] = np.sum(atg_prof * in_lp) / np.sum(in_lp)
                    if np.sum(out_lp) != 0:
                        df.at[idx, "outside mean"] = np.sum(atg_prof * out_lp) / np.sum(out_lp)
                    df.at[idx, "inside percentage"] = np.sum(in_lp) / len(out_lp)
                    df.at[idx, "outside percentage"] = np.sum(out_lp) / len(out_lp)
                else:
                    df.at[idx, "inside mean"] = np.nan
                    df.at[idx, "outside mean"] = np.nan
                    df.at[idx, "inside percentage"] = np.nan
                    df.at[idx, "outside percentage"] = np.nan
        else:
            missing_list.append(fname)
        plt.close('all')

    df.sort_values(['File name', 'mean ATG30'], ascending=[True, False], inplace=True)

    nlist = list(df['File name'])

    cnumber = []
    last_name = None
    cnt = 1
    for name in nlist:
        if name == last_name:
            cnt += 1
        else:
            cnt = 1
        last_name = name
        cnumber.append(cnt)

    df = df.assign(circle_number_sorted=cnumber)

    file_names = df['File name']
    atg_30s = df['mean ATG30']

    mean_atg_normalized = []

    last_name = None
    largest_atg30 = None
    for i, (name, atg30) in enumerate(zip(file_names, atg_30s)):

        if last_name == name:
            mean_atg_normalized.append(atg30 / largest_atg30)

        else:
            last_name = name
            mean_atg_normalized.append(1)
            largest_atg30 = atg30

    df = df.assign(mean_atg_normalized=mean_atg_normalized)

    df.to_excel(f'{store_path}/correlations.xlsx')

    dfm = pd.DataFrame(np.array(missing_list))
    dfm.columns = ["Missing images"]

    dfm.to_excel(f'{store_path}/skipped_images.xlsx')
