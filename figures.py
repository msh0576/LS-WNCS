from typing import final
from matplotlib import pyplot as plt
import numpy as np
import torch
import pandas as pd
import os
import seaborn as sns
import re
import csv
import ast
import json
from scipy.spatial import distance
from sklearn.datasets import fetch_openml
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1 import make_axes_locatable


def min_key_len(dict_, key_):
    '''
    Input:
        dict_: list of dictionary
        key_: string, key name to find the minimum length
    '''
    min_len = 100000000000000
    for each_dict in dict_:
        if len(each_dict[key_]) < min_len:
            min_len = len(each_dict[key_])
    return min_len

def incorp_key(dict_, key_, min_len):
    '''
    Input:
        dict_:
        key_: specific key to incoporate
    Output:
        matrix: np, (B(dict_ size), min_len)
    '''
    B = len(dict_)
    matrix = np.empty([B, min_len], dtype=np.float)
    for idx, each_dict in enumerate(dict_):
        matrix[idx] = each_dict[key_][:min_len]
    return matrix

FONT_SIZE = 5
FIGSIZE = (5.2,4.2)
FIGSIZE2 = (4.2,3.2)
LEGEND_SIZE = 12
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 13
BLUE_SERIES = ['#0000ff', '#0066ff', '#3399ff', '#66ccff']
BLUE_SERIES2 = ['#0d2afd', '#0b5fe3', '#00a4fa', '#0bd0e3', '#0dfdcf']
RED_SERIES = ['#cc0000', '#ff5050', '#ff9999' , '#ffcccc']
GREEN_SERIES = ['#006600', '#00cc00', '#00ff00', '#66ff99']
LINE_WIDTH = 3

def set_figure_size(ax, xticks=None, yticks=None, title='', xlabel='', ylabel=''):
    # === title, label size and bole ===
    ax.set_title(title, size=BIGGER_SIZE, weight='bold')
    ax.set_xlabel(xlabel, size=BIGGER_SIZE, weight='bold')
    ax.set_ylabel(ylabel, size=BIGGER_SIZE, weight='bold')

    # === x,y ticks size and weight ===
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(MEDIUM_SIZE)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(MEDIUM_SIZE)
        tick.label1.set_fontweight('bold')

    # === x and y ticks ===
    if xticks != None:
        ax.xaxis.set_ticks(xticks)
        ax.set_xlim(min(xticks), max(xticks))
    if yticks != None:
        ax.yaxis.set_ticks(yticks)
        ax.set_ylim(min(yticks), max(yticks))


def plot_figure_setting():
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def show_learning_rewards(common_paths, key, labels, 
                        title='', xlabel='', ylabel='',
                        xtickrange=[], ytickrange=[], fill_std=False, colors=None,
                        no_show=False, save_path=None):
    '''
    '''
    fig, ax = plt.subplots(figsize=FIGSIZE2)
    

    results = [pd.read_csv(os.path.join(path, 'log.csv')) for path in common_paths]
    c_idx = 0
    for result, label in zip(results, labels):
        if fill_std:
            if result.get('std_return') is not None:
                std_return = result['std_return']
            else:
                std_return = 0
            mean_return = result['return']
            if colors is not None:
                ax.plot(result['step'], mean_return, label=label, color=colors[c_idx], linewidth=LINE_WIDTH)
                ax.fill_between(result['step'], mean_return-std_return, mean_return+std_return, alpha=0.3, color=colors[c_idx])
                c_idx += 1
            else:
                ax.plot(result['step'], mean_return, label=label, linewidth=LINE_WIDTH)
                ax.fill_between(result['step'], mean_return-std_return, mean_return+std_return, alpha=0.3)
        else:
            ax.plot(result['step'], result[key], label=label)
        print(f"{label}: final mean_return:{mean_return.iloc[-1]} | std_return:{std_return.iloc[-1]}")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    if not xtickrange  and not ytickrange:
        set_figure_size(ax, xlabel=xlabel, ylabel=ylabel, title=title)
    elif xtickrange and ytickrange:
        set_figure_size(ax, xlabel=xlabel, ylabel=ylabel, title=title, xticks=xtickrange, yticks=ytickrange)
    elif xtickrange:
        set_figure_size(ax, xlabel=xlabel, ylabel=ylabel, title=title, xticks=xtickrange)
    elif ytickrange:
        set_figure_size(ax, xlabel=xlabel, ylabel=ylabel, title=title, yticks=ytickrange)

    
    if no_show:
        return fig, ax
    else:
        plt.legend(fontsize=LEGEND_SIZE)
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
        plt.show()

def show_schedule(common_paths, key, labels, 
    title='', xlabel='', ylabel='',
    xtickrange=[], ytickrange=[], save_path=None):
    results = [pd.read_csv(os.path.join(path, 'log.csv')) for path in common_paths]
    final_dt_ratios = [result[key].iloc[-1] for result in results]  # element is string
    # print(f"final_dt_ratios:\n {final_dt_ratios}")
    
    # ===== generate new dataset: integrate all schedule data =====
    dt_ratio_dict = {}
    min_length = 10000000
    max_length = 0
    for idx, final_dt_ratio in enumerate(final_dt_ratios):
        dt_ratio_dict[labels[idx]] = [int(string) for string in re.findall(r'\d+', final_dt_ratio)]
        if len(dt_ratio_dict[labels[idx]]) < min_length:
            min_length = len(dt_ratio_dict[labels[idx]])
        if len(dt_ratio_dict[labels[idx]]) > max_length:
            max_length = len(dt_ratio_dict[labels[idx]])
    # print(f"max_length:{max_length}")
    
    # == align with min_length ==
    # for key, val in dt_ratio_dict.items():
    #     dt_ratio_dict[key] = val[:min_length]
    # == align with max_length ==
    for key, val in dt_ratio_dict.items():
        padd_size = max_length - len(val)
        dt_ratio_dict[key] = val + [0] * padd_size
    # == change value to ratio ==
    for key, val in dt_ratio_dict.items():
        total = sum(dt_ratio_dict[key])
        dt_ratio_dict[key] = [ele/total for ele in dt_ratio_dict[key]]


    data = pd.DataFrame(dt_ratio_dict)
    data = data.transpose()
    # print(f"data:\n{data}")
    
    # ==== hitmap xticks setting ====
    num_ticks = 10
    depth_list = np.arange(max_length)
    # the index of the position of yticks
    xticks = np.linspace(0, len(depth_list) - 1, num_ticks, dtype=np.int)
    print(f"xticks;{xticks}")
    # the content of labels of these yticks
    xticklabels = [depth_list[idx] for idx in xticks]
    print(f"xticklabels:{xticklabels}")

    # ===== figure hitmap =====
    fig = plt.figure(figsize=(5,4))
    ax = sns.heatmap(data, cmap='Blues', xticklabels=5) # cmap='YlGnBu' , 'YlOrRd'
    
    
    # plt.title('Schedule ratio', fontsize=20)
    plt.xlabel('Sampling periods')
    if save_path is not None:
        # plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
    plt.show()

def show_traj(path, key, label, title='', xlabel='', ylabel='',
    xtickrange=[], ytickrange=[], save_path=None, show_latent=False):
    state_traj = np.loadtxt(os.path.join(path, 'state_log.csv'), delimiter=',') # [L, D_state]
    dt_traj = np.loadtxt(os.path.join(path, 'dt_log.csv'), delimiter=',')       # [L, ]
    action_traj = np.loadtxt(os.path.join(path, 'action_log.csv'), delimiter=',')

    L = state_traj.shape[0]
    half = round(L/2)
    state_0 = state_traj[:half, 0]  # cart position
    state_1 = state_traj[:half, 1]  # 
    state_2 = state_traj[:half, 2]
    state_3 = state_traj[:half, 3]
    state_4 = state_traj[:half, 4]
    sched_time = []
    for idx, val in enumerate(dt_traj):
        sched_time.append(np.sum(dt_traj[:idx]))
    sched_time = np.stack(sched_time)[:half]
    # print(f"sched_time:{sched_time}")
    # print(f"shape: {state_traj.shape} | {dt_traj.shape} | {action_traj.shape} | {sched_time.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(8,4))
    axes[0].plot(sched_time, state_0)
    axes[1].plot(sched_time, state_1)
    
    # === verticle line ===
    for time in sched_time:
        axes[0].axvline(x=time, color='k', linestyle='--')
        axes[1].axvline(x=time, color='k', linestyle='--')
    
    # === ylabel ===
    axes[0].set_ylabel('position')
    axes[1].set_ylabel('angle')
    axes[1].set_xlabel('step')
    
    # if save_path is not None:
    #     # plt.axis('off')
    #     plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
    # else:
    #     plt.show()
    
    # === show latent action ===
    if show_latent:
        latent_action_traj = np.loadtxt(os.path.join(path, 'latent_action_log.csv'), delimiter=',')
        L = latent_action_traj.shape[0]
        half = round(L/2)
        
        caction = action_traj[:half]
        dt = dt_traj[:half]
        
        latent_0 = latent_action_traj[:half, 0]
        latent_1 = latent_action_traj[:half, 1]
        latent_2 = latent_action_traj[:half, 2]
        latent_3 = latent_action_traj[:half, 3]
        latent_4 = latent_action_traj[:half, 4]
        
        fig2, axes2 = plt.subplots(7, 1, figsize=(10,6))
        axes2[0].plot(sched_time, latent_0)
        axes2[1].plot(sched_time, latent_1)
        axes2[2].plot(sched_time, latent_2)
        axes2[3].plot(sched_time, latent_3)
        axes2[4].plot(sched_time, latent_4)
        axes2[5].plot(sched_time, caction)
        axes2[6].plot(sched_time, dt)
        
        
        # === ylabel ===
        axes2[0].set_ylabel('z0')
        axes2[1].set_ylabel('z1')
        axes2[2].set_ylabel('z2')
        axes2[3].set_ylabel('z3')
        axes2[4].set_ylabel('z4')
        axes2[5].set_ylabel('caction')
        axes2[6].set_ylabel('daction')
        axes2[6].set_xlabel('step')
        
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    else:
        plt.show()
                

def show_motivation(path):
    result = pd.read_csv(os.path.join(path, 'log.csv'))
    returns_per_p = {}

    # === align returns along with each period ===
    dict_list = []
    # string to dictionary
    for dict_ in result['motivation_return'].iloc[:]:
        dict_list.append(ast.literal_eval(dict_))

        for key, val in dict_list[0].items():
            returns_per_p[key] = []

        for returns in dict_list:
            for key, val in returns.items():
                returns_per_p[key].append(val)
    
    # === show figure ===
    mean_size = 100
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, returns in returns_per_p.items():
        ax.plot(result['step'], returns, label=f'period {key}')
        mean_return = sum(returns[-mean_size:])/mean_size
        print(f"period {key} mean_return:{mean_return}")
    print(f"==============")
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    set_figure_size(ax, xlabel='Steps', ylabel='Cumulative epiosde rewards', title='')

    plt.legend()
    # plt.show()

def print_cont_cunt(paths, labels):
    results = [pd.read_csv(os.path.join(path, 'log.csv')) for path in paths]
    
    for idx, result in enumerate(results):
        final_cont_cunt = result['cont_cunt'].iloc[-1]
        print(f'"{labels[idx]}" - control count:{final_cont_cunt}')
    
def add_figure(fig, ax, x_, y_, label, line='k--',
        save_path=None,
        legend_loc=None,
    ):
    ax.plot(y_, x_, line, label=label)
    if legend_loc is not None:
        plt.legend(fontsize=LEGEND_SIZE, loc='center right', bbox_to_anchor=legend_loc)
    else:
        plt.legend(fontsize=LEGEND_SIZE)
        
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', pad_inches = 0.02)
    plt.show()

def show_improve_percentage(algo_dict, colors, save_path=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    col_idx = 0
    for key, val in algo_dict.items():
        ax.scatter(val[0], val[1], c=colors[col_idx], s=200, label=key)
        col_idx += 1
    set_figure_size(ax, xlabel='Improved control performance (%)', ylabel='Saved resource consumption (%)')
    plt.legend()
    if save_path is not None:
        # plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0)
    plt.show()

def show_state_exploration_v3(paths, envsteps, labels, episode=10, save_path=None):
    colors = ['#0b5fe3', 'orange', 'green', 'skyblue', 'red', 'purple', 'yellow']
    envsteps_len = len(envsteps)
    paths_len = len(paths)
    fig, ax = plt.subplots(paths_len, envsteps_len, figsize=(7.2, 6.2))
    fig.tight_layout(h_pad=2)
    
    for path_idx, path in enumerate(paths):
        for envstep_idx, envstep in enumerate(envsteps):
            episode_hybrid_action_traj = []
            
            for epi in range(episode):
                episode_hybrid_action_traj.append(np.loadtxt(os.path.join(path, f'envstep-{envstep}_episode-{epi}_hybrid_action_log.csv'), delimiter=','))
        
        
            # === distance for all episode hybrid actions ===
            # episode_hybrid_action_traj = np.array(episode_hybrid_action_traj)
            episode_hybrid_action_traj = np.concatenate(episode_hybrid_action_traj, 0)
            print(f"episode_hybrid_action_traj:{episode_hybrid_action_traj.shape}")
            caction = episode_hybrid_action_traj[:, 0]
            daction = episode_hybrid_action_traj[:, 1]
            ax[path_idx, envstep_idx].scatter(caction, daction, alpha=0.2, label=labels[path_idx], s=10, color=colors[path_idx])
            if path_idx == 0:
                set_figure_size(ax[path_idx, envstep_idx], title=f'train_{envstep}', xticks=[-1., -0.5, 0, 0.5, 1.], yticks=[-1., -0.5, 0, 0.5, 1.])
            else:
                set_figure_size(ax[path_idx, envstep_idx], xticks=[-1., -0.5, 0, 0.5, 1.], yticks=[-1., -0.5, 0, 0.5, 1.])
        ax[path_idx, envstep_idx].legend(fontsize=LEGEND_SIZE)
    # set_figure_size(ax[path_idx, envstep_idx], title=f'train_{envstep}', xlabel='Caction space', ylabel='Daction space')
    ax[paths_len-1, 0].set_xlabel('Continuous action space')
    ax[paths_len-1, 0].set_ylabel('Discrete action space')
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    plt.show()



def show_t_SNE(paths, envsteps, labels, episode=10, save_path=None):
    colors = ['blud', 'orange', 'green', 'skyblue', 'red', 'purple', 'yellow']
    envsteps_len = len(envsteps)
    paths_len = len(paths)
    fig, ax = plt.subplots(1, envsteps_len, figsize=(5.2, 5.2))
    fig.tight_layout(h_pad=2)
    
    for path_idx, path in enumerate(paths):
        for envstep_idx, envstep in enumerate(envsteps):
            episode_hybrid_action_traj = []
            episode_latent_action_traj = []
            episode_dt = []
            
            for epi in range(episode):
                episode_hybrid_action_traj.append(np.loadtxt(os.path.join(path, f'envstep-{envstep}_episode-{epi}_hybrid_action_log.csv'), delimiter=','))
                episode_latent_action_traj.append(np.loadtxt(os.path.join(path, f'envstep-{envstep}_episode-{epi}_latent_action_log.csv'), delimiter=','))
                episode_dt.append(np.loadtxt(os.path.join(path, f'envstep-{envstep}_episode-{epi}_dt_log.csv'), delimiter=','))
        
            # === distance for all episode hybrid actions ===
            episode_hybrid_action_traj = np.concatenate(episode_hybrid_action_traj, 0)
            episode_latent_action_traj = np.concatenate(episode_latent_action_traj, 0)
            episode_dt = np.concatenate(episode_dt, 0)
            print(f"episode_hybrid_action_traj:{episode_hybrid_action_traj.shape}")
            print(f"episode_latent_action_traj:{episode_latent_action_traj.shape}")
            print(f"episode_dt:{episode_dt.shape}")
            caction = episode_hybrid_action_traj[:, 0]
            daction = episode_hybrid_action_traj[:, 1]
            
            # === t-SNE ===
            tsne_np = TSNE(n_components=2).fit_transform(episode_latent_action_traj)
            tsnelabel_np = TSNE(n_components=1).fit_transform(episode_hybrid_action_traj)
            tsne_latent_df = pd.DataFrame(tsne_np, columns=['component 0', 'component 1'])
            tsnelabel_df = pd.DataFrame(tsnelabel_np, columns=['label3'])
            tsne_dt_df = pd.DataFrame(episode_dt, columns=['label'])
            tsne_caction_df = pd.DataFrame(caction, columns=['label2'])
            tsne_df = pd.concat([tsne_latent_df, tsne_dt_df, tsne_caction_df, tsnelabel_df], axis=1)
            
            tmp_ax = ax if envsteps_len == 1 else ax[envstep_idx]
            ax_ = tmp_ax.scatter(tsne_np[:, 0], tsne_np[:, 1], c=tsne_df['label3'])
            
            set_figure_size(tmp_ax)
            tmp_ax.tick_params(labelsize=15)
            
            cbar = fig.colorbar(ax_)
            cbar.ax.tick_params(labelsize=15)
    
    # print(f"tsne_df:{tsne_df}")
            
    plt.legend(fontsize=LEGEND_SIZE)
    # plt.colorbar()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    plt.show()


def show_vae_loss(common_paths, labels,
                save_path=None):
    fig, ax = plt.subplots(2, 1, figsize=FIGSIZE)
    
    
    results = [pd.read_csv(os.path.join(path, 'model', 'vae_logs.csv')) for path in common_paths]
    
    for result, label in zip(results, labels):
        length = len(result['dyn_loss'])
        interval = 1000
        indexes = list(range(0, length, interval))
        # === dyn_loss ===
        ax[0].plot(result['dyn_loss'].iloc[indexes], label=label, linewidth=LINE_WIDTH)
        # ax[0].set_ylabel('dynamic loss')
        set_figure_size(ax[0], ylabel='Dynamic loss')
        
        # === reconstruction_loss ===
        ax[1].plot(result['recon_loss'].iloc[indexes], label=label, linewidth=LINE_WIDTH)
        set_figure_size(ax[1], xlabel='Train iteration', ylabel='Reconstruction \nloss')
        

    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    plt.show()
    
    
def show_energy_and_control(paths, labels, xticks, title='', save_path=None):
    fig, ax1 = plt.subplots(figsize=FIGSIZE2)
    color1 = BLUE_SERIES[0]
    color2 = RED_SERIES[0]
    
    results = [pd.read_csv(os.path.join(path, 'log.csv')) for path in paths]
    
    result_dict = {
        'cont_cunt':[],
        'cont_perf':[],
    }
    for idx, result in enumerate(results):
        final_cont_cunt = result['cont_cunt'].iloc[-1]
        result_dict['cont_cunt'].append(final_cont_cunt)
        result_dict['cont_perf'].append(result['return'].iloc[-1])
    assert len(result_dict['cont_cunt']) == len(xticks)
    
    print(f"energy loss coefficient:{xticks}")
    print(f"control performance:{result_dict['cont_perf']}")
    print(f"energy consumtion (control conunt): {result_dict['cont_cunt']}")
    
    
    ax1.plot(xticks, result_dict['cont_cunt'], linewidth=LINE_WIDTH, color=color1)
    ax1.set_xlabel('Coefficient of energy loss function', size=BIGGER_SIZE, weight='bold')
    ax1.set_ylabel('Average number of \n control count \n (energy consumption)', size=BIGGER_SIZE, weight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.xaxis.set_ticks(xticks)
    ax1.xaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    ax1.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    
    # ax1.set_xlim(min(xticks), max(xticks))
    
    ax2 = ax1.twinx()    
    ax2.plot(xticks, result_dict['cont_perf'], linewidth=LINE_WIDTH, color=color2)
    ax2.set_ylabel('Average control performance', size=BIGGER_SIZE, weight='bold')
    ax2.yaxis.set_tick_params(labelsize=MEDIUM_SIZE)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.yticks()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight',pad_inches = 0.02)
    plt.show()


if __name__ == '__main__':
    img_save_path = 'results/Figures/Co-learning'
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    test_1 = 'logs/cartpole-swingup/20230616_095940_plas_latent-LHEC_test-seed4'
    test_2 = 'logs/cartpole-swingup/20230616_224154_plas_latent-LHEC_test_loss01-seed2'
    


    multi_plas_plant1_path1 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi-seed1' # multi_policy O | eps->0.2 | transition 수정 | dt_reward X |
    multi_plas_plant1_path2 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi_P50-seed1' # multi_policy O | eps->0.2 | transition 수정 | maximum period 30->50 | dt_reward X |
    multi_plas_plant1_path3 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi_P50-seed2' # multi_policy O | eps->0.2 | transition 수정 | maximum period 30->50 | dt_reward O |
    multi_plas_plant1_path4 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi-seed2' # multi_policy O | eps->0.2 | transition 수정 | maximum period 30 |  dt_reward O |
    
    multi_plas_plant2_path2 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant2_multipi-seed1' # multi_policy O | eps->0.2 | transition 수정 |
    multi_plas_plant4_path3 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant4_multipi-seed1' # multiagent O | eps->0.2 | transition memorize 방법 수정
    multi_plas_plant8_path2 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant8_multipi-seed1' # multipolicy O | eps->0.2 | transition memorize 방법 수정
    
    hsac_path2 = 'logs/cartpole-swingup/hsac-slotSched-seed1'
    mpdqn_path2 = 'logs/cartpole-swingup/mpdqn_nstep-slotSched-seed1'
    multi_plas_plant1_exploration_path11 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi_exploration-seed11' # vae training 후 | latent_dim 2 | decoder exploration O
    multi_plas_plant1_exploration_path13 = 'logs/cartpole-swingup/plas_latent-multi_slotSched_plant1_multipi_exploration-seed13' # latent_dim 30 | action_relable O | decoder exploration O
    sac_hybrid_exploration_path3 = 'logs/cartpole-swingup/sac-slotSched_hybridaction_exploration-seed3' # sac에 hybrid action 적용 | measure at policy training 0 and 10000 | 100 evaluation episodes | 
    
    # === energy consumption ===
    lhec_energyloss00 = 'logs/cartpole-swingup/20230206_082138_plas_latent-LHEC_LatentDim30_LossEnergy_energycoeff00-seed4' 
    lhec_energyloss05 = 'logs/cartpole-swingup/20230206_082137_plas_latent-LHEC_LatentDim30_LossEnergy_energycoeff05-seed4' 
    lhec_energyloss10 = 'logs/cartpole-swingup/20230206_082136_plas_latent-LHEC_LatentDim30_LossEnergy_energycoeff10-seed4' 
    
    
    
    print(f'=========== learning performance ==========')
    fig, ax = show_learning_rewards(
        [
            multi_plas_plant1_path4, 
            hsac_path2, 
            mpdqn_path2], key='return',
        labels=['LS-WNCS', 'Hybrid-SAC', 'MP-DQN'], 
        ylabel='Cumulative episode rewards', xlabel='Episodes',
        fill_std=True,
        no_show=True,
        # xtickrange = [100000, 250000]
    )
    add_figure(fig, ax, x_=[838]*3000, y_=range(3000, 6000), 
        label='SAC \n(optimal period 8)', line='k--',
        save_path=os.path.join(img_save_path, 'baseline'),
        legend_loc=(1., 0.65)
    )
    # ===== compare baseline RL algorithms =====
    show_schedule([
        multi_plas_plant1_path4, 
        hsac_path2, 
        mpdqn_path2], 
        key='dt_ratio', 
        labels=['LS-WNCS', 'Hybrid-SAC', 'MP-DQN'],
        save_path=os.path.join(img_save_path, 'schedule_ratio_wrt_baselines')
    )
    # ===== quantitative results =====
    
    
    print(f'============= CVAE effects =============')
    # ===== selected action distribution wrt latent space dimension =====
    show_state_exploration_v3(
        [multi_plas_plant1_exploration_path13, multi_plas_plant1_exploration_path11, sac_hybrid_exploration_path3],
        envsteps=[0, 100000],
        episode=100,
        labels=['with CVAE (dim 30)', 'with CVAE (dim 2)', 'without CVAE'],
        save_path=os.path.join(img_save_path, 'action_distribution')
    )
    
    show_learning_rewards(
        [
            multi_plas_plant1_exploration_path13, 
            multi_plas_plant1_exploration_path11, 
            sac_hybrid_exploration_path3], key='return',
        labels=['with CVAE (dim 30)', 'with CVAE (dim 2)', 'without CVAE'], 
        ylabel='Cumulative episode rewards', xlabel='Episodes',
        fill_std=True,
        save_path=os.path.join(img_save_path, 'performance_wrt_VAE')
    )
    
    
      
    # ===== energy loss coefficiency =====
    # === energy consumption
    show_energy_and_control(
        paths = [lhec_energyloss00, lhec_energyloss05, lhec_energyloss10], 
        labels= ['energy', 'control'],
        xticks=[0., 0.5, 1.0],
        save_path=os.path.join(img_save_path, 'EnergyConsumptionCoefficient')
    )
    # === energy save
    
    print(f'========== discounted reward update method ===========')
    show_learning_rewards(
        [multi_plas_plant1_path4, multi_plas_plant1_path3, multi_plas_plant1_path1, multi_plas_plant1_path2], key='return',
        labels=['P30 w/ discR', 'P50 w/ discR', 'P30 w/o discR', 'P50 w/o discR'], 
        ylabel='Cumulative episode rewards', xlabel='Episodes',
        fill_std=True,
        colors=BLUE_SERIES[:2] + RED_SERIES[:2],
        save_path=os.path.join(img_save_path, 'discountedReward')
        
        # xtickrange = [100000, 250000]
    )
    
    show_schedule([multi_plas_plant1_path4, multi_plas_plant1_path3, multi_plas_plant1_path1, multi_plas_plant1_path2], 
        key='dt_ratio', 
        labels=['P30 w/ discR', 'P50 w/ discR', 'P30 w/o discR', 'P50 w/o discR'],
        save_path=os.path.join(img_save_path, 'schedule_ratio_wrt_discounted_reward_update')
    )
    
    
    print(f'========= multi-loops =========')
    show_learning_rewards(
        [multi_plas_plant1_path4, multi_plas_plant2_path2, multi_plas_plant4_path3, multi_plas_plant8_path2,
        #  multi_plas_plant16_path1, multi_plas_plant30_path1
         ], key='return',
        labels=['# 1', "# 2", '# 4', '# 8',
                # '# 16', '# 30'
                ], 
        ylabel='Cumulative episode rewards', xlabel='Episodes',
        fill_std=True,
        save_path=os.path.join(img_save_path, 'multiagent')
        # xtickrange = [100000, 250000]
    )
    
    print(f"========== Network resource usage ==========")
    print_cont_cunt(
        [
        multi_plas_plant1_path4, 
         hsac_path2, 
         mpdqn_path2], 
        labels=['LS-WNCS', 
                'Hybrid-SAC', 'MP-DQN'])
    

    
    