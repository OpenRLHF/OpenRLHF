"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

启动 web 页面可视化 RL 训练过程中间 metric 的数据状态。

Author: pankeyu
Date: 2023/10/31
"""
import os
import copy
import traceback

try:
    import ujson as json
except:
    import json
    print('`pip install ujson` can be faster.')

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff 


st.set_page_config(
    page_title="RL Logging Board",
    page_icon="📖",
    layout='wide'
)


def load_log_file(
    logdir: os.PathLike,
    max_samples_each_step: int
):
    """
    解析本地log文件。

    Args:
        logdir (os.PathLike): _description_
        max_samples_each_step (int): _description_
    """
    st.session_state['logging_name'] = logdir
    st.session_state['max_samples_each_step'] = max_samples_each_step
    st.session_state['logging_data'] = {}
    error_lines, success_lines = 0, 0
    
    all_logs = os.listdir(logdir)
    
    progress_text = f"Processing all files..."
    loading_files_bar = st.progress(0., text=progress_text)

    progress_text = f"Processing each file samples..."
    loading_samples_bar = st.progress(0., text=progress_text)
    
    
    for log_index in range(len(all_logs)):
        
        if not all_logs[log_index].endswith('.jsonl'):
            continue
        
        rl_log_file = os.path.join(
            logdir, 
            all_logs[log_index]
        )
        
        mock_max_lines_num = 10000

        with open(rl_log_file, 'r', encoding='utf8', errors='ignore') as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                    data['step'] = int(data['step'])
                    if data['step'] not in st.session_state['logging_data']:
                        st.session_state['logging_data'][data['step']] = {
                            'prompt': [],
                            'response': [],
                            'ref_response': [],
                            'reward': [],
                            'ref_reward': [],
                            'response_tokens': [],
                            'logprobs': [],
                            'ref_logprobs': [],
                            'probs': [],
                            'ref_probs': [],
                            'values': [],
                            'token_rewards': [],
                            'kl': [],
                            'avg_kl': [],
                            'sum_kl': [],
                            'log_ratio': [],
                            'avg_log_ratio': [],
                            'sum_log_ratio': [],
                            'valid_reward': [],
                            'ref_valid_reward': [],
                            'response_tokens_len': [],
                        }
                    elif len(st.session_state['logging_data'][data['step']]['prompt']) >= max_samples_each_step:
                            percentage = (i + 1) / mock_max_lines_num
                            percentage = min(percentage, 1.0)
                            loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {i + 1} / {mock_max_lines_num} samples in each files...")
                        
                    for key in st.session_state['logging_data'][data['step']]:
                        if key in data:
                            st.session_state['logging_data'][data['step']][key].append(data[key])
                    
                    if 'response_tokens' in data:
                        st.session_state['logging_data'][data['step']]['response_tokens_len'].append(len(data['response_tokens']))
                        
                    if 'logprobs' in data and 'ref_logprobs' in data:
                        logp = np.array(data['logprobs'])
                        ref_logp = np.array(data['ref_logprobs'])
                        log_ratio = logp - ref_logp
                        kl = np.exp(log_ratio) - 1 - log_ratio
                        st.session_state['logging_data'][data['step']]['log_ratio'].append(log_ratio.tolist())
                        st.session_state['logging_data'][data['step']]['avg_log_ratio'].append(np.nanmean(log_ratio))
                        st.session_state['logging_data'][data['step']]['sum_log_ratio'].append(np.nansum(log_ratio))
                        st.session_state['logging_data'][data['step']]['kl'].append(kl.tolist())
                        st.session_state['logging_data'][data['step']]['avg_kl'].append(np.nanmean(kl))
                        st.session_state['logging_data'][data['step']]['sum_kl'].append(np.nansum(kl))
                        st.session_state['logging_data'][data['step']]['probs'].append(np.exp(logp).tolist())
                        st.session_state['logging_data'][data['step']]['ref_probs'].append(np.exp(ref_logp).tolist())
                    
                    success_lines += 1
                    
                except:
                    print(traceback.format_exc())
                    error_lines += 1
                
                percentage = (i + 1) / mock_max_lines_num
                percentage = min(percentage, 1.0)
                loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {i + 1} / {mock_max_lines_num} samples...")

    percentage = 1.0
    loading_samples_bar.progress(percentage, text=f"[{int(percentage * 100)}%] Processing {(success_lines + error_lines)} / {(success_lines + error_lines)} samples...")

    file_percentage = (log_index + 1) / len(all_logs)
    loading_files_bar.progress(file_percentage, text=f"[{int(file_percentage * 100)}%] Loading {log_index + 1} / {len(all_logs)} files...")
    
    st.toast(
        f'Loaded {success_lines + error_lines} sample(s), sucess: {success_lines}, error: {error_lines}.', 
        icon='🎉'
    )

    if not st.session_state['logging_data']:
        st.warning(f'No log file(s) found in {logdir}.', icon='⚠️')
        st.stop()

    all_steps = [int(s) for s in list(st.session_state["logging_data"].keys())]
    all_steps.sort()
    st.session_state['max_step_index'] = max(all_steps)
    st.session_state['min_step_index'] = min(all_steps)
    st.session_state['step_gap'] = 1 if len(all_steps) < 2 else all_steps[1] - all_steps[0]
    
    rewards_dict = {'step': [], 'reward': [], 'ref_reward': []}
    for step in st.session_state['logging_data']:
        st.session_state['logging_data'][step]['avg_reward'] = sum(st.session_state['logging_data'][step]['reward']) / len(st.session_state['logging_data'][step]['reward'])
        
        current_step_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['response']]
        st.session_state['logging_data'][step]['avg_length'] = int(sum(current_step_resp_length) / len(current_step_resp_length))
        
        current_step_ref_resp_length = [len(resp) for resp in st.session_state['logging_data'][step]['ref_response']]
        st.session_state['logging_data'][step]['avg_ref_length'] = int(sum(current_step_ref_resp_length) / len(current_step_ref_resp_length)) if len(current_step_ref_resp_length) else 0
        
        if len(st.session_state['logging_data'][step]['ref_reward']):
            st.session_state['logging_data'][step]['avg_ref_reward'] = sum(st.session_state['logging_data'][step]['ref_reward']) / len(st.session_state['logging_data'][step]['ref_reward']) if len(st.session_state['logging_data'][step]['ref_reward']) else 0
        else:
            st.session_state['logging_data'][step]['avg_ref_reward'] = 0
        rewards_dict['step'].append(step)
        rewards_dict['reward'].append(st.session_state['logging_data'][step]['avg_reward'])
        rewards_dict['ref_reward'].append(st.session_state['logging_data'][step]['avg_ref_reward'])
    
    rewards_df = pd.DataFrame.from_dict(rewards_dict)
    st.session_state['reward_df'] = rewards_df.set_index('step')


def plot_filled_line(
    x: list,
    y_list_list: list,
    data_names: list,
    colors: list,
    title=None
):
    """
    绘制带有阴影的折线图，阴影上下界为当前x对应的y列表中的最大、最小值。

    Args:
        x (list): step 横轴索引
        y_list_list (line_num, steps, step_wise): 可绘制多条直线，维度为：绘制折线条数，总共的step数，每个step对应几个y值
        data_names (list): 每条折线的名字列表
        colors (list): 每条折线的颜色列表（rgb）, e.g. -> ['255,171,171']
    """
    fig = go.Figure()
    
    x_rev = x[::-1]
    for i in range(len(y_list_list)):
        y_list = y_list_list[i]
        y_mean, y_lower, y_upper = [], [], []
        for y in y_list:
            y_arr = np.array(y)
            mean, std = float(y_arr.mean()), float(y_arr.std())
            y_mean.append(mean)
            y_lower.append(mean - std)
            y_upper.append(mean + std)
            # y_lower.append(min(y))
            # y_upper.append(max(y))
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='toself',
            fillcolor=f'rgba({colors[i]},0.1)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=data_names[i],
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            line_color=f'rgb({colors[i]})',
            name=data_names[i],
        ))

    fig.update_traces(mode='lines')
    
    if title:
        fig.update_layout(
            title=title,
            legend=dict(orientation="h")
        )

    return fig


def init_sidebar():
    """
    侧边栏实例化。
    """
    st.sidebar.markdown(
        "<h1 style='text-align: center;'>📖 RL Logging Board</h1>", 
        unsafe_allow_html=True
    )

    base_root_path = st.sidebar.text_input(
        "Log(s) Root Path",
        value='../../../logs/rl_logging_board',
    )
    
    if not os.path.exists(base_root_path):
        st.warning(f'Log(s) Root Path: `{base_root_path}` is not exists.', icon='⚠️')
        st.stop()
    
    all_log_path_in_logdir = os.listdir(base_root_path)
    
    if not all_log_path_in_logdir:
        st.warning('No log files found.')
        st.code("""Logging Dir should be like:  
Base Log Dir  
    |__eval_topk_0_topp_1 (dir for evaluate logs)   
    |   |__eval.jsonl  
    |__topk_0_topp_1 (dir for training logs, only for rl logs)  
        |__rollout_data_rank_0_1313.jsonl  
    ...   
""")
        st.stop()

    log_name = st.sidebar.selectbox(
        'Choose Log Name',
        options=all_log_path_in_logdir,
        index=len(all_log_path_in_logdir) - 1
    )
    
    max_samples_each_step = st.sidebar.number_input(
        'Max Samples Each Step',
        help='当step batch size 过大时可能会造成平台卡顿，可设置阈值来下采样每个step的数据。',
        value=128,
        max_value=10240,
        min_value=1
    )
    
    load_btn = st.sidebar.button(
        "Load & View",
        use_container_width=True
    )
    
    if load_btn and (
        'logging_data' not in st.session_state 
        or 
        log_name != st.session_state['logging_name']
        or
        max_samples_each_step != st.session_state.get('max_samples_each_step', -1)
    ):
        load_log_file(
            os.path.join(base_root_path, log_name), 
            max_samples_each_step
        )
    
    with st.sidebar.expander('🧩 module setting', expanded=True):
        st.session_state['show_reward_logging'] = st.checkbox('Reward 曲线图', value=True)
        st.session_state['var_scaling'] = st.slider('Variance Scaling', min_value=0.1, max_value=1.0, value=0.2, help='Reward 曲线图阴影面积调整（对方差做 scaling）。')
        st.session_state['zero_shift'] = st.checkbox('Zero Shift', value=False, help='是否将所有reward曲线的第一项都平移到0（仅用于对比变化趋势）。')
        st.session_state['show_response'] = st.checkbox('Response 对比', value=True)

    with st.sidebar.expander('⚙️ show details setting', expanded=True):
        st.session_state['use_logp_as_kl'] = st.checkbox('Use LogP as KL', value=True, help='在 Reward 曲线图中用 LogProb 替代 KL 展示。')
        st.session_state['drop_pad'] = st.checkbox('Drop Padding Token', value=True)
        st.session_state['pad_token'] = st.text_input('Pad Token', value='<PAD>', disabled=not st.session_state['drop_pad'])
        st.session_state['drop_sys_prompt'] = st.checkbox('Drop System Prompt', value=True)
        st.session_state['end_token_of_sys_prompt'] = st.text_input('End Token of System Prompt', value='<endofsystem>', disabled=not st.session_state['drop_sys_prompt'])
        st.session_state['show_charts'] = st.checkbox('Show Charts', value=True)
        st.session_state['show_batch_samples'] = st.checkbox('Show Batch Samples', value=True)
        st.session_state['show_samples_pair'] = st.checkbox('Show Samples Pair', value=True)
        st.session_state['show_token_heat_map'] = st.checkbox('Show Heat Map', value=True)

def plot_filled_line(
    x: list,
    y_list_list: list,
    data_names: list,
    colors: list,
    title=None,
    var_scaling=1.
):
    """
    绘制带有阴影的折线图，阴影上下界为当前x对应的y列表中的最大、最小值。

    Args:
        x (list): step 横轴索引
        y_list_list (line_num, steps, step_wise): 可绘制多条直线，维度为：绘制折线条数，总共的step数，每个step对应几个y值
        data_names (list): 每条折线的名字列表
        colors (list): 每条折线的颜色列表（rgb）, e.g. -> ['255,171,171']
    """
    fig = go.Figure()
    
    x_rev = x[::-1]
    for i in range(len(y_list_list)):
        y_list = y_list_list[i]
        zero_shift_value = 0
        y_mean, y_lower, y_upper = [], [], []
        
        for idx, y in enumerate(y_list):    
            y_arr = np.array(y)
            if idx == 0 and st.session_state['zero_shift']:
                zero_shift_value = np.nanmean(y_arr)
            
            y_arr = y_arr - zero_shift_value
            mean, std = float(np.nanmean(y_arr)), float(np.nanstd(y_arr))
            std *= var_scaling
            y_mean.append(mean)
            y_lower.append(mean - std)
            y_upper.append(mean + std)
        y_lower = y_lower[::-1]

        fig.add_trace(go.Scatter(
            x=x + x_rev,
            y=y_upper + y_lower,
            fill='toself',
            fillcolor=f'rgba({colors[i]},0.1)',
            line_color='rgba(255,255,255,0)',
            showlegend=False,
            name=data_names[i],
        ))
        fig.add_trace(go.Scatter(
            x=x, y=y_mean,
            line_color=f'rgb({colors[i]})',
            name=data_names[i],
        ))

    fig.update_traces(mode='lines')
    
    if title:
        fig.update_layout(
            title=title,
            legend=dict(orientation="h")
        )

    return fig


def main_page():
    """
    Metrics Page.
    """
    if "logging_data" not in st.session_state:
        st.info("Please Press 「Load & View」Button to load log.")
    else:
        if st.session_state['show_reward_logging']:
            step_reward_tab, step_kl_tab, resp_len_tab = st.tabs([
                'Step-Reward', 
                'Step-KL', 
                'Step-RespLen'
            ])
            
            with step_reward_tab:
                steps, reward, ref_reward, valid_reward, ref_valid_reward = [], [], [], [], []
                for step, value_dict in st.session_state['logging_data'].items():
                    steps.append(step)
                    reward.append(value_dict['reward'])

                    if value_dict['ref_reward']:
                        ref_reward.append(value_dict['ref_reward'])
                    
                    if value_dict['valid_reward']:
                        valid_reward.append(value_dict['valid_reward'])
                    
                    if value_dict['ref_valid_reward']:
                        ref_valid_reward.append(value_dict['ref_valid_reward'])
                
                all_curves = {
                    'ref_reward': {
                        'value': ref_reward,
                        'color': '132,201,255'
                    },
                    'reward': {
                        'value': reward,
                        'color': '255,171,171'
                    }, 
                    'ref_valid_reward': {
                        'value': ref_valid_reward,
                        'color': '132,155,200'
                    }, 
                    'valid_reward': {
                        'value': valid_reward,
                        'color': '200,155,200'
                    }
                }
                
                candidate_curves = [key for key in all_curves if all_curves[key]['value']]
                
                show_curves = st.multiselect(
                    'Show Rewards',
                    candidate_curves,
                    candidate_curves,
                    label_visibility='collapsed'
                )
                
                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[all_curves[r]['value'] for r in show_curves],
                    data_names=show_curves,
                    colors=[all_curves[r]['color'] for r in show_curves],
                    title='👾 Rewards Logging (Step level)',
                    var_scaling=st.session_state['var_scaling']
                )

                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)

            with step_kl_tab:
                steps, kl = [], []

                if st.session_state['use_logp_as_kl']:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if all(value_dict['avg_log_ratio']): 
                            steps.append(step)
                            kl.append(value_dict['avg_log_ratio'])
                else:
                    for step, value_dict in st.session_state['logging_data'].items():
                        if all(value_dict['kl']):
                            steps.append(step)
                            kl.append(value_dict['avg_kl'])
                
                reward_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[kl],
                    data_names=['KL'],
                    colors=['255,165,0'],
                    title='👾 KL Logging (Step level)'
                )
                st.plotly_chart(reward_fig, theme="streamlit", use_container_width=True)
            
            with resp_len_tab:
                steps, resp_len = [], []

                for step, value_dict in st.session_state['logging_data'].items():
                    if value_dict['response_tokens_len']:
                        steps.append(step)
                        resp_len.append(value_dict['response_tokens_len'])

                resp_len_fig = plot_filled_line(
                    x=steps,
                    y_list_list=[resp_len],
                    data_names=['resp_len'],
                    colors=['255,165,0'],
                    title='👾 Response Length Logging (Step level)'
                )
                st.plotly_chart(resp_len_fig, theme="streamlit", use_container_width=True)
        
        if st.session_state['show_response']:
            st.markdown('⚡️ **Each Step Response**')
            
            if st.session_state['min_step_index'] == st.session_state['max_step_index']:
                step_index = st.session_state['min_step_index']
            elif (
                len(st.session_state['logging_data']) > 2 
                and 
                list(st.session_state['logging_data'].keys())[2] - list(st.session_state['logging_data'].keys())[1] != list(st.session_state['logging_data'].keys())[1] - list(st.session_state['logging_data'].keys())[0]
            ):
                step_index = st.selectbox(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    list(st.session_state['logging_data'].keys()),
                    index=0
                )
            else:
                step_index = st.slider(
                    f"Step Index({st.session_state['max_step_index']} total steps):",
                    min_value=st.session_state['min_step_index'],
                    max_value=st.session_state['max_step_index'],
                    value=st.session_state['min_step_index'],
                    step=st.session_state['step_gap']
                )

            cur_step_content_dict = st.session_state['logging_data'][step_index]
            cur_step_filtered_content_dict = copy.deepcopy(cur_step_content_dict)
            
            cur_step_filtered_content_dict['prompt'] = []
            for prompt in cur_step_content_dict['prompt']:
                if st.session_state['drop_pad']:
                    prompt = prompt.replace(st.session_state['pad_token'], '').strip()
                if st.session_state['drop_sys_prompt']:
                    prompt = prompt.split(st.session_state['end_token_of_sys_prompt'])[-1]
                cur_step_filtered_content_dict['prompt'].append(prompt)

            cur_step_filtered_content_dict['response'] = [c.replace(st.session_state['pad_token'], '').strip() if st.session_state['drop_pad'] else c for c in cur_step_content_dict['response']]
            cur_step_filtered_content_dict['reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['ref_reward'])]
            cur_step_filtered_content_dict['valid_reward_gap'] = [r - ref_r for r, ref_r in zip(cur_step_content_dict['reward'], cur_step_content_dict['valid_reward'])]
            
            if st.session_state['show_charts']:

                if not cur_step_filtered_content_dict['ref_reward']:
                    cur_step_filtered_content_dict['ref_reward'] = [0] * len(cur_step_filtered_content_dict['reward'])

                c1, c2, c3 = st.columns([6, 6, 6])

                with c1:                                                    # reward 分布
                    reward_distribution_dict = {
                        'sample_index': [],
                        'reward': [],
                        'tag': []
                    }
                    for sample_index, (reward, ref_reward) in enumerate(zip(cur_step_filtered_content_dict['reward'], cur_step_filtered_content_dict['ref_reward'])):
                        reward_distribution_dict['sample_index'].append(sample_index)
                        reward_distribution_dict['reward'].append(reward)
                        reward_distribution_dict['tag'].append('Reward')
                        reward_distribution_dict['sample_index'].append(sample_index)
                        reward_distribution_dict['reward'].append(ref_reward)
                        reward_distribution_dict['tag'].append('Ref Reward')

                    reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                    fig = px.bar(
                        reward_distribution_df, 
                        x="sample_index", 
                        y="reward", 
                        color="tag",
                        barmode='group',
                        color_discrete_sequence=px.colors.diverging.Portland,
                        title="Reward in current batch samples"
                    )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                
                with c2:                                                    # reward gap 分布
                    reward_distribution_dict = {
                        'sample_index': [i for i in range(len(cur_step_filtered_content_dict['reward_gap']))],
                        'reward_gap': cur_step_filtered_content_dict['reward_gap']
                    }
                    reward_distribution_df = pd.DataFrame.from_dict(reward_distribution_dict)
                    fig = px.bar(
                        reward_distribution_df, 
                        x="sample_index", 
                        y="reward_gap", 
                        color="reward_gap", 
                        color_discrete_sequence=['red'],
                        title="Reward Gap (r - ref_r) in current batch"
                    )
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                with c3:                                                    # reward 方差分布
                    if cur_step_filtered_content_dict['ref_reward']:
                        hist_data = [
                            cur_step_filtered_content_dict['ref_reward'],
                            cur_step_filtered_content_dict['reward'],
                        ]
                        group_labels = ['Ref Rewards', 'Rewards']
                    else:
                        hist_data = [cur_step_filtered_content_dict['reward']]
                        group_labels = ['Rewards']

                    fig = ff.create_distplot(
                        hist_data, 
                        group_labels, 
                        bin_size=[.02, .02],
                        curve_type='normal'
                    )
                    fig.update_layout(title="Reward Distribution in current batch")
                    st.plotly_chart(fig, use_container_width=True)

            showed_keys = [
                'prompt', 
                'response', 
                'reward', 
                'valid_reward', 
                'avg_log_ratio', 
                'sum_log_ratio', 
                'avg_kl', 
                'sum_kl', 
                'ref_response', 
                'ref_reward', 
                'ref_valid_reward', 
                'reward_gap', 
                'valid_reward_gap'
            ]
            candidate_keys = [k for k in showed_keys if cur_step_filtered_content_dict[k]]
            content_dict = dict([(k, cur_step_filtered_content_dict[k]) for k in candidate_keys])
            content_df = pd.DataFrame.from_dict(content_dict)
            
            if st.session_state['show_batch_samples']:
                st.dataframe(
                    content_df, 
                    use_container_width=True,
                    height=350
                )

            if st.session_state['show_samples_pair']:    
                
                c1, c2, c3 = st.columns([1, 1, 4])
                with c1:
                    if step_index == st.session_state['min_step_index']:
                        delta_char = 0
                    else:
                        try:
                            cur_avg_len = st.session_state['logging_data'][step_index]['avg_length']
                            last_avg_len = st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_length']
                            delta_char = cur_avg_len - last_avg_len
                        except:
                            delta_char = 0
                    st.metric(                                                  # actor在当前step下的平均回复长度，delta为与上一个step的比较
                        'Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_length']} 字",
                        delta=f'{delta_char} 字'
                    )
                
                with c2:                                                        # ref_model在当前step下的平均回复长度，delta为与上一个step的比较
                    try:
                        delta_char = 0 if step_index == st.session_state['min_step_index'] else st.session_state['logging_data'][step_index]['avg_ref_length'] - st.session_state['logging_data'][step_index-st.session_state['step_gap']]['avg_ref_length']
                    except:
                        delta_char = 0
                    st.metric(
                        'Ref Response Average Length',
                        value=f"{st.session_state['logging_data'][step_index]['avg_ref_length']} 字",
                        delta=f'{delta_char} 字'
                    )
                
                with c3:
                    sample_index = st.number_input(
                        f'Sample index in current step batch: ', 
                        min_value=0,
                        max_value=len(cur_step_filtered_content_dict['response']) - 1,
                        value=0
                    )
                
                # 单样本展示 response - ref_response 的回复
                c1, c2, c3, c4 = st.columns([4, 4, 4, 2])
                with c1:
                    st.markdown('<font color="#B0C4DE">Prompt</font>', unsafe_allow_html=True)
                    content = cur_step_filtered_content_dict["prompt"][sample_index].replace('\n', '  \n').replace('~', '～')
                    st.markdown(
                        f'<font color="#B0C4DE">{content}</font>',
                        unsafe_allow_html=True
                    )
                with c2:
                    st.markdown(':green[Response]')
                    content = cur_step_filtered_content_dict["response"][sample_index].replace('\n', '  \n').replace('~', '～')
                    st.markdown(
                        f'<font color="#3DD56D">{content}</font>',
                        unsafe_allow_html=True
                    )
                with c3:
                    st.markdown(':blue[Ref Response]')
                    if (
                            "ref_response" in cur_step_filtered_content_dict
                            and
                            cur_step_filtered_content_dict["ref_response"]
                    ):
                        content = cur_step_filtered_content_dict["ref_response"][sample_index].replace('\n', '  \n').replace('~', '～')
                        st.markdown(
                            f'<font color="#60B4FF">{content}</font>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.info('No `ref_response` found in log line data.')
                with c4:
                    st.markdown(':orange[Reward Gap]')
                    reward_gap = round(cur_step_filtered_content_dict["reward_gap"][sample_index], 4) if cur_step_filtered_content_dict["reward_gap"] else 0.
                    st.metric(
                        ' ', 
                        value=reward_gap
                    )

                # 展示更详细的 token-level 的信息
                if 'token_rewards' in cur_step_filtered_content_dict and cur_step_filtered_content_dict['token_rewards']:
                    # 检查 resp_tokens 的长度和 logprobs 的长度是否对齐
                    resp_token_len = len(cur_step_filtered_content_dict['response_tokens'][sample_index])
                    logp_len = len(cur_step_filtered_content_dict['logprobs'][sample_index])
                    if resp_token_len != logp_len:
                        st.info(
                            f'Note: `resp_tokens` (len: {resp_token_len}) is not equal to `logprobs` (len: {logp_len}), this may caused by <PAD> tokens, CLIP response tokens!',
                            icon='⚠️'
                        )
                        cur_step_filtered_content_dict['response_tokens'][sample_index] = cur_step_filtered_content_dict['response_tokens'][sample_index][:logp_len]
                    
                    show_values = st.multiselect(
                        'Select show value(s)',
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob'],
                        ['token_reward', 'log_ratio', 'kl', 'token_value', 'logp', 'ref_logp', 'prob', 'ref_prob']
                    )
                    
                    new_dict, index_list = {}, []
                    
                    if st.session_state['drop_pad'] and cur_step_filtered_content_dict['response_tokens'][sample_index][-1] == st.session_state['pad_token']:
                        first_pad_token_idx = cur_step_filtered_content_dict['response_tokens'][sample_index].index(st.session_state['pad_token'])
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index][:first_pad_token_idx]
                    else:
                        response_tokens_without_pad_token = cur_step_filtered_content_dict['response_tokens'][sample_index]
                    
                    for token_idx in range(len(response_tokens_without_pad_token)):
                        if cur_step_filtered_content_dict['response_tokens']:
                            resp_token = cur_step_filtered_content_dict['response_tokens'][sample_index][token_idx]
                            resp_token = f'{token_idx} - {resp_token}'
                            if resp_token not in new_dict:
                                new_dict[resp_token] = []
                        
                        if cur_step_filtered_content_dict['token_rewards']:
                            token_reward = cur_step_filtered_content_dict['token_rewards'][sample_index][token_idx]
                            if 'token_reward' in show_values:
                                new_dict[resp_token].append(token_reward)
                                if 'token_reward' not in index_list:
                                    index_list.append('token_reward')
                        
                        if cur_step_filtered_content_dict['log_ratio']:
                            log_ratio = cur_step_filtered_content_dict['log_ratio'][sample_index][token_idx]
                            if 'log_ratio' in show_values:
                                new_dict[resp_token].append(log_ratio)
                                if 'log_ratio' not in index_list:
                                    index_list.append('log_ratio')
                        
                        if cur_step_filtered_content_dict['kl']:
                            kl = cur_step_filtered_content_dict['kl'][sample_index][token_idx]
                            if 'kl' in show_values:
                                new_dict[resp_token].append(kl)
                                if 'kl' not in index_list:
                                    index_list.append('kl')
                        
                        if cur_step_filtered_content_dict['values']:
                            value = cur_step_filtered_content_dict['values'][sample_index][token_idx]
                            if 'token_value' in show_values:
                                new_dict[resp_token].append(value)
                                if 'token_value' not in index_list:
                                    index_list.append('token_value')
                        
                        if cur_step_filtered_content_dict['logprobs']:
                            logp = cur_step_filtered_content_dict['logprobs'][sample_index][token_idx]
                            if 'logp' in show_values:
                                new_dict[resp_token].append(logp)
                                if 'logp' not in index_list:
                                    index_list.append('logp')

                        if cur_step_filtered_content_dict['ref_logprobs']:
                            ref_logp = cur_step_filtered_content_dict['ref_logprobs'][sample_index][token_idx]
                            if 'ref_logp' in show_values:
                                new_dict[resp_token].append(ref_logp)
                                if 'ref_logp' not in index_list:
                                    index_list.append('ref_logp')
                        
                        if cur_step_filtered_content_dict['probs']:
                            prob = cur_step_filtered_content_dict['probs'][sample_index][token_idx]
                            if 'prob' in show_values:
                                new_dict[resp_token].append(prob)
                                if 'prob' not in index_list:
                                    index_list.append('prob')
                        
                        if cur_step_filtered_content_dict['ref_probs']:
                            ref_prob = cur_step_filtered_content_dict['ref_probs'][sample_index][token_idx]
                            if 'ref_prob' in show_values:
                                new_dict[resp_token].append(ref_prob)
                                if 'ref_prob' not in index_list:
                                    index_list.append('ref_prob')
                                    
                    try:
                        token_level_df = pd.DataFrame.from_dict(new_dict)
                        renamed_index_dict = dict((i, name) for i, name in enumerate(index_list))
                        token_level_df.rename(
                            index=renamed_index_dict, 
                            inplace=True
                        )
                        
                        st.dataframe(
                            token_level_df.style.background_gradient(axis=1, cmap="binary"), 
                            use_container_width=True
                        )
                        
                        if st.session_state['show_token_heat_map']:
                            fig = px.imshow(
                                token_level_df, 
                                text_auto=True,
                                aspect="auto",
                                color_continuous_scale="balance",
                            )
                            fig.update_xaxes(side="top")
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                    except Exception as e:
                        st.error(f'Error occured: {e}.')
                        st.write(new_dict)


if __name__ == '__main__':
    init_sidebar()
    main_page()