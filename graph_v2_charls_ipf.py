from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import os
import time
from typing import TypedDict, List, Dict, Any, Optional
from langchain_openai import OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd
import numpy as np
from scipy import stats
import json
import asyncio
import aiohttp
import aiofiles
from sklearn.preprocessing import LabelEncoder
from ipfn import ipfn
from sklearn.metrics import normalized_mutual_info_score
import random
from collections import Counter

# ---加载环境变量
load_dotenv(override=True)
time1 = time.time()

# ---定义图的状态类型
class GraphState(TypedDict):

    df: pd.DataFrame                 # 原始 200 条数据
    d_cols: List[str]                # 人口统计学列名
    h_cols: List[str]                # 高级信息列名
    df_large: pd.DataFrame           # 万条待填充的硅基样本 (只包含 D 列)    
    validated_edges: List[Dict]      # 节点1统计验证后的边
    s_target: Dict                   # 节点 B 输出: 最终的目标分布
    profiles: Dict[str, Any]         # 结构: { "ProfileID_xyz": {"condition": "Age=Young,Income=Low", "indices": [0, 5, 99...]} }
    grouping_keys: List[str]        # 记录哪些 D 列是关键列 (用于分组)
    current_iteration: int          # 当前迭代次数
    generation_plan: Dict           # 传递给 Generator 的计划
    failed_attempts: List[Dict]     # 记录失败的填充尝试
    ignored_features: List[str]   # 记录无法填充的特征，防止死循环
    finished: bool                  # 标记整个流程是否完成
    corrections: Dict                 # 记录 LLM 对联合分布的修正结果

# ---定义中间工具与函数
# 计算归一化互信息 (NMI),转换为字符串以处理离散分类变量
def calculate_nmi(df: pd.DataFrame, col_a: str, col_b: str) -> float:

    val_a = df[col_a].astype(str)
    val_b = df[col_b].astype(str)
    return normalized_mutual_info_score(val_a, val_b)


# ---定义图的节点
# 节点 0: 定义初始状态(input_jsonl\real_distribution_metrics)
def initialize_state(state: GraphState) -> GraphState:
    d_cols = ['age_bin', 'gender', 'income_bin', 'family_size', 'marry', 'edu', 'health_status']
    h_cols = ['hospital', 'exercise', 'ins', 'satlife', 'social_need']

    df_real = pd.read_csv(r"/root/v2/CHARLS_processed_2020.csv")
    df_real = df_real.drop(columns=['row_id','iwy'])
    df_real = df_real[df_real["income_total"] >= 0]
    df_real['age_bin'] = pd.cut(df_real['age'], bins=[0, 59, 64, 69, 74, 79, np.inf], labels=['60-', '60-64', '65-69', '70-74', '75-79', '80+'])
    df_real['income_bin'] = pd.qcut(df_real['income_total'], q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
    # 删除原始连续列，保留分箱后的列
    df_real = df_real.drop(columns=['age', 'income_total'])
    df_real = df_real[df_real["family_size"] <= 10]
    # df_real['family_size'] = pd.cut(df_real['family_size'], bins=[0, 1, 2, 4, np.inf], labels=['1', '2', '3-4', '5+'])
    df_real = df_real.dropna()

    df_large = pd.read_csv(r"/root/v2/CHARLS_ipf.csv")
    # df_large只有d变量，还需要补齐 H 特征列名
    for h in h_cols:
        df_large[h] = np.nan

    df = df_real.sample(n=200, random_state=24)
    
    current_iteration = 0

    return {
        "df": df,
        "df_large": df_large,
        "d_cols": d_cols,
        "h_cols": h_cols,
        "current_iteration": current_iteration,
        "validated_edges": [],
        "generation_plan": {},
        "failed_attempts": [],
        "ignored_features": [],
        "profiles": {},
        "grouping_keys": [],
        "s_target": {},
        "finished": False,
        "corrections": {}
    }


# 节点1: LLM 提出语义依赖假设,小样本统计剪枝
async def llm_proposal(state: GraphState) -> GraphState:
    df = state["df"]
    d_cols = state["d_cols"]
    h_cols = state["h_cols"]
    # 准备全量数据字符串 (只转换一次以节省开销)变成json方便llm阅读
    full_df_dict = df.astype(object).fillna("").replace({np.nan: None}).to_dict(orient="records")
 
    # 参数设置
    TOTAL_RUNS = 10    # 提议总次数
    VOTE_THRESHOLD = 6 # 必须出现多少次才保留
    NMI_THRESHOLD = 0.02 # 互信息阈值   
    print(f"--- 启动并发图结构提议 (Runs: {TOTAL_RUNS}, Min Votes: {VOTE_THRESHOLD}) ---")

    # 定义单次运行的逻辑
    async def _single_run(run_id):
        llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.8)

        template = """
        [System Ref: {time}]
        你是一个专业的社会学家。请分析以下变量列表，指出哪些变量之间在逻辑上存在较强的因果或相关关系。
        **提出18-23条边。**
        【变量列表】
        人口统计学特征 (D): {d_cols}
        高级特征 (H): {h_cols}
        各特征代表含义如下：
            gender：性别
            age_bin：年龄组
            marry：婚姻状况
            income_bin：总收入水平
            family_size：家庭人数
            edu：教育水平
            health_status：健康状况
            hospital：住院情况
            exercise：运动习惯
            ins：退休金状况
            satlife：生活满意度
            social_need：日常社交需求

        请只关注:
        1. D -> H
        2. H <-> H
        
        参考样本为{df}
        只输出一个 JSON 格式的列表，不要任何解释。
        格式示例: [["age_bin", "hospital"], ["family_size", "ins"]]
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        try:
            # 异步调用
            result_str = await chain.ainvoke({
                "time": time.time(),
                "d_cols": d_cols, 
                "h_cols": h_cols, 
                "df": full_df_dict # 传入全量数据
            })
            
            # 简单清洗 (沿用原逻辑)
            clean_json = result_str.replace("```json", "").replace("```", "").strip()
            # 如果 result_str 包含 <think>，json.loads 可能会报错，
            # 但如果你确认之前的代码能跑，说明 parser 可能只返回了最后的部分，或者 deepseek 行为符合预期
            # 这里尝试找到 JSON 的起止位置进行截取，增加一点点鲁棒性而不影响原逻辑
            if "[" in clean_json and "]" in clean_json:
                start = clean_json.find("[")
                end = clean_json.rfind("]") + 1
                clean_json = clean_json[start:end]

            proposed_edges = json.loads(clean_json)
            
            # 内部 NMI 剪枝
            valid_pairs = []
            all_cols = set(d_cols + h_cols)
            
            for edge in proposed_edges:
                if len(edge) != 2: continue
                
                # [关键] 排序以保证 A-B 和 B-A 被视为同一条边
                col_a, col_b = sorted([edge[0], edge[1]])
                
                if col_a not in all_cols or col_b not in all_cols:
                    continue
                    
                nmi = calculate_nmi(df, col_a, col_b)
                if nmi >= NMI_THRESHOLD:
                    valid_pairs.append((col_a, col_b)) # 存为元组供后续计数
            
            return valid_pairs
            
        except Exception as e:
            print(f"Run {run_id} failed: {e}")
            return []

    # 1. 并发执行 10 次
    tasks = [_single_run(i) for i in range(TOTAL_RUNS)]
    results = await asyncio.gather(*tasks)
    
    # 2. 统计投票
    # 展平结果列表
    all_votes = [pair for run_res in results for pair in run_res]
    vote_counts = Counter(all_votes)
    
    final_validated_edges = []
    
    print(f"\n=== 投票筛选结果 (阈值 >= {VOTE_THRESHOLD}) ===")
    
    # 3. 筛选最终结果
    for pair, count in vote_counts.most_common():
        if count >= VOTE_THRESHOLD:
            col_a, col_b = pair
            # 取最后一次的 NMI 分数作为记录 (NMI是确定性的，算一次即可)
            score = calculate_nmi(df, col_a, col_b)
            final_validated_edges.append({"pair": [col_a, col_b], "score": score})
            print(f"  [保留] {col_a} <-> {col_b} (得票: {count}/{TOTAL_RUNS})")
        # else:
        #     print(f"  [淘汰] {pair[0]} <-> {pair[1]} (得票: {count})")

    print(f"最终保留了 {len(final_validated_edges)} 条高置信度边。")
    
    return {"validated_edges": final_validated_edges}

# 节点2: Bootstrap重采样构建目标置信分布
def bootstrap_target(state: GraphState) -> GraphState:
    df = state["df"]
    h_cols = state["h_cols"]
    validated_edges = state["validated_edges"]
    
    n_iterations = 1000  # 重采样次数
    
    # 初始化累加器
    marginal_acc = {col: {} for col in h_cols}
    # joint_acc Key 格式: "ColA|ColB"
    joint_acc = {} 
    for edge in validated_edges:
        key_str = f"{edge['pair'][0]}|{edge['pair'][1]}"
        joint_acc[key_str] = {}
    
    print(f"正在进行 {n_iterations} 次 Bootstrap 重采样...")
    
    for _ in range(n_iterations):
        # 有放回抽样
        sample_df = df.sample(n=len(df), replace=True)
        
        # 1. 累积边缘分布
        for col in h_cols:
            counts = sample_df[col].value_counts(normalize=True).to_dict()
            for cat, freq in counts.items():
                marginal_acc[col][cat] = marginal_acc[col].get(cat, 0.0) + freq
                
        # 2. 累积联合分布 (基于验证过的边)
        for edge in validated_edges:
            col_a, col_b = edge['pair']
            pair_key_str = f"{col_a}|{col_b}"
            
            # 计算联合频率
            joint_counts = sample_df.groupby([col_a, col_b], observed=False).size() / len(sample_df)
            for (val_a, val_b), freq in joint_counts.items():
                combo_key = f"{val_a}|{val_b}"
                joint_acc[pair_key_str][combo_key] = joint_acc[pair_key_str].get(combo_key, 0.0) + freq

    # 计算平均值
    final_marginals = {}
    for col, counts in marginal_acc.items():
        final_marginals[col] = {k: v / n_iterations for k, v in counts.items()}
        
    final_joints = {}
    for key_str, counts in joint_acc.items():
        final_joints[key_str] = {k: v / n_iterations for k, v in counts.items()}
        
    s_target = {"marginals": final_marginals, "joints": final_joints}
    print(">>> 正在启动 LLM 联合分布修正 (Reality Check)...")
    
    # 1. 筛选出样本中概率极低 (< 0.003) 的组合
    suspicious_candidates = {}
    has_candidates = False
    
    for key_str, dist in final_joints.items():
        # 找出概率 < 0.003 但 > 0 的项 (完全为0的项通常不在字典里，除非初始化过)
        # 如果需要检查未出现的组合，需要先做笛卡尔积填充，这里只检查“出现过但很稀有”的
        low_probs = [k for k, v in dist.items() if v < 0.003]
        if low_probs:
            suspicious_candidates[key_str] = low_probs
            has_candidates = True
            
    if has_candidates:
        llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.8)
        
        template = """
        你是一名社会学和统计学专家。
        我们在分析人口数据时，发现以下特征组合在样本中出现的概率极低（<0.3%）。
        这可能是因为样本偏差，也可能是因为这些组合在现实世界中确实不合逻辑。
        
        【任务】
        请根据常识判断以下组合：
        1. 如果该组合在现实中**较为合理**（例如“高收入”但“年龄30”），只是样本没采到，请将其概率修正为 **0.005**。
        2. 如果该组合在现实中**确实极少见或不逻辑**（例如“博士学历”但“年龄10岁”），请保持原样（即不输出）。
        3. 如果不确定，请保持原样（即不输出），大部分组合应尽量保持原样（即不输出），避免误伤合理组合。
        4. 最多只能修正2个组合，最少可以不进行修正。
        【待审查数据】
        {candidates}
        
        【输出要求】
        1. 必须输出标准的 JSON 格式。
        2. 格式结构必须与输入 Key 保持一致：
           {{
             "特征A|特征B": {{
                "值3|值4": 0.005
             }}
           }}
        3. **仅输出需要修改的项**。不需要修改的项不要包含在 JSON 中。
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()
        
        try:
            # 将字典转为字符串供 LLM 阅读
            candidates_str = json.dumps(suspicious_candidates, indent=2, ensure_ascii=False)
            
            result_str = chain.invoke({"candidates": candidates_str})
            
            # [关键处理] DeepSeek Reasoner 会输出 <think>...</think>
            # 必须剥离思维链，只保留最后的 JSON 内容
            if "</think>" in result_str:
                json_part = result_str.split("</think>")[-1].strip()
            else:
                json_part = result_str.strip()
                
            # 清理可能的 markdown 符号（双重保险）
            clean_json = json_part.replace("```json", "").replace("```", "").strip()
            
            if clean_json:
                corrections = json.loads(clean_json)
                print(corrections)
            else:
                print("LLM 认为所有低概率项均合理，未做修改。")
                
        except json.JSONDecodeError:
            print(f"LLM 返回格式错误，跳过修正。原始内容片段: {result_str[:100]}...")
        except Exception as e:
            print(f"LLM 修正过程发生未知错误: {e}")
    else:
        print("未发现低概率(<0.003)的可疑组合，跳过 LLM 修正。")

    print(s_target)
    
    # 更新状态
    return {"s_target": s_target, "corrections": corrections}


# 节点3: 将万条待填充的硅基样本进行分组
def profiling(state: GraphState) -> GraphState:
    df_large = state["df_large"]
    validated_edges = state["validated_edges"]
    d_cols = state["d_cols"]
    
    # 并不是所有 D 列都要用来分组，只有那些 Step A 验证过会影响 H 的 D 列才重要。
    relevant_d = set()
    for edge in validated_edges:
        col_a, col_b = edge['pair']
        # 如果 col_a 是 D列 且 col_b 是 H列 (D->H)
        if col_a in d_cols:
            relevant_d.add(col_a)
        # 如果 col_b 是 D列 (反向相关也算)
        if col_b in d_cols:
            relevant_d.add(col_b)
            
    # 如果没有发现任何强依赖，兜底策略：使用所有 D 列
    if not relevant_d:
        print("警告:未发现D->H强依赖，将使用所有D列进行分组。")
        grouping_keys = d_cols
    else:
        grouping_keys = list(relevant_d)
        
    print(f"关键分组特征: {grouping_keys}")

    # 步骤 2: 执行分桶 (Group By)
    # -------------------------------------------------------
    # 为了处理方便，我们将分组特征组合成一个字符串 ID，或者直接用 tuple,这里我们生成一个 Profile 字典
    
    profiles = {}
    
    # 使用 Pandas 的 groupby 快速分组
    # group_name 是一个 tuple，代表 (Value_Key1, Value_Key2...)
    # group_indices 是该组在 df_large 中的索引列表
    groups = df_large.groupby(grouping_keys)
    
    for group_vals, group_df in groups:
        # 1. 构建易读的条件描述字符串
        # 如果只有一个 key，group_vals 不是 tuple，需要转换
        if not isinstance(group_vals, tuple):
            group_vals = (group_vals,)
            
        condition_parts = []
        for k, v in zip(grouping_keys, group_vals):
            condition_parts.append(f"{k}={v}")
        condition_str = ", ".join(condition_parts)
        
        # 2. 生成唯一的 Profile ID (哈希或直接用字符串)
        # 简单的 ID: P_0, P_1... 或者使用条件哈希
        profile_id = f"Profile_{hash(condition_str) % 10000:04d}"
        
        # 3. 存储该组的元数据
        profiles[profile_id] = {
            "description": condition_str,
            "values": dict(zip(grouping_keys, group_vals)), # {'Age': 'Young', ...}
            "indices": group_df.index.tolist(),             # 这一组包含哪几千个人
            "count": len(group_df)
        }
        
    print(f"分组完成: 将 {len(df_large)} 划分为 {len(profiles)} 个 Profile 画像组。")
    
    # 展示前几个示例
    first_few = list(profiles.keys())[:2]
    for pid in first_few:
        p = profiles[pid]
        print(f"  - {pid}: {p['description']} (人数: {p['count']})")

    # 更新状态
    return {
        "profiles": profiles,
        "grouping_keys": grouping_keys
    }


# 节点4: 规划本轮生成任务计算差异 -> LLM 制定计划
def planner(state: GraphState) -> GraphState:
    """
    1. 动态扫描所有高级特征列，找出当前最大的统计缺口。
    2. 针对该缺口，找出哪些 Profile 还有空位 (NaN)。
    3. [新增] 检查 H-H 依赖，剔除已填特征与目标冲突的个体。
    4. 查 D-H 联合分布表，剔除逻辑上不合理的 Profile。
    5. 让 LLM 在剩下的合格 Profile 中分配生成名额。
    """    
    # 1. 解包状态
    df_large = state["df_large"]
    s_target = state["s_target"]
    profiles = state["profiles"]
    validated_edges = state["validated_edges"]
    
    # [新增] 获取忽略列表，防止死循环
    ignored_features = set(state.get("ignored_features", []))

    total_population = len(df_large)

    # 步骤 1: 全局扫描，寻找最紧迫的统计缺口 (Scarcity-based Scheduling)
    # ------------------------------------------------------------------
    # 策略变更：不再单纯寻找 gap 最大的，而是寻找 (gap / available_slots) 最大的
    # 原理：优先满足“供需比”最紧张的特征，防止 easy 任务占满所有坑位导致 hard 任务无处可填
    
    max_urgency = -1.0 # 初始化最大紧迫度
    target_feature = None
    target_category = None
    max_gap = 0
    
    # 遍历所有高级特征 (H cols)
    for h_col, target_dist in s_target["marginals"].items():
        # [新增] 如果该特征已被标记为无法填充，跳过
        if h_col in ignored_features:
            continue

        # --- [关键修改 1] 计算供给端 (Supply): 该列还有多少物理空位 ---
        # 这里计算的是硬性的 NaN 数量。如果 available_slots 很小，说明该特征快填满了。
        available_slots = df_large[h_col].isna().sum()

        # 如果该列已经填满了 (没有 NaN)，即使有统计缺口也无法操作，直接跳过
        if available_slots == 0:
            continue

        # 统计该列当前非空数据的分布
        valid_series = df_large[h_col].dropna()
        current_counts = valid_series.value_counts().to_dict()
            
        # 对比目标 (Demand)
        for cat, prob in target_dist.items():
            target_count = int(total_population * prob)
            existing_count = current_counts.get(cat, 0)
            gap = target_count - existing_count
            
            if gap > 0:
                # --- [关键修改 2] 计算紧迫度 (Urgency Score) ---
                # 公式：紧迫度 = 缺口需求 / 剩余可用空位
                # 场景 A: 缺口 500，剩余空位 10000 -> 紧迫度 0.05 (不急，空位大把)
                # 场景 B: 缺口 50， 剩余空位 60    -> 紧迫度 0.83 (超级急！再不填没位置了)
                urgency = gap / available_slots
                
                # 寻找紧迫度最高的任务
                if urgency > max_urgency:
                    max_urgency = urgency
                    target_feature = h_col
                    target_category = cat
                    max_gap = gap

    print(f"当前最大缺口: 特征[{target_feature}] - 类别[{target_category}] (缺 {max_gap} 人)")

    # 如果缺口很小或没有缺口
    if max_gap <= 0 or target_feature is None:
        print(">>> 无显著缺口或所有列已填充完毕，规划器待机。")
        
        return {"generation_plan": {},
                "finished": True
                }

    batch_size = min(max_gap, 300) 

    # ------------------------------------------------------------------
    # 步骤 2: 候选人筛选与逻辑守门员 (Hard Filtering)
    # ------------------------------------------------------------------
    
    # 2.1 基础筛选：找出哪些 Profile 在这个 target_feature 上还有空位
    pid_vacancies = {}
    nan_indices_set = set(df_large[df_large[target_feature].isna()].index)
    print(f"正在扫描 {len(nan_indices_set)} 个空位...")
    for pid, meta in profiles.items():
        profile_indices = set(meta['indices'])
        available_indices = list(profile_indices & nan_indices_set)
        
        if available_indices:
            pid_vacancies[pid] = available_indices
    print(f"{len(pid_vacancies)} 个 Profile 还有空位可以填充 {target_feature}。")
    # [新增] 准备联合分布查询表
    joints_lookup = s_target.get("joints", {})

    # 2.2 H-H 依赖检查：剔除那些“已填写的其他H特征”与“当前目标”冲突的行
    cleaned_pid_vacancies = {}
    
    for pid, indices in pid_vacancies.items():
        # 默认所有 indices 都是好的，除非被 H-H 规则剔除
        valid_indices_in_profile = set(indices)
        
        # 遍历所有边，寻找涉及 target_feature 的 H-H 边
        for edge in validated_edges:
            pair = edge['pair']
            if target_feature not in pair: continue
            
            # 找到另一端特征 (可能是 D 也可能是 H)
            other_col = pair[0] if pair[1] == target_feature else pair[1]
            
            # 如果 other_col 是 D列，已经在 Profile 定义里固定了，稍后在 2.3 检查
            # 这里只处理 other_col 是 H列 (高级特征) 的情况，且该 H列 必须已经被填了值
            if other_col not in state["d_cols"]: # 这是一个 H-H 依赖
                
                # 获取这批人在 other_col 上的值
                current_values = df_large.loc[list(valid_indices_in_profile), other_col]
                
                # 找出非空的行 (已填值的行需要检查逻辑冲突)
                filled_rows = current_values.dropna()
                
                if filled_rows.empty:
                    continue # 没人填过这个依赖列，无法判断冲突，跳过
                
                # 检查冲突
                rows_to_remove = set()
                for idx, val in filled_rows.items():
                    # 构建查询 Key
                    # 必须保证 Key 顺序与 validated_edges 一致
                    if pair[0] == other_col:
                        table_key = f"{other_col}|{target_feature}"
                        combo_key = f"{val}|{target_category}"
                    else:
                        table_key = f"{target_feature}|{other_col}"
                        combo_key = f"{target_category}|{val}"
                    
                    # 查表
                    prob = joints_lookup.get(table_key, {}).get(combo_key, 0.0)
                    
                    # 严格过滤：如果联合概率极低，说明已填的值与当前目标冲突
                    if prob < 0.003:
                        rows_to_remove.add(idx)
                
                # 从可用列表中移除冲突行
                if rows_to_remove:
                    valid_indices_in_profile -= rows_to_remove
        
        # 如果经过一轮清洗还有剩余名额，保留该 Profile
        if valid_indices_in_profile:
            cleaned_pid_vacancies[pid] = list(valid_indices_in_profile)

    # 用清洗后的列表替换原列表
    pid_vacancies = cleaned_pid_vacancies

    # 2.3 逻辑守门员：基于 Profile 定义 (D特征) 查表
    # 这里检查的是 D -> H 的固有逻辑冲突
    valid_profiles_for_task = []
    logic_exclusion_count = 0
    
    for pid in pid_vacancies.keys():
        profile_meta = profiles[pid]
        is_logical = True
        
        for edge in validated_edges:
            pair = edge['pair']
            
            # 这里的逻辑只针对 D -> H 检查 (H->H 已经在上面做过行级检查了)
            d_col, h_col = None, None
            
            if pair[1] == target_feature and pair[0] in profile_meta['values']:
                d_col, h_col = pair[0], pair[1]
                table_key = f"{d_col}|{h_col}"
                # [修正] 强制转字符串
                combo_key = f"{str(profile_meta['values'][d_col])}|{str(target_category)}"
                
            elif pair[0] == target_feature and pair[1] in profile_meta['values']:
                d_col, h_col = pair[1], pair[0]
                table_key = f"{h_col}|{d_col}"
                # [修正] 强制转字符串
                combo_key = f"{str(target_category)}|{str(profile_meta['values'][d_col])}"
            else:
                continue 
            
            if table_key in joints_lookup:
                prob = joints_lookup[table_key].get(combo_key, 0.0)
                if prob < 0.003:
                    is_logical = False
                    break
        
        if is_logical:
            valid_profiles_for_task.append(pid)
        else:
            logic_exclusion_count += 1

    print(f"逻辑过滤: 剔除 {logic_exclusion_count} 个不合理画像，剩余 {len(valid_profiles_for_task)} 个可用画像。")

    # [新增] 死循环保护机制
    if not valid_profiles_for_task:
        print(f"警告: 特征 {target_feature}={target_category} 无法找到合逻辑的 Profile。将其加入忽略列表。")
        # 更新忽略列表并返回，避免死循环
        new_ignored = list(ignored_features)
        new_ignored.append(target_feature)
        return {
            "generation_plan": {}, 
            "ignored_features": new_ignored  # 需要在 GraphState 类型定义中增加此字段
        }

    # ------------------------------------------------------------------
    # 步骤 3: LLM 提案 (Proposal Generation)
    # ------------------------------------------------------------------
    random.shuffle(valid_profiles_for_task)
    top_candidates = valid_profiles_for_task[:30]
    
    profile_desc_list = []
    for pid in top_candidates:
        # 注意：这里使用的是清洗过后的剩余空位数量
        count_available = len(pid_vacancies[pid])
        desc = f"ID: {pid} | 特征: {profiles[pid]['description']} | 剩余空位: {count_available}"
        profile_desc_list.append(desc)
        
    profile_context_str = "\n".join(profile_desc_list)

    failed_attempts = state.get("failed_attempts", [])
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.8)
    
    template = """
    你是一个负责人口数据合成的规划师。
    
    【当前任务】
    我们需要生成特征 [{feature}] 为 [{category}] 的样本。
    本次目标生成数量: {count} 人。
    
    【候选人群画像 (已通过逻辑筛选)】
    以下人群在逻辑上都允许具备该特征。
    请根据社会学常识，判断哪些人群最可能具备该特征，并分配生成名额。
    {profiles}
    
    【指令】
    1. 请从上述 ID 中选择合适的画像。
    2. 分配的数量 **绝对不能超过** 该 ID 的“剩余空位”。
    3. 所有分配的数量之和应尽量接近 {count}。
    4. 只能返回 JSON，格式如下:
    {{
        "Profile_ID_A": 数量,
        "Profile_ID_B": 数量
    }}

    PS上一轮出现的错误为{failed_attempts}，如果有请避免类似错误发生，如果没有请忽略。
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    
    generation_plan = {}
    
    try:
        print(">>> 正在请求 LLM 进行分配规划...")
        result_str = chain.invoke({
            "feature": target_feature,
            "category": target_category,
            "count": batch_size,
            "profiles": profile_context_str,
            "failed_attempts": failed_attempts
        })
        
        clean_json = result_str.replace("```json", "").replace("```", "").strip()
        allocation_dict = json.loads(clean_json)
        
        for pid, count in allocation_dict.items():
            if count > 0:
                # 使用清洗后的 vacancies 列表长度做校验
                max_allowed = len(pid_vacancies.get(pid, []))
                final_count = min(count, max_allowed)
                
                if final_count > 0:
                    generation_plan[pid] = {
                        target_feature: {
                            "value": target_category,
                            "count": final_count
                        }
                    }
        print(f"LLM 规划成功，将生成 {len(generation_plan)} 个组的样本。")
        
    except Exception as e:
        print(f"LLM 规划解析失败: {e}")
        generation_plan = {}

    return {
        "generation_plan": generation_plan,
        "current_iteration": state.get("current_iteration", 0) + 1,
    }


# 节点5: 生成批次数据
async def generator(state: GraphState) -> GraphState:
    print("---生成数据ing---")
    print(time.time() - time1)
    # 1. 解包状态
    df_large = state["df_large"]
    plan = state["generation_plan"]
    profiles = state["profiles"]
    
    # 初始化统计
    success_count = 0
    reject_count = 0
    
    # 修复：failed_attempts 应该累加，而不是每次覆盖
    # 获取上一轮的失败记录，或者初始化为空
    previous_failures = state.get("failed_attempts", [])
    current_failures = []

    # 初始化 LLM 和 Chain
    llm = ChatDeepSeek(model="deepseek-chat", temperature=1)
    
    template = """
    你是一个严格的数据逻辑审查员。请判断以下数据填充任务是否在语义上合理。

    【当前个体的已知画像】
    {profile_str}

    【待执行的填充任务】
    需要将特征 [{feature}] 设定为: {value}

    【判断标准】
    1. 检查是否存在明显的语义冲突。
    2. 考虑社会学常识。
    3. 如果略有不寻常但可能发生（例如：高学历但低收入），请判为合理（true）。
    4. 只有在逻辑完全无法自洽时，才判为拒绝（false）。

    【输出格式】
    只能输出一个 JSON 对象：
    {{
        "is_reasonable": true,
        "reason": "简短的理由或合理化解释"
    }}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    # --- 准备异步任务 ---
    sem = asyncio.Semaphore(10)

    tasks = []
    task_metadata = [] # 用于存储任务对应的元数据 (indices, feature, value 等)

    print(f"正在构建任务列表...")

    for pid, features_dict in plan.items():
        if pid not in profiles: 
            continue
        
        all_indices = profiles[pid]['indices']
        
        for feature_name, instruction in features_dict.items():
            target_val = instruction['value']
            count_needed = instruction['count']
            
            # --- 步骤 A: 锁定操作行 (CPU 密集型，保持同步) ---
            # 找出该 Profile 下，该特征目前为空的行
            candidate_mask = df_large.loc[all_indices, feature_name].isna()
            candidate_indices = df_large.loc[all_indices][candidate_mask].index.tolist()
            
            if not candidate_indices:
                print(f"  [Skip] Profile {pid} 无可用空位填充 {feature_name}。")
                continue

            # 截断所需数量
            if len(candidate_indices) > count_needed:
                selected_indices = candidate_indices[:count_needed]
            else:
                selected_indices = candidate_indices
            
            # --- 准备上下文 ---
            # 抽取代表样本
            rep_idx = selected_indices[0]
            existing_data = df_large.loc[rep_idx].dropna().to_dict()
            profile_context = [f"{k}: {v}" for k, v in existing_data.items()]
            profile_str = " | ".join(profile_context)

            # --- 定义单个异步校验函数 ---
            async def verify_logic(p_str, feat, val, sem_lock):
                async with sem_lock:
                    try:
                        result_str = await chain.ainvoke({
                            "profile_str": p_str,
                            "feature": feat,
                            "value": val
                        })
                        clean_json = result_str.replace("```json", "").replace("```", "").strip()
                        # 处理可能出现的非 JSON 尾部字符
                        if "}" in clean_json:
                             clean_json = clean_json[:clean_json.rfind("}")+1]
                        
                        res = json.loads(clean_json)
                        return res.get("is_reasonable", False), res.get("reason", "")
                    except Exception as e:
                        print(f"  [Async Error] {feat}={val}: {e}")
                        return False, "Error during LLM validation"

            # 将任务加入列表
            tasks.append(verify_logic(profile_str, feature_name, target_val, sem))
            
            # 记录元数据，以便后续对应结果
            task_metadata.append({
                "pid": pid,
                "indices": selected_indices,
                "feature": feature_name,
                "value": target_val
            })

    # --- 步骤 B: 并行执行所有校验 (I/O 密集型) ---
    if not tasks:
        print("本轮无有效生成任务。")
        return state

    print(f"开始并行执行 {len(tasks)} 个校验任务...")
    results = await asyncio.gather(*tasks)

    # --- 步骤 C: 处理结果并更新 DataFrame (同步执行) ---
    print("校验完成，正在应用更改...")
    
    for i, (is_valid, reason) in enumerate(results):
        meta = task_metadata[i]
        indices = meta["indices"]
        feature = meta["feature"]
        value = meta["value"]
        pid = meta["pid"]

        if is_valid:
            # 批量赋值
            df_large.loc[indices, feature] = value
            success_count += len(indices)
            # 可选: print(f"  [OK] {pid}: {feature}={value}")
        else:
            reject_count += len(indices)
            current_failures.append({
                "pid": pid, 
                "feature": feature, 
                "value": value,
                "reason": reason # 记录理由供 Planner 参考
            })
            print(f"  [REJECT] {pid} {feature}={value} | Reason: {reason}")

    print(f"执行完毕。成功填充: {success_count} 人, 拒绝/跳过: {reject_count} 人。")
    
    # 合并失败记录 (保留最近的 N 条，防止 context 过爆)
    final_failures = previous_failures + current_failures
    if len(final_failures) > 20:
        final_failures = final_failures[-20:]

    # 更新状态
    return {
        "df_large": df_large,
        "failed_attempts": final_failures
    }


# 节点6：对含空值行进行二次筛选与填充
def second_fill(state: GraphState) -> GraphState:
    with open("/root/v2/second_fill_log.txt", "a", encoding="utf-8") as log_f:
        log_f.write(f"\n--- 二次填充日志 {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")

    df_large = state["df_large"]
    h_cols = state["h_cols"]
    corrections = state.get("corrections", {})
    if not corrections:
        print("无 LLM 修正，跳过二次填充。")
        return {"df_large": df_large}
    
    # 筛选出仍有空值的行
    rows_with_nans = df_large[df_large[h_cols].isna().any(axis=1)]
    remaining_count = len(rows_with_nans)
    print(f"二次填充: 仍有 {remaining_count} 行含有空值。")
    target_fill_base = int(0.003 * len(df_large))

    # 遍历每一个修正规则
    for cols_key, vals_dict in corrections.items():
        # 解析特征名 "特征A|特征B"
        try:
            col_a, col_b = cols_key.split("|")
        except ValueError:
            print(f"[警告] 键格式错误: {cols_key}，应为 'ColA|ColB' 格式，跳过。")
            continue
            
        # 检查列是否存在于 DataFrame 中
        if col_a not in df_large.columns or col_b not in df_large.columns:
            print(f"[跳过] 列 {col_a} 或 {col_b} 不在 DataFrame 中。")
            continue

        # 遍历该特征组合下的所有值组合 "值3|值4"
        for vals_key, probability in vals_dict.items():
            try:
                val_a_str, val_b_str = vals_key.split("|")
            except ValueError:
                print(f"[警告] 值格式错误: {vals_key}，应为 'ValA|ValB' 格式，跳过。")
                continue
            
            # JSON中的值通常是字符串，但 DataFrame 中可能是 int/float。
            # 这里尝试根据 DataFrame 的列类型转换值的类型，否则匹配会失败 (例如 "3" != 3)。
            try:
                # 如果列是 float64, '3' -> 3.0; 如果列是 object, '3' -> '3'
                val_a = df_large[col_a].dtype.type(val_a_str)
                val_b = df_large[col_b].dtype.type(val_b_str)
            except Exception:
                # 如果转换极其困难（如 Object 类型），则使用原始字符串
                val_a, val_b = val_a_str, val_b_str

            # 条件1: A是目标值, B是空
            cond1 = (df_large[col_a] == val_a) & (df_large[col_b].isna())
            # 条件2: B是目标值, A是空
            cond2 = (df_large[col_b] == val_b) & (df_large[col_a].isna())
            # 条件3: A和B都是空
            cond3 = (df_large[col_a].isna()) & (df_large[col_b].isna())
            
            # 合并所有符合条件的行
            target_mask = cond1 | cond2 | cond3
            
            # 获取符合条件的索引
            candidate_indices = df_large.index[target_mask]
            available_count = len(candidate_indices)
            
            if available_count == 0:
                continue
                
            # 计算实际需要填充的数量：min(0.003 * len, available)
            fill_count = min(target_fill_base, available_count)
            
            # 随机采样索引 (无放回)
            # 必须使用 np.random.choice 而不是直接切片，以避免偏差
            fill_indices = np.random.choice(candidate_indices, fill_count, replace=False)
            
            # 执行填充
            df_large.loc[fill_indices, col_a] = val_a
            df_large.loc[fill_indices, col_b] = val_b
            # 输出日志
            with open("/root/v2/second_fill_log.txt", "a", encoding="utf-8") as log_f:
                log_entry = f"规则 [{cols_key}] - [{vals_key}]: {fill_indices.tolist()}\n"
                log_f.write(log_entry)
            
            print(f"规则 [{cols_key}]: {vals_key} 填充了 {fill_count} 行 (候选行: {available_count})")
  
    with open(r"/root/v2/synthetic_data.csv", "w", encoding="utf-8") as f:
        f.write(df_large.to_csv(index=False))
    print("二次填充完成，已保存至 synthetic_data.csv。")
    return {"df_large": df_large}

# --- 构建图
# 构建“生成-评估-校准-优化”(GECO)图
workflow = StateGraph(GraphState)
# 设置入口点
workflow.set_entry_point("initialize_state")
# 添加节点
workflow.add_node("initialize_state", initialize_state)
workflow.add_node("llm_proposal", llm_proposal)
workflow.add_node("bootstrap_target", bootstrap_target)
workflow.add_node("profiling", profiling)
workflow.add_node("planner", planner)
workflow.add_node("generator", generator)
workflow.add_node("second_fill", second_fill)

# 添加普通边
workflow.add_edge("initialize_state", "llm_proposal")
workflow.add_edge("llm_proposal", "bootstrap_target")
workflow.add_edge("bootstrap_target", "profiling")
workflow.add_edge("profiling", "planner")
workflow.add_edge("generator", "planner")
workflow.add_edge("second_fill", END)

# 添加条件边
# ---定义图的条件边
def continue_plan(state: GraphState) -> str:
    # 检查是否还有生成计划未完成
    if state["finished"]:
        return "second_fill"
    else:
        return "generator"
workflow.add_conditional_edges(
    "planner",
    continue_plan,
    {
        "second_fill": "second_fill",
        "generator": "generator"
    }
)


# 编译图
app = workflow.compile()

if __name__ == "__main__":
    async def main():
        await app.ainvoke(
                        input={},  # 这里放你的输入数据
                        config={"recursion_limit": 10000}
                        )

    asyncio.run(main())