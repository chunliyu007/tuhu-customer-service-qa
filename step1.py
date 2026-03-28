#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step1.py

里程碑 Milestone 1：本地数据读取与 Task 1 定量清洗统计。

功能概述：
1. 读取途虎智能客服离线对话数据（支持 CSV / Excel）。
2. 处理冗长字段名，映射为内部简短字段名，便于后续开发。
3. 基础数据清洗：丢弃 sessionid 为空的脏数据，去除首尾空格，处理简单换行。
4. 完成 Task 1 统计：
   - 总数据行数
   - toolsmodelanswer 字段中含有 <tools> 标签的数据行数量
   - <tools> 标签中各具体工具名称及调用次数
5. 按 userid + sessionid 聚合多行对话，形成按时间排序的会话列表，并打印聚合后的总会话数量。

使用说明：
    python step1.py --input 智能客服对话0316test.xlsx
    python step1.py --input data.csv

注意：
1. 本脚本只实现 Milestone 1，不涉及大模型调用和 HTML 报告生成。
2. 后续 Milestone 会在新的脚本 / 模块中实现，避免一次性写完所有代码。
"""

import argparse
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd


# ===========================
# 配置与常量
# ===========================

# 可能的字段名候选列表（根据 PRD 和表结构常见命名进行覆盖）
POSSIBLE_SESSION_ID_COLUMNS = [
    "sessionid",
    "session_id",
    "thyc_customer_service_tuhu_intent_15_log_explian_jv.sessionid",
]

POSSIBLE_USER_ID_COLUMNS = [
    "userid",
    "user_id",
    "uid",
    "thyc_customer_service_tuhu_intent_15_log_explian_jv.userid",
]

POSSIBLE_TOOLS_ANSWER_COLUMNS = [
    "toolsmodelanswer",
    "tools_model_answer",
    "tools_answer",
    "thyc_customer_service_tuhu_intent_15_log_explian_jv.toolsmodelanswer",
]

# 对话内容字段（这里只是为后续里程碑预留，当前里程碑仅用于聚合）
POSSIBLE_DIALOGUE_SENTENCE_COLUMNS = [
    "dialoguesentence",
    "dialogue_sentence",
    "user_query",
]

POSSIBLE_ANSWER_COLUMNS = [
    "answer",
    "robot_answer",
]

POSSIBLE_REWRITE_QUERY_COLUMNS = [
    "rewritequery",
    "rewrite_query",
]

# 时间排序字段候选：优先使用这些字段进行会话内排序
POSSIBLE_TIME_COLUMNS = [
    "createtime",
    "create_time",
    "logtime",
    "log_time",
    "timestamp",
]


# ===========================
# 工具函数
# ===========================

def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """
    在给定列名列表中，根据候选名列表找到最合适的列名。

    优先级：
    1. 精确大小写不敏感匹配
    2. 去掉表前缀（如 xxx.yyy）后的精确匹配
    3. 简单包含匹配（用于一定程度的容错）
    """
    lower_map: Dict[str, str] = {c.lower(): c for c in columns}

    # 1. 精确匹配（不区分大小写）
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]

    # 2. 去掉前缀后的精确匹配
    #    例如：thyc_customer_service_tuhu_intent_15_log_explian_jv.sessionid
    #    只保留最后一段 sessionid
    stripped_map: Dict[str, str] = {}
    for col in columns:
        stripped = col.split(".")[-1].lower()
        stripped_map.setdefault(stripped, col)

    for cand in candidates:
        key = cand.split(".")[-1].lower()
        if key in stripped_map:
            return stripped_map[key]

    # 3. 简单包含匹配（防止奇怪命名，例如 session_id_x 等）
    for cand in candidates:
        key = cand.split(".")[-1].lower()
        for col in columns:
            if key in col.lower():
                return col

    return None


def read_input_file(path: str) -> pd.DataFrame:
    """
    读取输入文件，支持 CSV / Excel。

    返回原始 DataFrame（暂不清洗）。
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in [".csv", ".txt"]:
        # 使用 utf-8-sig，可以兼容 BOM
        df = pd.read_csv(path, encoding="utf-8-sig")
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"不支持的文件类型：{ext} ，请使用 CSV 或 Excel。")

    return df


def basic_cleaning(df: pd.DataFrame, session_col: str) -> pd.DataFrame:
    """
    对原始数据进行基础清洗：
    1. 丢弃 sessionid 为空的行。
    2. 去除字符串字段首尾空格。
    3. 简单处理 Excel 导出导致的换行符（替换为单个空格）。
    """
    # 1. 丢弃 sessionid 为空的行
    before = len(df)
    df = df.dropna(subset=[session_col])
    # 有些 sessionid 可能是空字符串或全是空白
    df = df[df[session_col].astype(str).str.strip() != ""]
    after = len(df)

    print(f"清洗：丢弃 sessionid 为空的脏数据行数：{before - after}")

    # 2 & 3. 处理所有 object 类型列
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        # 统一转换为字符串，避免 None/NaN 直接参与字符串操作时出问题
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\r\n]+", " ", regex=True)
            .str.strip()
        )

    return df


def parse_tools_usage(
    df: pd.DataFrame, tools_col: Optional[str]
) -> Tuple[int, Dict[str, int]]:
    """
    统计 tools 使用情况。

    返回：
        tools_rows_count: 含有 <tools> 标签的行数
        tools_counter: {tool_name: 调用次数}
    """
    if tools_col is None or tools_col not in df.columns:
        print("警告：未找到 toolsmodelanswer 字段，无法统计工具调用情况。")
        return 0, {}

    series = df[tools_col].fillna("").astype(str)

    # 含有 <tools> 标签的行
    tools_pattern = re.compile(r"<\s*tools\b", flags=re.IGNORECASE)
    has_tools_mask = series.str.contains(tools_pattern)
    tools_rows_count = int(has_tools_mask.sum())

    # 提取 <tools>...</tools> 内的内容并统计工具名
    # 假设工具名形如 get_baoYang_product_info（函数式命名）
    tag_content_pattern = re.compile(
        r"<\s*tools[^>]*>(.*?)<\s*/\s*tools\s*>", flags=re.IGNORECASE | re.DOTALL
    )
    tool_name_pattern = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")

    tools_counter: Dict[str, int] = {}

    for text in series[has_tools_mask]:
        # 可能存在多个 <tools> 标签
        for match in tag_content_pattern.finditer(text):
            inner = match.group(1)
            for name in tool_name_pattern.findall(inner):
                # 过滤掉过于通用的词（如 tools、result 等），简单规则避免噪音
                lower = name.lower()
                if lower in {"tools", "result", "response", "data"}:
                    continue
                tools_counter[lower] = tools_counter.get(lower, 0) + 1

    return tools_rows_count, tools_counter


def group_sessions(
    df: pd.DataFrame,
    session_col: str,
    user_col: Optional[str],
) -> pd.DataFrame:
    """
    按 userid + sessionid 对多行对话进行聚合，并在每个会话内按时间排序。

    为方便后续里程碑，这里返回一个新的 DataFrame：
        - user_id
        - session_id
        - rows: 原始行在本会话中的顺序索引列表（或整个子 DataFrame 的 JSON 串）

    当前 Milestone 仅需要统计总会话数量，因此只要保证 groupby 正确即可。
    """
    group_keys = [session_col]
    if user_col is not None and user_col in df.columns:
        group_keys.insert(0, user_col)
        user_key_name = "user_id"
    else:
        user_key_name = "user_id"

    # 尝试检测时间字段，用于会话内排序
    time_col: Optional[str] = None
    detected = detect_column(list(df.columns), POSSIBLE_TIME_COLUMNS)
    if detected is not None:
        time_col = detected
        # 转换为 datetime，无法解析的设为 NaT
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # 建一个排序键：优先时间，其次原始行号
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_index"})
    if time_col is not None:
        df = df.sort_values(by=[time_col, "_orig_index"], kind="mergesort")
    else:
        df = df.sort_values(by=["_orig_index"], kind="mergesort")

    grouped = df.groupby(group_keys, sort=False)

    # 构造一个精简版的聚合 DataFrame，方便后续 Milestone 直接复用
    records = []
    for keys, sub in grouped:
        if isinstance(keys, tuple):
            if len(keys) == 2:
                uid, sid = keys
            else:
                # 理论上不会到这里，这里只是容错
                uid, sid = None, keys[-1]
        else:
            uid, sid = None, keys

        records.append(
            {
                user_key_name: uid,
                "session_id": sid,
                "row_indices": sub["_orig_index"].tolist(),
            }
        )

    session_df = pd.DataFrame.from_records(records)
    return session_df


def print_task1_summary(
    total_rows: int,
    tools_rows_count: int,
    tools_counter: Dict[str, int],
    session_count: int,
) -> None:
    """
    在终端打印 Task 1 的统计结果以及聚合后的总会话数量。
    """
    print("=" * 80)
    print("Task 1 定量统计结果")
    print("=" * 80)
    print(f"总数据行数：{total_rows}")
    print(f"含 <tools> 标签的数据行数：{tools_rows_count}")
    print("")
    print("各具体工具调用次数（按调用次数从高到低排序）：")

    if not tools_counter:
        print("  未检测到任何工具调用。")
    else:
        sorted_items = sorted(
            tools_counter.items(), key=lambda x: x[1], reverse=True
        )
        for name, count in sorted_items:
            print(f"  {name}: {count}")

    print("")
    print(f"聚合后的总会话（Session）数量：{session_count}")
    print("=" * 80)


def main() -> None:
    """
    主函数：解析命令行参数，执行读取、清洗、统计和聚合逻辑。
    """
    parser = argparse.ArgumentParser(
        description="途虎智能客服对话审查 Agent - Milestone 1 (本地数据读取与 Task 1 定量统计)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入文件路径（支持 .csv / .xls / .xlsx），例如：智能客服对话0316test.xlsx",
    )

    args = parser.parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    print(f"开始读取文件：{input_path}")
    df_raw = read_input_file(input_path)
    print(f"读取完成，原始行数：{len(df_raw)}，列数：{len(df_raw.columns)}")

    # 检测关键字段
    columns = list(df_raw.columns)
    session_col = detect_column(columns, POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError(
            "无法在表头中识别 sessionid 字段，请检查表结构或在代码中补充字段映射。"
        )

    user_col = detect_column(columns, POSSIBLE_USER_ID_COLUMNS)
    tools_col = detect_column(columns, POSSIBLE_TOOLS_ANSWER_COLUMNS)

    print(f"检测到 sessionid 字段名：{session_col}")
    if user_col:
        print(f"检测到 userid 字段名：{user_col}")
    else:
        print("警告：未检测到 userid 字段，将仅按 sessionid 聚合会话。")

    if tools_col:
        print(f"检测到 toolsmodelanswer 字段名：{tools_col}")
    else:
        print("警告：未检测到 toolsmodelanswer 字段，无法统计工具调用情况。")

    # 基础清洗
    df = basic_cleaning(df_raw, session_col=session_col)

    # Task 1 统计
    total_rows = len(df)
    tools_rows_count, tools_counter = parse_tools_usage(df, tools_col)

    # 会话聚合
    session_df = group_sessions(df, session_col=session_col, user_col=user_col)
    session_count = len(session_df)

    # 打印统计结果
    print_task1_summary(
        total_rows=total_rows,
        tools_rows_count=tools_rows_count,
        tools_counter=tools_counter,
        session_count=session_count,
    )


if __name__ == "__main__":
    main()

