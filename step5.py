#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5.py

里程碑 Milestone 5：本地 Prompt 知识库动态加载（Text Reader）。

V2.0 增量开发要求：
1. 在 V1.0 基座上新增“按需动态挂载 Prompt 文档”的能力，绝不修改或破坏已有里程碑逻辑。
2. 实现独立函数 load_prompt_docs(tool_names)，根据工具名集合按需加载本地文档：
   - 默认始终加载 人设2*.txt / 人设2*.md（回答人设）。
   - 仅针对当前会话中实际出现的工具名，加载对应的 Tool 文档（例如 get_baoyang_product_info.txt / .md）。
3. 在提取单条 Session 时，打印本次实际拼装给 LLM 的“动态 Prompt 字符串”的长度，
   并列出被加载的文档清单，用于人工验证“按需加载、未加载无关文档”。

注意：
1. 本脚本不调用大模型，只做本地文本拼装与调试打印。
2. 后续 Milestone 6/7 会在此基础上接入新的 JSON 结构与全局反思逻辑。
"""

import argparse
import os
import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd

import step1


# ===========================
# 工具解析相关配置
# ===========================

TOOLS_TAG_PATTERN = re.compile(
    r"<\s*tools[^>]*>(.*?)<\s*/\s*tools\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)

TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def extract_tools_from_texts(texts: Iterable[str]) -> Set[str]:
    """
    从多条 toolsmodelanswer 文本中解析出出现过的工具名集合。

    解析规则：
    1. 先提取 <tools>...</tools> 之间的内容；
    2. 再在内部用 TOOL_NAME_PATTERN 抽取可能的函数式工具名；
    3. 过滤掉通用噪音单词（如 tools / result / data 等）。
    """
    tools: Set[str] = set()
    for text in texts:
        if not text:
            continue
        # 找到所有 <tools>...</tools>
        for m in TOOLS_TAG_PATTERN.finditer(str(text)):
            inner = m.group(1)
            for name in TOOL_NAME_PATTERN.findall(inner):
                lower = name.lower()
                if lower in {"tools", "result", "response", "data"}:
                    continue
                tools.add(lower)
    return tools


def load_prompt_docs(
    tool_names: Iterable[str],
    base_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    按需动态加载本地 Prompt 文档。

    参数：
        tool_names: 当前会话中出现的工具名集合（小写）。
        base_dir:   文档所在目录，默认使用当前脚本所在目录。

    文档加载规则：
    1. 人设文档：
       - 自动搜索 人设2*.txt / 人设2*.md，全部加载（默认回答人设）。
    2. Tool 文档：
       - 对于每个工具名 t，优先匹配同名文件：
         - {t}.txt, {t}.md
       - 也允许一定程度的前缀匹配（如 get_baoyang_product_info_xxx.txt）。

    返回：
        {文件名: 文件内容字符串}
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))

    tool_set = {str(t).lower() for t in tool_names if t}
    docs: Dict[str, str] = {}

    try:
        file_list = os.listdir(base_dir)
    except OSError as exc:
        print(f"警告：无法列举目录 {base_dir} 下的文件，错误：{exc}")
        return docs

    # 1. 人设2 文档（回答人设，始终加载）
    for fname in file_list:
        lower = fname.lower()
        # 兼容 txt / md
        if not (lower.endswith(".txt") or lower.endswith(".md")):
            continue
        # 前缀匹配“人设2”
        if fname.startswith("人设2"):
            path = os.path.join(base_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs[fname] = f.read()
            except Exception as exc:
                print(f"警告：读取人设文档 {fname} 失败：{exc}")

    # 2. Tool 文档（仅加载当前会话实际用到的工具）
    #    先建立一个从“无扩展小写文件名”到完整文件名的索引，便于匹配
    name_index: Dict[str, List[str]] = {}
    for fname in file_list:
        lower = fname.lower()
        if not (lower.endswith(".txt") or lower.endswith(".md")):
            continue
        stem = os.path.splitext(lower)[0]  # 去掉扩展名
        name_index.setdefault(stem, []).append(fname)

    for tool in tool_set:
        # 精确同名匹配：{tool}.txt / {tool}.md
        direct_key = tool
        matched_files: List[str] = []

        if direct_key in name_index:
            matched_files.extend(name_index[direct_key])
        else:
            # 容错：允许文件名以工具名为前缀，例如 get_baoyang_product_info_xxx.txt
            for stem, fnames in name_index.items():
                if stem.startswith(tool):
                    matched_files.extend(fnames)

        for fname in matched_files:
            if fname in docs:
                continue  # 已加载过（避免重复）
            path = os.path.join(base_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs[fname] = f.read()
            except Exception as exc:
                print(f"警告：读取 Tool 文档 {fname} 失败：{exc}")

    return docs


def build_dynamic_prompt_string(
    persona_and_tool_docs: Dict[str, str],
) -> str:
    """
    根据已加载的人设文档与 Tool 文档，拼装一个准备喂给 LLM 的“知识库段落”字符串。

    注意：这里只是示例拼装，后续 Milestone 6 会在此基础上再叠加会话内容与更完整的 System Prompt。
    """
    parts: List[str] = []
    # 简单按照文件名排序，保证输出稳定
    for fname in sorted(persona_and_tool_docs.keys()):
        content = persona_and_tool_docs[fname]
        parts.append(f"【文档：{fname}】\n{content}\n")
    return "\n".join(parts)


def prepare_demo_sessions(
    df: pd.DataFrame,
    session_df: pd.DataFrame,
    tools_col: Optional[str],
    max_sessions: int = 3,
) -> List[Tuple[Dict, Set[str]]]:
    """
    为 Milestone 5 准备若干用于演示的 Session：
        - 从 session_df 中取前 max_sessions 个会话；
        - 从原始 df 中抽取其 toolsmodelanswer 文本；
        - 解析得到当前 Session 中出现过的工具名集合。

    返回列表：
        [
          (session_record_dict, tool_name_set),
          ...
        ]
    """
    if tools_col is None or tools_col not in df.columns:
        print("警告：未检测到 toolsmodelanswer 字段，无法演示基于工具名的动态挂载。")
        return []

    n = min(max_sessions, len(session_df))
    demo: List[Tuple[Dict, Set[str]]] = []

    for idx in range(n):
        session_record = session_df.iloc[idx].to_dict()
        row_indices = session_record.get("row_indices") or []
        if not row_indices:
            continue

        # 从原始 df 中取出该 Session 内所有 toolsmodelanswer 文本
        sub = df.loc[row_indices]
        texts = sub[tools_col].fillna("").astype(str).tolist()
        tool_set = extract_tools_from_texts(texts)
        demo.append((session_record, tool_set))

    return demo


def main() -> None:
    """
    命令行入口：
    - 读取输入文件；
    - 聚合 Session；
    - 针对前若干个 Session：
        - 解析工具名 -> 动态加载对应 Prompt 文档；
        - 拼装知识库段落字符串；
        - 在终端打印：Session 基本信息、工具名列表、实际加载的文档列表、拼装字符串长度。
    """
    parser = argparse.ArgumentParser(
        description="途虎智能客服对话审查 Agent - Milestone 5 (Prompt 知识库动态挂载调试)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="原始对话数据文件路径（支持 .csv / .xls / .xlsx）。",
    )
    parser.add_argument(
        "--sessions",
        "-n",
        type=int,
        default=3,
        help="用于演示动态挂载逻辑的 Session 数量，默认 3 个。",
    )

    args = parser.parse_args()
    input_path = args.input

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    print(f"开始读取文件：{input_path}")
    df_raw = step1.read_input_file(input_path)
    print(f"读取完成，原始行数：{len(df_raw)}，列数：{len(df_raw.columns)}")

    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")

    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    tools_col = step1.detect_column(columns, step1.POSSIBLE_TOOLS_ANSWER_COLUMNS)

    print(f"检测到 sessionid 字段名：{session_col}")
    if user_col:
        print(f"检测到 userid 字段名：{user_col}")
    else:
        print("警告：未检测到 userid 字段，将仅按 sessionid 聚合会话。")

    if tools_col:
        print(f"检测到 toolsmodelanswer 字段名：{tools_col}")
    else:
        print("警告：未检测到 toolsmodelanswer 字段，本次无法根据工具名挂载 Tool Prompt 文档。")

    # 使用与 V1.0 相同的清洗与聚合逻辑，保证数据一致性
    df = step1.basic_cleaning(df_raw, session_col=session_col)
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    print(f"聚合完成，总会话数量：{len(session_df)}")

    if len(session_df) == 0:
        print("没有可用的会话数据，程序结束。")
        return

    # 准备若干演示 Session
    demos = prepare_demo_sessions(
        df=df,
        session_df=session_df,
        tools_col=tools_col,
        max_sessions=args.sessions,
    )

    if not demos:
        print("未能准备出可用于演示的 Session（可能缺少 tools 字段），程序结束。")
        return

    print("\n================ 动态 Prompt 挂载演示 ================")
    for idx, (session_record, tool_set) in enumerate(demos, start=1):
        user_id = session_record.get("user_id")
        session_id = session_record.get("session_id")

        print("-" * 80)
        print(f"Demo Session {idx}")
        print(f"用户ID: {user_id}    SessionID: {session_id}")
        print(f"该 Session 内解析到的工具名集合：{sorted(tool_set) if tool_set else '（无工具调用）'}")

        docs = load_prompt_docs(tool_set)
        loaded_filenames = sorted(docs.keys())
        print(f"本次实际加载的 Prompt 文档文件：{loaded_filenames if loaded_filenames else '（无文档被加载）'}")

        prompt_string = build_dynamic_prompt_string(docs)
        print(f"动态拼装的 Prompt 知识库字符串长度（字符数）：{len(prompt_string)}")

        # 为方便人工抽查，可在需要时打印前若干字符示意（默认不打印太长内容）
        preview_len = min(200, len(prompt_string))
        if preview_len > 0:
            print("Prompt 内容前 200 字预览：")
            print(prompt_string[:preview_len].replace("\n", "\\n"))

    print("\nMilestone 5 演示结束，请根据上述输出检查：")
    print("1）每个 Session 只加载了 人设2* 和自身涉及到的 Tool 文档；")
    print("2）未出现与当前 Session 无关的 Tool 文档；")
    print("3）Prompt 字符串长度在可控范围内，为后续 Map 阶段打下基础。")


if __name__ == "__main__":
    main()

