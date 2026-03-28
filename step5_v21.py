#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step5_v21.py

V2.1 Milestone 5：Excel 字典映射与 Docx 动态加载模块（Text Reader）。

目标：
1. 读取「意图描述和对应专家策略.xlsx」并构建映射字典（特征/意图 -> 专家策略英文前缀）。
2. 使用 python-docx 读取 人设1、人设2 以及专家策略 .docx 文档的纯文本。
3. 对每个 Session：从 toolsmodelanswer 中解析工具名/触发特征，并结合映射表推断应该挂载哪个专家策略 docx。
4. 组装 build_context_prompt(session_data)：当前 Session 完整历史 + 人设1文档 + 人设2文档 + 匹配到的 1 份专家策略 docx。

本脚本只做「动态挂载与 Prompt 拼装调试」，不调用大模型。

验收输出（按 PRD）：
- 终端打印：
  “当前行触发了 [xxx] 意图，已成功动态挂载 [xxx_专家策略.docx]”
  并打印组装后的 Prompt 字符总长度。
"""

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
from docx import Document

import step1


# ===========================
# 工具解析：从 toolsmodelanswer 文本里抽取工具名
# ===========================

TOOLS_TAG_PATTERN = re.compile(
    r"<\s*tools[^>]*>(.*?)<\s*/\s*tools\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
TOOL_NAME_PATTERN = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def extract_tool_names_from_toolsmodelanswer(text: str) -> Set[str]:
    """
    从 toolsmodelanswer 中提取 <tools>...</tools> 内出现的工具名。
    """
    if not text:
        return set()
    text = str(text)
    tool_names: Set[str] = set()

    for m in TOOLS_TAG_PATTERN.finditer(text):
        inner = m.group(1)
        for name in TOOL_NAME_PATTERN.findall(inner):
            lower = name.lower()
            # 过滤明显噪音
            if lower in {"tools", "result", "response", "data"}:
                continue
            tool_names.add(lower)

    return tool_names


def normalize_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ===========================
# Word Reader
# ===========================


def read_docx_plain_text(docx_path: str) -> str:
    """
    读取 .docx 并抽取纯文本（按段落拼接）。
    """
    document = Document(docx_path)
    paras: List[str] = []
    for p in document.paragraphs:
        t = (p.text or "").strip()
        if t:
            paras.append(t)
    return "\n".join(paras).strip()


def find_text_file_by_glob_patterns(base_dir: str, patterns: Sequence[str]) -> List[str]:
    """
    用简单正则/前缀匹配在目录中查找候选 txt/md 文件（不依赖 glob 模块避免过度复杂）。
    patterns 允许写前缀：如 "人设1" 会匹配 "人设1*.txt/md"。
    """
    candidates: List[str] = []
    try:
        names = os.listdir(base_dir)
    except OSError:
        return candidates

    for name in names:
        lower = name.lower()
        if not (lower.endswith(".txt") or lower.endswith(".md")):
            continue
        for pat in patterns:
            if pat and name.startswith(pat):
                candidates.append(os.path.join(base_dir, name))
                break
            # 允许 pat 只给关键字
            if pat and pat.lower() in lower:
                candidates.append(os.path.join(base_dir, name))
                break
    # 去重并排序
    return sorted(list(dict.fromkeys(candidates)))


def find_docx_file_by_keyword(base_dir: str, keyword: str) -> List[str]:
    """
    在目录下通过 keyword 进行 .docx 模糊匹配返回候选列表。
    """
    candidates: List[str] = []
    try:
        names = os.listdir(base_dir)
    except OSError:
        return candidates

    key = (keyword or "").lower()
    if not key:
        return candidates

    for name in names:
        lower = name.lower()
        if lower.endswith(".docx") and key in lower:
            candidates.append(os.path.join(base_dir, name))

    return sorted(list(dict.fromkeys(candidates)))


def read_text_plain_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_doc_or_text_plain_text(path: str) -> str:
    """
    读取 .docx 或 .txt/.md 的纯文本：
    - .docx：使用 python-docx 提取段落文本
    - 其他：使用 utf-8 读取整文件
    """
    lower = path.lower()
    if lower.endswith(".docx"):
        return read_docx_plain_text(path)
    return read_text_plain_text(path)


# ===========================
# Excel 映射字典
# ===========================


@dataclass
class MappingRow:
    intent_desc: str
    strategy_prefix: str


def infer_mapping_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    尝试从 Excel 列名里推断 “意图描述” 与 “对应专家策略英文前缀” 两列。
    """
    cols = [str(c) for c in df.columns]
    # 候选：意图列
    intent_candidates = [c for c in cols if ("意图" in c) or ("intent" in c.lower()) or ("描述" in c)]
    # 候选：策略列
    strategy_candidates = [
        c for c in cols if ("策略" in c) or ("strategy" in c.lower()) or ("对应" in c) or ("英文" in c.lower())
    ]

    if intent_candidates and strategy_candidates:
        return intent_candidates[0], strategy_candidates[0]

    # 兜底：使用前两列
    if len(cols) < 2:
        raise ValueError("映射 Excel 至少需要两列：意图描述 和 对应专家策略。")
    return cols[0], cols[1]


def load_intent_strategy_mapping(
    mapping_xlsx_path: str,
) -> List[MappingRow]:
    """
    读取「意图描述和对应专家策略.xlsx」，返回映射行列表。
    """
    if not os.path.exists(mapping_xlsx_path):
        raise FileNotFoundError(f"未找到映射文件：{mapping_xlsx_path}")

    df = pd.read_excel(mapping_xlsx_path)
    intent_col, strategy_col = infer_mapping_columns(df)
    records: List[MappingRow] = []

    for _, row in df.iterrows():
        intent_desc = normalize_text(row.get(intent_col, ""))
        strategy_prefix = normalize_text(row.get(strategy_col, ""))
        if not intent_desc or not strategy_prefix:
            continue
        records.append(MappingRow(intent_desc=intent_desc, strategy_prefix=strategy_prefix))

    if not records:
        raise ValueError("映射表加载后为空，请检查 Excel 内容/列名。")

    return records


# ===========================
# 模糊寻址：基于 strategy_prefix 找 .docx
# ===========================


def find_docx_by_fuzzy_prefix(
    strategy_prefix: str,
    strategies_dir: str,
) -> Optional[str]:
    """
    在 strategies_dir 内通过模糊匹配寻找专家策略 docx。

    先做“包含匹配”；如果失败，则降级为 token 级匹配（忽略常见后缀如 tool/prompt 等）。
    """
    if not strategy_prefix:
        return None
    if not os.path.exists(strategies_dir):
        return None

    prefix_lower = str(strategy_prefix).lower()
    docx_names: List[str] = []
    try:
        docx_names = [n for n in os.listdir(strategies_dir) if n.lower().endswith(".docx")]
    except OSError:
        return None

    # 1) 精确包含匹配（最高优先级）
    contains_matches: List[str] = []
    for name in docx_names:
        if prefix_lower in name.lower():
            contains_matches.append(os.path.join(strategies_dir, name))
    if contains_matches:
        return sorted(list(dict.fromkeys(contains_matches)))[0]

    # 2) token 级匹配：忽略常见噪音后缀（如 tool/prompt/strategy/expert）
    stopwords = {"tool", "tools", "prompt", "strategy", "expert", "experts"}

    def tokenize_alnum_underscore(s: str) -> Set[str]:
        s = (s or "").lower()
        # 仅保留英数/下划线，其余替换为空分隔符
        cleaned = re.sub(r"[^a-z0-9_]+", "_", s)
        tokens = [t for t in cleaned.split("_") if t]
        return {t for t in tokens if t not in stopwords}

    prefix_tokens = tokenize_alnum_underscore(prefix_lower)
    if not prefix_tokens:
        return None

    best_path: Optional[str] = None
    best_score = -1

    for name in docx_names:
        file_tokens = tokenize_alnum_underscore(name.lower())
        score = len(prefix_tokens & file_tokens)
        if score > best_score:
            best_score = score
            best_path = os.path.join(strategies_dir, name)

    # 至少要匹配到 1 个 token，避免误配
    if best_score >= 1:
        return best_path

    return None


# ===========================
# 意图/策略英文前缀推断（V2.1 关键：toolsmodelanswer -> 映射表关联）
# ===========================


def score_mapping_match(
    mapping: MappingRow,
    tool_names: Set[str],
    search_text: str,
) -> int:
    """
    为某条 mapping 行计算匹配得分。

    由于我们在本阶段没有看到你完整的“意图描述”字段格式，
    这里采用可泛化的启发式策略：
    1) 如果 strategy_prefix 本身出现在 toolsmodelanswer/上下文中 => 强命中
    2) 如果 mapping intent_desc 中提取到的英数 token 与 tool_names 有交集 => 加分
    3) 如果 intent_desc 的短关键词能在 search_text 中做子串命中 => 加分
    """
    score = 0
    st = search_text.lower()

    sp = (mapping.strategy_prefix or "").lower()
    if sp and sp in st:
        score += 1000

    # intent_desc 中提取英数 token（若意图描述里包含工具名/英文前缀，则可命中）
    tokens = TOOL_NAME_PATTERN.findall(mapping.intent_desc or "")
    tokens_lower = [t.lower() for t in tokens if t]
    if tokens_lower:
        inter = set(tokens_lower) & set(tool_names)
        score += 100 * len(inter)

    # 再做中文短意图的子串命中（长度太长会误伤，因此做限制）
    intent_desc_norm = normalize_text(mapping.intent_desc or "")
    if intent_desc_norm:
        # 只取前 12 个字符做子串命中（避免过长导致噪音）
        snippet = intent_desc_norm[:12].strip()
        if snippet and len(snippet) >= 3 and snippet in search_text:
            score += 30

    return score


def infer_strategy_from_session_line(
    tool_names: Set[str],
    search_text: str,
    mapping_rows: List[MappingRow],
) -> Tuple[Optional[MappingRow], int]:
    """
    基于 toolsmodelanswer/上下文推断应该挂载的 mapping 行。
    返回：
      - best_mapping_row 或 None
      - best_score
    """
    best: Optional[MappingRow] = None
    best_score = -1

    for m in mapping_rows:
        s = score_mapping_match(m, tool_names=tool_names, search_text=search_text)
        if s > best_score:
            best = m
            best_score = s

    return best, best_score


# ===========================
# Prompt 构建：build_context_prompt
# ===========================


def build_session_history_text(
    conversation_rows: List[Dict[str, Optional[str]]],
) -> str:
    """
    把一个 Session 的多轮对话拼成纯文本历史，供大模型使用。
    """
    parts: List[str] = []
    for i, row in enumerate(conversation_rows, start=1):
        user_q = normalize_text(row.get("dialoguesentence") or "")
        rewrite_q = normalize_text(row.get("rewritequery") or "")
        answer = normalize_text(row.get("answer") or "")
        parts.append(f"[轮次 {i}]")
        parts.append(f"用户问题：{user_q}")
        parts.append(f"重写问题：{rewrite_q}")
        parts.append(f"客服回答：{answer}")
    return "\n".join(parts).strip()


def build_context_prompt(
    session_history_text: str,
    persona1_text: str,
    persona2_text: str,
    expert_strategy_text: str,
) -> str:
    """
    严格按 PRD 组装：
    当前 Session 完整历史 + 人设1文档 + 人设2文档 + 匹配到的 1 份专家策略.docx内容。
    """
    return (
        "【当前 Session 完整历史】\n"
        + session_history_text
        + "\n\n【人设1 Router（意图识别与工具推荐）】\n"
        + persona1_text
        + "\n\n【人设2 Answerer（最终回复生成）】\n"
        + persona2_text
        + "\n\n【匹配到的专家策略（Expert Strategy）】\n"
        + expert_strategy_text
    )


def extract_conversation_rows_for_session(
    df,
    row_indices: List[int],
    dialogue_col: Optional[str],
    rewrite_col: Optional[str],
    answer_col: Optional[str],
) -> List[Dict[str, Optional[str]]]:
    """
    从原始 DataFrame 抽取 session 对话记录，按原顺序返回。
    """
    sub = df.loc[row_indices]
    rows: List[Dict[str, Optional[str]]] = []

    # sub 的索引顺序可能并非严格 0..n，因此按子表原顺序 iterrows
    for _, row in sub.iterrows():
        rows.append(
            {
                "dialoguesentence": (
                    row[dialogue_col] if dialogue_col and dialogue_col in sub.columns else None
                ),
                "rewritequery": (
                    row[rewrite_col] if rewrite_col and rewrite_col in sub.columns else None
                ),
                "answer": (
                    row[answer_col] if answer_col and answer_col in sub.columns else None
                ),
            }
        )
    return rows


# ===========================
# Milestone 5：主调试流程（只跑 1 条测试数据）
# ===========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2.1 Milestone 5 - Excel 映射与 Docx 动态加载调试"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="原始对话数据文件路径（支持 .csv / .xls / .xlsx）。",
    )
    parser.add_argument(
        "--mapping",
        "-m",
        default="意图描述和对应专家策略.xlsx",
        help="意图描述和对应专家策略.xlsx 路径，默认当前目录下的同名文件。",
    )
    parser.add_argument(
        "--strategies-dir",
        "-s",
        default=".",
        help="专家策略 docx 文件所在目录（默认当前目录）。",
    )
    parser.add_argument(
        "--persona1",
        default="人设1意图识别与判断使用哪些工具.docx",
        help="人设1文档路径（默认：人设1意图识别与判断使用哪些工具.docx），如果找不到将尝试模糊匹配人设1*.docx。",
    )
    parser.add_argument(
        "--persona2",
        default="人设2 根据工具返回的结果和常识回答问题.docx",
        help="人设2文档路径（默认：人设2 根据工具返回的结果和常识回答问题.docx），如果找不到将尝试模糊匹配人设2*.docx。",
    )
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 1) 加载映射表
    mapping_rows = load_intent_strategy_mapping(args.mapping)
    print(f"已加载映射表：{len(mapping_rows)} 条映射记录")

    # 2) 读取 persona1/persona2
    base_dir = os.path.dirname(os.path.abspath(__file__))

    persona1_path = args.persona1
    if not os.path.exists(persona1_path):
        candidates = find_docx_file_by_keyword(base_dir, "人设1")
        persona1_path = candidates[0] if candidates else ""
    if not persona1_path or not os.path.exists(persona1_path):
        raise FileNotFoundError("无法找到 人设1 文档，请提供 --persona1 或确保目录下存在 人设1*.docx。")
    persona1_text = read_doc_or_text_plain_text(persona1_path)

    persona2_path = args.persona2
    if not os.path.exists(persona2_path):
        candidates = find_docx_file_by_keyword(base_dir, "人设2")
        persona2_path = candidates[0] if candidates else ""
    if not persona2_path or not os.path.exists(persona2_path):
        raise FileNotFoundError("无法找到 人设2 文档，请提供 --persona2 或确保目录下存在 人设2*.docx。")
    persona2_text = read_doc_or_text_plain_text(persona2_path)

    # 3) 读取数据并聚合 session（复用 milestone 1 的同一套清洗/字段识别）
    print(f"开始读取文件：{input_path}")
    df_raw = step1.read_input_file(input_path)

    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    tools_col = step1.detect_column(columns, step1.POSSIBLE_TOOLS_ANSWER_COLUMNS)
    dialogue_col = step1.detect_column(columns, step1.POSSIBLE_DIALOGUE_SENTENCE_COLUMNS)
    rewrite_col = step1.detect_column(columns, step1.POSSIBLE_REWRITE_QUERY_COLUMNS)
    answer_col = step1.detect_column(columns, step1.POSSIBLE_ANSWER_COLUMNS)

    if not tools_col:
        raise ValueError("无法识别 toolsmodelanswer 字段，请检查表结构。")

    print(f"检测到 toolsmodelanswer 字段名：{tools_col}")

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    if len(session_df) == 0:
        print("没有可用 session 数据，程序结束。")
        return

    # 4) 取 1 个测试 Session + 其中 1 条“当前行”（用于推断意图/策略）
    test_session_record = session_df.iloc[0].to_dict()
    row_indices = test_session_record.get("row_indices") or []
    if not row_indices:
        raise ValueError("测试 Session 内没有 row_indices。")

    # 当前行：优先取 toolsmodelanswer 非空的那一行
    sub = df.loc[row_indices]
    current_row_idx: Optional[int] = None
    current_tools_text: str = ""
    for idx, r in sub.iterrows():
        t = normalize_text(r.get(tools_col, ""))
        if t:
            current_row_idx = idx
            current_tools_text = t
            break
    if current_row_idx is None:
        # 兜底：取第一行
        current_row_idx = sub.index[0]
        current_tools_text = normalize_text(sub.iloc[0].get(tools_col, ""))

    # 5) 解析 toolsmodelanswer 触发特征
    tool_names = extract_tool_names_from_toolsmodelanswer(current_tools_text)

    # 组装 search_text：包含当前行 toolsmodelanswer，以及该 session 当前轮次可能的上下文
    current_dialogue = ""
    current_rewrite = ""
    current_answer = ""
    try:
        cur_row = df.loc[current_row_idx]
        if dialogue_col and dialogue_col in df.columns:
            current_dialogue = normalize_text(cur_row.get(dialogue_col, ""))
        if rewrite_col and rewrite_col in df.columns:
            current_rewrite = normalize_text(cur_row.get(rewrite_col, ""))
        if answer_col and answer_col in df.columns:
            current_answer = normalize_text(cur_row.get(answer_col, ""))
    except Exception:
        # 容错：即使取不到也不影响继续推断
        pass

    search_text = "\n".join(
        [
            normalize_text(current_tools_text),
            current_dialogue,
            current_rewrite,
            current_answer,
        ]
    )

    # 6) 用启发式打分推断 mapping 行（intent_desc -> strategy_prefix）
    best_mapping, best_score = infer_strategy_from_session_line(
        tool_names=tool_names, search_text=search_text, mapping_rows=mapping_rows
    )

    if best_mapping is None:
        print("无法推断策略映射（best_mapping 为 None），程序结束。")
        return

    # 7) 模糊寻址：根据 strategy_prefix 找对应 .docx
    expert_docx_path = find_docx_by_fuzzy_prefix(
        strategy_prefix=best_mapping.strategy_prefix,
        strategies_dir=args.strategies_dir,
    )
    if not expert_docx_path:
        raise FileNotFoundError(
            f"未在目录 {args.strategies_dir} 中找到匹配 strategy_prefix={best_mapping.strategy_prefix} 的 .docx 文件。"
        )

    expert_docx_filename = os.path.basename(expert_docx_path)

    # 8) 构建完整 session prompt（用于后续 Milestone 6 调用大模型）
    conversation_rows = extract_conversation_rows_for_session(
        df=df,
        row_indices=row_indices,
        dialogue_col=dialogue_col,
        rewrite_col=rewrite_col,
        answer_col=answer_col,
    )

    session_history_text = build_session_history_text(conversation_rows)
    expert_strategy_text = read_docx_plain_text(expert_docx_path)
    context_prompt = build_context_prompt(
        session_history_text=session_history_text,
        persona1_text=persona1_text,
        persona2_text=persona2_text,
        expert_strategy_text=expert_strategy_text,
    )

    # 9) 交付验收输出
    user_id = test_session_record.get("user_id")
    session_id = test_session_record.get("session_id")

    print("\n================ Milestone 5 验收输出 ================")
    print(f"测试 Session：user_id={user_id}，session_id={session_id}")
    print(f"当前行解析到工具名集合：{sorted(tool_names)}")
    print(
        f"当前行触发了 [{best_mapping.intent_desc}] 意图，已成功动态挂载 [{expert_docx_filename}]"
    )
    print(f"匹配得分：{best_score}")
    print(f"组装后的 Prompt 字符总长度：{len(context_prompt)}")
    print("=======================================================\n")

    # 为了避免提示过长，不额外打印 context_prompt 内容。


if __name__ == "__main__":
    main()

