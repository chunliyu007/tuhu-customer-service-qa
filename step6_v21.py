#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step6_v21.py

里程碑 Milestone 6：微观审查与二级分类输出（Map 阶段，前 5 个 Session）。

功能：
1. 读取离线数据并聚合 Session（复用 step1）
2. 对前 N 个 Session（默认 5）逐条调用大模型审查
3. 动态挂载：根据当前 Session 的 toolsmodelanswer 提取工具名 -> 查映射表 -> 选择 1 份专家策略 docx
4. Prompt 组装：当前 Session 完整历史 + 人设1 + 人设2 + 1 份专家策略内容
5. LLM 输出 JSON（严格按 V2.1 规范 3）
6. 终端打印这 N 个 Session 的 JSON 评价结果，便于你人工验收

说明：
- 本脚本不做并发（Milestone 6 以 Map 小批量为主）
- 错误容错：若 JSON 解析失败，会记录一份兜底结果并继续下一条
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import openai
import pandas as pd
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

import step1
import step2
import step5_v21


def compress_consecutive_same_dialogue_rows(
    df,
    session_col: str,
    dialogue_col: Optional[str],
    answer_col: Optional[str],
) :
    """
    在同一个 session 内，将“连续且用户问题相同”的多行压缩为一行。

    合并策略：
    1) 仅在同 session 且连续相同 dialoguesentence 时触发合并；
    2) answer 按时间顺序拼接为一段完整回复；
    3) 其他字段默认保留最后一行（最新）值。
    """
    # 没有对话字段或回答字段时，不做压缩
    if not dialogue_col or dialogue_col not in df.columns:
        return df
    if not answer_col or answer_col not in df.columns:
        return df

    # 先复制，避免影响外部 DataFrame 引用
    work_df = df.copy()
    work_df = work_df.reset_index(drop=False).rename(columns={"index": "_orig_index_for_merge"})

    # 按 session + 时间（如存在）+ 原始行号排序，确保“连续”定义稳定
    time_col = step1.detect_column(list(work_df.columns), step1.POSSIBLE_TIME_COLUMNS)
    if time_col and time_col in work_df.columns:
        work_df[time_col] = pd.to_datetime(work_df[time_col], errors="coerce")
        work_df = work_df.sort_values(
            by=[session_col, time_col, "_orig_index_for_merge"],
            kind="mergesort",
        )
    else:
        work_df = work_df.sort_values(
            by=[session_col, "_orig_index_for_merge"],
            kind="mergesort",
        )

    merged_records: List[Dict[str, Any]] = []

    # 逐 session 扫描，识别“连续相同问题”的块
    for _, sub in work_df.groupby(session_col, sort=False):
        sub = sub.reset_index(drop=True)
        i = 0
        while i < len(sub):
            cur_q = step5_v21.normalize_text(sub.at[i, dialogue_col])

            # 向后吃掉连续相同问题的行
            j = i + 1
            while j < len(sub):
                nxt_q = step5_v21.normalize_text(sub.at[j, dialogue_col])
                if nxt_q != cur_q:
                    break
                j += 1

            block = sub.iloc[i:j]

            # 以最后一行作为“主行”，保留最新辅助字段（tools/rewrite 等）
            last_row = block.iloc[-1].to_dict()

            # answer 按顺序拼接，过滤空串并去重“连续重复片段”
            answers: List[str] = []
            prev_ans = ""
            for _, r in block.iterrows():
                a = step5_v21.normalize_text(r.get(answer_col, ""))
                if not a:
                    continue
                if a == prev_ans:
                    continue
                answers.append(a)
                prev_ans = a
            merged_answer = "\n".join(answers).strip()
            last_row[answer_col] = merged_answer

            # dialoguesentence 明确保留块内问题
            last_row[dialogue_col] = cur_q

            merged_records.append(last_row)
            i = j

    if not merged_records:
        return df

    merged_df = pd.DataFrame.from_records(merged_records)

    # 清理临时列，恢复干净索引
    if "_orig_index_for_merge" in merged_df.columns:
        merged_df = merged_df.drop(columns=["_orig_index_for_merge"])
    merged_df = merged_df.reset_index(drop=True)
    return merged_df


def pick_preview_row_for_merged_session(
    conversation_rows: List[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """
    选择一个用于“合并数据验证”的代表行：
    优先选择 answer 含换行（说明已发生多段拼接）的行，否则返回第一行。
    """
    if not conversation_rows:
        return {"dialoguesentence": "", "answer": ""}

    for row in conversation_rows:
        ans = step5_v21.normalize_text(row.get("answer") or "")
        if "\n" in ans:
            return row
    return conversation_rows[0]


# ===========================
# JSON 解析与校验
# ===========================


def sanitize_llm_output(text: str) -> str:
    """
    将 LLM 输出尽量清洗成“单个 JSON 对象字符串”。
    """
    text = (text or "").strip()

    # 去掉 ```json ... ``` 外壳（如果模型输出了）
    code_block_pattern = re.compile(
        r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE
    )
    m = code_block_pattern.match(text)
    if m:
        text = m.group(1).strip()

    # 仅保留从第一个 { 到最后一个 } 的部分
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()
    return text


def count_chinese_chars(s: str) -> int:
    """
    统计中文汉字数量，用于 secondary_category <=4 的约束校验。
    """
    return len(re.findall(r"[\u4e00-\u9fff]", s or ""))


def normalize_bool(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv == "true":
            return True
        if vv == "false":
            return False
    return v


def validate_secondary_category(sec: Any) -> None:
    sec_str = str(sec or "")
    # 只做校验，不强行改写，便于你观察模型是否遵守约束
    if count_chinese_chars(sec_str) > 4:
        print(
            f"警告：secondary_category 中文字符数 > 4（当前='{sec_str}'，count={count_chinese_chars(sec_str)})"
        )


# ===========================
# Milestone 8：人审校准（导出/注入）
# ===========================


def truncate_text(s: str, max_chars: int) -> str:
    """
    防止人审案例注入过长导致系统提示词过大。
    """
    if not s:
        return ""
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def build_human_calibration_fewshot_text(
    human_tasks_xlsx_path: str,
    *,
    max_cases: int = 12,
    max_dialogue_chars: int = 900,
) -> str:
    """
    若存在 `human_calibration_tasks.xlsx` 且含有人工修正（人工修正判定非空），
    则把人类专家的尺度以 few-shot 形式追加到 system prompt。
    """
    if not os.path.exists(human_tasks_xlsx_path):
        return ""

    # 延迟导入：避免你只跑 Map 时因依赖缺失直接失败
    try:
        import pandas as pd
    except Exception as exc:
        print(f"警告：导入 pandas 失败，跳过人审注入：{type(exc).__name__}: {exc}")
        return ""

    try:
        df = pd.read_excel(human_tasks_xlsx_path)
    except Exception as exc:
        print(f"警告：读取人审表失败，跳过人审注入：{type(exc).__name__}: {exc}")
        return ""

    required_cols = [
        "SessionID",
        "对话历史 (User & Bot)",
        "LLM判定原因 (common_sense_reason)",
        "人工修正判定",
        "人工给出的真实原因",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        # 结构不对时不强行注入，避免污染评判
        print(f"警告：人审表缺少列 {missing_cols}，跳过人审注入。")
        return ""

    manual_col = df["人工修正判定"]
    manual_rows = df[manual_col.notna() & (manual_col.astype(str).str.strip() != "")]
    if len(manual_rows) == 0:
        return ""

    manual_rows = manual_rows.head(max_cases)

    blocks: List[str] = []
    for i in range(min(len(manual_rows), max_cases)):
        row = manual_rows.iloc[i]
        dialogue_hist = truncate_text(
            str(row.get("对话历史 (User & Bot)") or ""),
            max_chars=max_dialogue_chars,
        )
        manual_judge = str(row.get("人工修正判定") or "").strip()
        true_reason = truncate_text(
            str(row.get("人工给出的真实原因") or ""),
            max_chars=max_dialogue_chars,
        )
        llm_reason = truncate_text(
            str(row.get("LLM判定原因 (common_sense_reason)") or ""),
            max_chars=420,
        )

        blocks.append(
            "案例（人类专家校准）：\n"
            f"- 案例编号：{i + 1}\n"
            f"- LLM之前的common_sense_reason（仅供参考）：{llm_reason}\n"
            f"- 对话历史（User & Bot）：\n{dialogue_hist}\n"
            f"- 人工修正判定：{manual_judge}\n"
            f"- 人工真实原因：{true_reason}\n"
            "指令：当你再次遇到相似语境时，请对齐“人工修正判定”和“人工真实原因”所反映的体验尺度；"
            "尤其是对 answer_issue 的 Common Sense 体验评估，不要因为细节不完美就作出错误的“不合理”。"
        )

    return "【重要校准：请学习以下人类专家的真实判例（用于尺度对齐）】\n\n" + "\n\n".join(blocks)


def export_unreasonable_cases_to_human_tasks(
    results: List[Dict[str, Any]],
    human_tasks_xlsx_path: str,
) -> None:
    """
    Milestone 8（导出阶段）：
    把所有 LLM 判定 answer_issue = "不合理" 的案例导出到 human_calibration_tasks.xlsx。

    设计目标：
    1) 不覆盖用户已填写的“人工修正判定/人工给出的真实原因”；
    2) 会更新/补齐新导出的“SessionID/对话历史/LLM判定原因”；
    3) 保留旧文件中可能已有的人审条目（不删行）。
    """
    new_rows: List[Dict[str, Any]] = []
    for it in results:
        llm_result = it.get("llm_result") or {}
        if llm_result.get("answer_issue") != "不合理":
            continue

        session_id = it.get("session_id")
        session_history_text = it.get("session_history_text") or ""
        common_sense_reason = llm_result.get("common_sense_reason") or ""
        if not session_id:
            continue

        new_rows.append(
            {
                "SessionID": session_id,
                "对话历史 (User & Bot)": session_history_text,
                "LLM判定原因 (common_sense_reason)": common_sense_reason,
                "人工修正判定": "",
                "人工给出的真实原因": "",
            }
        )

    if not new_rows:
        print("Milestone 8：本次没有可导出的“不合理”案例，跳过导出。")
        return

    existing_df = None
    if os.path.exists(human_tasks_xlsx_path):
        try:
            import pandas as pd

            existing_df = pd.read_excel(human_tasks_xlsx_path)
        except Exception as exc:
            print(
                f"警告：读取旧人审表失败，将以本次导出覆盖重建：{type(exc).__name__}: {exc}"
            )
            existing_df = None

    cols = [
        "SessionID",
        "对话历史 (User & Bot)",
        "LLM判定原因 (common_sense_reason)",
        "人工修正判定",
        "人工给出的真实原因",
    ]

    import pandas as pd

    if existing_df is None or existing_df.empty:
        df_existing = pd.DataFrame(columns=cols)
    else:
        df_existing = existing_df.copy()
        for c in cols:
            if c not in df_existing.columns:
                df_existing[c] = ""

    def _cell_to_str(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, float) and (v != v):  # NaN
            return ""
        return str(v).strip()

    existing_map: Dict[str, Dict[str, str]] = {}
    for row in df_existing.itertuples(index=False):
        sid = _cell_to_str(getattr(row, "SessionID"))
        if not sid:
            continue
        existing_map[sid] = {
            "人工修正判定": _cell_to_str(getattr(row, "人工修正判定", "")),
            "人工给出的真实原因": _cell_to_str(
                getattr(row, "人工给出的真实原因", "")
            ),
        }

    for r in new_rows:
        sid = str(r["SessionID"]).strip()
        if sid in existing_map:
            r["人工修正判定"] = existing_map[sid]["人工修正判定"]
            r["人工给出的真实原因"] = existing_map[sid]["人工给出的真实原因"]

    df_new = pd.DataFrame(new_rows, columns=cols)
    df_all = pd.concat([df_existing[cols], df_new[cols]], ignore_index=True)
    df_all["SessionID"] = df_all["SessionID"].astype(str)
    df_all = df_all.drop_duplicates(subset=["SessionID"], keep="last")

    for sid, fields in existing_map.items():
        if not fields["人工修正判定"] and not fields["人工给出的真实原因"]:
            continue
        mask = df_all["SessionID"] == sid
        if mask.any():
            df_all.loc[mask, "人工修正判定"] = fields["人工修正判定"]
            df_all.loc[mask, "人工给出的真实原因"] = fields["人工给出的真实原因"]

    try:
        df_all.to_excel(human_tasks_xlsx_path, index=False)
        print(f"Milestone 8：人审任务表已导出到：{human_tasks_xlsx_path}")
    except Exception as exc:
        print(f"警告：导出 human_calibration_tasks.xlsx 失败：{type(exc).__name__}: {exc}")

# ===========================
# 大模型调用（带重试）
# ===========================


def make_retry_decorator(max_attempts: int = 3):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.APIError,
            )
        ),
    )


def call_llm_for_map_review(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_content: str,
) -> str:
    """
    调用 LLM，返回原始 content 字符串。
    """

    @make_retry_decorator(max_attempts=3)
    def _call() -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    return _call()


# ===========================
# System Prompt（V2.1 规范 3）
# ===========================


def build_system_prompt_v21() -> str:
    # V2.1 Map 阶段：双层评价逻辑 + 结构化 JSON 强约束
    # 第一层（体验/常识）决定 answer_issue；第二层（仅当不合理时）溯源 prompt_defect_suspected/defect_reason。
    return (
        "你是途虎养车资深门店店长兼用户体验专家。"
        "你需要对提供的【当前 Session 对话历史 + 人设1 + 人设2 + 当前挂载的 1 份专家策略】进行微观审查。\n\n"
        "双层评价逻辑（必须按顺序执行）：\n"
        "第一层（Common Sense 体验评价，用于决定 answer_issue）：\n"
        "你必须同时遵循下面两大核心原则与判断标准：\n\n"
        "原则一升级：【全局语义块包容（彻底豁免分段/多轮连发）】\n"
        "聚合思维：当你在对话历史中看到客服连续发送多条短句（例如：第一句“亲亲稍等~”，第二句“帮您看了一下”，第三句“这款机油完全适配您的爱车哦”），你绝对不允许将它们拆开单独评判！"
        "你必须在脑海中将这连续的几句话合并成一个完整的“语义块”来理解。\n\n"
        "豁免条件：只要这个“语义块”的最终结果解答了用户的问题，即使中间穿插了“嗯嗯”、“稍等”、“正在查询”等无效信息的碎片短句，也必须无条件判定为 answer_issue = \"合理\"。\n\n"
        "强约束：先聚合语义块，再评判 common sense 体验。严禁对语义块内部的短句逐段打分；必须以合并后的整体结果是否完成意图承接、是否阻塞体验为唯一判断依据。\n\n"
        "强约束判例学习（必须遵循）：\n"
        "用户：“前片那（前刹车片呢？）”\n\n"
        "客服（分段连发）：“亲~” -> “购买时请确认适配信息哈” -> “根据您的车型，前刹车片也需要更换一套4片哦~”\n\n"
        "AI 你的正确判定：合理。\n"
        "判定原因：虽然客服话术被切碎成三段，且单看前两句似乎没用，但综合整个语义块，客服已经准确告知用户“需要买一套4片”，意图承接完成，体验没有阻塞；因此必须判定为合理。\n\n"
        "原则二：【核心判断标准：意图承接与体验阻塞】\n"
        "判断“不合理”，必须围绕以下维度：意图承接是否正确、体验是否被阻塞（导致用户无法继续完成目标/必须反复追问/被生硬拒绝/陷入死循环）。"
        "只要出现下面四类情况之一，就必须判定 answer_issue = \"不合理\"。\n\n"
        "（A）答非所问 / 意图识别错误（必须判不合理）\n"
        "示例：用户问的是假设性问题（如“如果我现在下单，多久能到？”），客服却去查询用户历史已有订单信息，仍然不能直接回答假设性到达时间。\n\n"
        "（B）机械触发 / 缺乏常识应对（必须判不合理）\n"
        "示例：用户清晰描述车辆具体故障症状（如异响/报警灯/动力不足），客服没有基于汽车常识解释或给出针对性处理建议，反而触发生硬的“特定故障通用问答模板”，与用户症状不匹配。\n\n"
        "（C）场景未覆盖 / 无法承接需求（必须判不合理）\n"
        "示例：用户要求把两款机油/轮胎进行“对比推荐”，客服只给出单品信息，完全没有承接“对比”的决策需求，也没有把差异点讲清。\n\n"
        "（D）转化 / 体验阻塞（必须判不合理）\n"
        "示例：客服生硬拒绝用户、汽车常识性错误、或者进入“死循环”反复要求用户提供车型信息而惹怒用户，导致用户无法推进购买/咨询/决策。\n\n"
        "反之：\n"
        "只要客服准确承接用户意图、解决问题或给出可行的下一步路径，并且没有触发体验阻塞（没有死循环/没有明显常识错误/没有把用户引向无关内容），即使表述不够完美或存在分段/话术碎的问题，也必须判定 answer_issue = \"合理\"。\n\n"
        "第二层（根因溯源，用于决定 prompt_defect_suspected 和 defect_reason）：\n"
        "1) 仅当第一层得到 answer_issue = \"不合理\" 时，才允许你审阅【人设1 / 人设2 / 当前挂载的专家策略文档】。\n"
        "2) 反推导致“回答不合理”的根因是：人设1/人设2/当前专家策略的哪条规则或缺失机制，可能导致翻车。\n"
        "3) 若怀疑根因来自文档规则：prompt_defect_suspected = true，并在 defect_reason 中指出“疑似是哪个文档 + 哪条规则/缺失点”造成问题。\n"
        "4) 若 answer_issue = \"合理\"：prompt_defect_suspected 必须为 false，defect_reason 必须输出空字符串 \"\"。\n\n"
        "分类与输出要求：\n"
        "1) rewrite_issue：仅基于 rewrite 是否清晰准确表达用户意图，输出“合理/不合理”。\n"
        "2) primary_category：从“轮胎/机油/电瓶/保养/...”中选择最贴近的一类。\n"
        "3) secondary_category：必须是动名词组合，且绝不能超过 4 个中文字符。\n"
        "4) common_sense_reason：必须给出基于日常沟通的常理与体验判断的简述（始终需要输出）。\n"
        "   - common_sense_reason 只能引用【用户问题 + 客服回答】本身带来的体验/常识判断原因。\n"
        "   - 绝不能提及任何“系统规则/人设/专家策略/文档/Prompt/工具调用/策略缺失”等词或含义。\n"
        "   - 若 answer_issue 为“不合理”，请说明客服哪里让人听着不舒服、敷衍、答非所问或违背汽配常识；若为“合理”，说明哪里让人觉得专业、清晰或有效解决疑虑。\n"
        "5) is_typical：本案是否具有代表性（true/false）。\n\n"
        "输出强约束：\n"
        "- 你必须只返回一个合法的 JSON 对象字符串，不要输出任何额外文字。\n"
        "- 不要使用 Markdown 代码块，不要在外层包裹 ```json。\n"
        "- JSON 字段名必须严格一致，字段值按枚举与类型输出。\n\n"
        "JSON 结构（字段名完全一致）：\n"
        "{\n"
        '  "rewrite_issue": "合理/不合理",\n'
        '  "answer_issue": "合理/不合理",\n'
        '  "primary_category": "轮胎/机油/电瓶/保养/...",\n'
        '  "secondary_category": "产品推荐",\n'
        '  "common_sense_reason": "简述",\n'
        '  "prompt_defect_suspected": true/false,\n'
        '  "defect_reason": "如果疑似为 true 的简述",\n'
        '  "is_typical": true/false\n'
        "}\n"
        "注意：若 prompt_defect_suspected 为 false，defect_reason 必须为空字符串 \"\"。"
        "另外 common_sense_reason 始终必须为非空字符串。"
    )


# ===========================
# 动态挂载选择逻辑（基于 toolsmodelanswer + 映射表）
# ===========================


def choose_expert_strategy_for_session(
    df,
    row_indices: List[int],
    tools_col: Optional[str],
    dialogue_col: Optional[str],
    rewrite_col: Optional[str],
    answer_col: Optional[str],
    mapping_rows: List[step5_v21.MappingRow],
    strategies_dir: str,
) -> Tuple[Optional[step5_v21.MappingRow], int, Set[str], str, str, str, Optional[str]]:
    """
    为某个 session 选择专家策略。

    返回：
      - best_mapping_row
      - best_score
      - tool_names
      - current_tools_text
      - current_dialogue
      - current_rewrite
      - current_answer
      - expert_docx_path
    """
    sub = df.loc[row_indices]

    current_tools_text: str = ""
    current_dialogue: str = ""
    current_rewrite: str = ""
    current_answer: str = ""

    current_row_idx: Optional[Any] = None

    # 优先找 toolsmodelanswer 非空的那一行
    if tools_col and tools_col in sub.columns:
        for idx, r in sub.iterrows():
            t = step5_v21.normalize_text(r.get(tools_col, ""))
            if t:
                current_row_idx = idx
                current_tools_text = t
                break
        if current_row_idx is None and len(sub) > 0:
            current_row_idx = sub.index[0]
    else:
        current_row_idx = sub.index[0] if len(sub) > 0 else None

    if current_row_idx is not None:
        r = df.loc[current_row_idx]
        current_tools_text = (
            step5_v21.normalize_text(r.get(tools_col, "")) if tools_col else ""
        )
        current_dialogue = (
            step5_v21.normalize_text(r.get(dialogue_col, "")) if dialogue_col else ""
        )
        current_rewrite = (
            step5_v21.normalize_text(r.get(rewrite_col, "")) if rewrite_col else ""
        )
        current_answer = (
            step5_v21.normalize_text(r.get(answer_col, "")) if answer_col else ""
        )

    tool_names = step5_v21.extract_tool_names_from_toolsmodelanswer(current_tools_text)

    search_text = "\n".join(
        [
            step5_v21.normalize_text(current_tools_text),
            current_dialogue,
            current_rewrite,
            current_answer,
        ]
    )

    best_mapping, best_score = step5_v21.infer_strategy_from_session_line(
        tool_names=tool_names,
        search_text=search_text,
        mapping_rows=mapping_rows,
    )

    expert_docx_path: Optional[str] = None
    if best_mapping is not None:
        expert_docx_path = step5_v21.find_docx_by_fuzzy_prefix(
            strategy_prefix=best_mapping.strategy_prefix,
            strategies_dir=strategies_dir,
        )

    return (
        best_mapping,
        best_score,
        tool_names,
        current_tools_text,
        current_dialogue,
        current_rewrite,
        current_answer,
        expert_docx_path,
    )


# ===========================
# 主程序：只跑前 5 个 Session
# ===========================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="途虎智能客服 Agent - Milestone 6 (V2.1 Map 阶段，前 5 Session)"
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
        help="意图描述和对应专家策略.xlsx 路径。",
    )
    parser.add_argument(
        "--strategies-dir",
        "-s",
        default=".",
        help="专家策略 docx 目录（用于模糊寻址）。",
    )
    parser.add_argument(
        "--persona1",
        default="人设1意图识别与判断使用哪些工具.docx",
        help="人设1 docx 路径。",
    )
    parser.add_argument(
        "--persona2",
        default="人设2 根据工具返回的结果和常识回答问题.docx",
        help="人设2 docx 路径。",
    )
    parser.add_argument(
        "--sessions",
        "-n",
        type=int,
        default=5,
        help="只跑前 N 个 Session（默认 5）。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tuhu_session_analysis_step6_v21.json",
        help="Map 阶段结果输出文件名（JSON）。",
    )
    parser.add_argument(
        "--human-tasks-xlsx",
        default="human_calibration_tasks.xlsx",
        help="Milestone 8 人审任务表路径（用于导出/读取注入）。",
    )
    parser.add_argument(
        "--disable-human-prompt-injection",
        action="store_true",
        help="禁用从 human_calibration_tasks.xlsx 读取人工修正并追加到 system prompt。",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在：{args.input}")

    # 1) 读取并聚合 session
    df_raw = step1.read_input_file(args.input)
    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)

    tools_col = step1.detect_column(columns, step1.POSSIBLE_TOOLS_ANSWER_COLUMNS)
    if not tools_col:
        raise ValueError("无法识别 toolsmodelanswer 字段，请检查表结构。")

    dialogue_col = step1.detect_column(columns, step1.POSSIBLE_DIALOGUE_SENTENCE_COLUMNS)
    rewrite_col = step1.detect_column(columns, step1.POSSIBLE_REWRITE_QUERY_COLUMNS)
    answer_col = step1.detect_column(columns, step1.POSSIBLE_ANSWER_COLUMNS)

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    rows_before_compress = len(df)
    df = compress_consecutive_same_dialogue_rows(
        df=df,
        session_col=session_col,
        dialogue_col=dialogue_col,
        answer_col=answer_col,
    )
    rows_after_compress = len(df)
    compressed_count = rows_before_compress - rows_after_compress
    compressed_ratio = (
        (compressed_count / rows_before_compress) * 100.0
        if rows_before_compress > 0
        else 0.0
    )
    print(
        "会话压缩完成："
        f"原始行数={rows_before_compress}，压缩后行数={rows_after_compress}，"
        f"减少={compressed_count}（{compressed_ratio:.2f}%）"
    )
    # 改为“单轮意图”粒度：每一行都是一个可评估单元（已经过同问连续分段合并）
    if len(df) == 0:
        print("没有可用数据行，程序结束。")
        return

    # 排序后按行评估，保证“前 N 条”稳定
    eval_df = df.copy().reset_index(drop=False).rename(columns={"index": "_eval_orig_index"})
    time_col_eval = step1.detect_column(list(eval_df.columns), step1.POSSIBLE_TIME_COLUMNS)
    if time_col_eval and time_col_eval in eval_df.columns:
        eval_df[time_col_eval] = pd.to_datetime(eval_df[time_col_eval], errors="coerce")
        eval_df = eval_df.sort_values(by=[time_col_eval, "_eval_orig_index"], kind="mergesort")
    else:
        eval_df = eval_df.sort_values(by=["_eval_orig_index"], kind="mergesort")
    eval_df = eval_df.reset_index(drop=True)

    total_units = len(eval_df)
    n = max(1, min(args.sessions, total_units))
    print(f"Milestone 6：将审查前 {n} 条单轮数据（总计 {total_units}）。")

    # 2) 读取映射表、persona1/persona2（用于 V2.1 动态挂载上下文注入）
    mapping_rows = step5_v21.load_intent_strategy_mapping(args.mapping)
    persona1_text = step5_v21.read_doc_or_text_plain_text(args.persona1)
    persona2_text = step5_v21.read_doc_or_text_plain_text(args.persona2)

    # doc 读取缓存（减少重复 io）
    doc_cache: Dict[str, str] = {}

    def get_doc_text(path: str) -> str:
        if path in doc_cache:
            return doc_cache[path]
        t = step5_v21.read_doc_or_text_plain_text(path)
        doc_cache[path] = t
        return t

    # 3) 初始化 LLM client
    client = OpenAI(api_key=step2.API_KEY, base_url=step2.API_BASE_URL)
    model_name = step2.MODEL_NAME
    system_prompt = build_system_prompt_v21()
    if not args.disable_human_prompt_injection:
        human_text = build_human_calibration_fewshot_text(args.human_tasks_xlsx)
        if human_text:
            system_prompt = system_prompt + "\n\n" + human_text
            print(f"Milestone 8：已追加人审校准 few-shot（文件：{args.human_tasks_xlsx}）。")

    results: List[Dict[str, Any]] = []

    # rewrite_history 仅作为上下文补充，不参与“当前轮次”边界判断
    history_col = step1.detect_column(
        list(eval_df.columns),
        ["rewrite_history", "rewritehistory", "history", "dialogue_history"],
    )

    for idx in tqdm(range(n), desc="Milestone6进度"):
        unit_row = eval_df.iloc[idx].to_dict()
        row_indices = [int(unit_row.get("_eval_orig_index"))]

        (
            best_mapping,
            best_score,
            tool_names,
            current_tools_text,
            current_dialogue,
            current_rewrite,
            current_answer,
            expert_docx_path,
        ) = choose_expert_strategy_for_session(
            df=df,
            row_indices=row_indices,
            tools_col=tools_col,
            dialogue_col=dialogue_col,
            rewrite_col=rewrite_col,
            answer_col=answer_col,
            mapping_rows=mapping_rows,
            strategies_dir=args.strategies_dir,
        )

        if best_mapping is None or expert_docx_path is None:
            print(
                f"第 {idx + 1} 条：未能完成映射挂载（best_mapping={best_mapping} expert_docx_path={expert_docx_path}），跳过。"
            )
            continue

        conversation_rows = step5_v21.extract_conversation_rows_for_session(
            df=df,
            row_indices=row_indices,
            dialogue_col=dialogue_col,
            rewrite_col=rewrite_col,
            answer_col=answer_col,
        )

        # 动作1：打印前 3 条“合并后的数据”验数（在 LLM 调用前）
        if idx < 3:
            preview_row = pick_preview_row_for_merged_session(conversation_rows)
            preview_q = str(preview_row.get("dialoguesentence") or "").strip()
            preview_a = str(preview_row.get("answer") or "").strip()
            print("=== 合并数据验证 ===")
            print(f"[SessionID]: {unit_row.get(session_col)}")
            print(f"[用户问题]: {preview_q}")
            print(f"[拼接后的客服回答]: {preview_a}")
            print("===================")

        # 只评估“当前这一轮”的问题与拼接后回答，避免整段 session 污染当前轮次判断
        current_row = conversation_rows[0] if conversation_rows else {}
        current_user_q = str(current_row.get("dialoguesentence") or "").strip()
        current_rewrite_q = str(current_row.get("rewritequery") or "").strip()
        current_answer_text = str(current_row.get("answer") or "").strip()
        rewrite_history_text = (
            str(unit_row.get(history_col) or "").strip()
            if history_col and history_col in eval_df.columns
            else ""
        )

        session_history_text = (
            "【当前评估轮次】\n"
            f"用户问题：{current_user_q}\n"
            f"重写问题：{current_rewrite_q}\n"
            f"客服回答（已合并分段）：{current_answer_text}\n"
        )
        if rewrite_history_text:
            session_history_text += f"\n【rewrite_history（仅辅助上下文）】\n{rewrite_history_text}\n"

        expert_strategy_text = get_doc_text(expert_docx_path)
        context_prompt = step5_v21.build_context_prompt(
            session_history_text=session_history_text,
            persona1_text=persona1_text,
            persona2_text=persona2_text,
            expert_strategy_text=expert_strategy_text,
        )
        user_content = "请基于下面内容完成微观审查，并输出 V2.1 规范 3 的 JSON：\n\n" + context_prompt + "\n"

        # 4) LLM 调用
        try:
            llm_raw = call_llm_for_map_review(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                user_content=user_content,
            )
            llm_sanitized = sanitize_llm_output(llm_raw)
            parsed = json.loads(llm_sanitized)
        except Exception as exc:
            parsed = {
                "rewrite_issue": "不合理",
                "answer_issue": "不合理",
                "primary_category": "其他",
                "secondary_category": "错误结果",
                "prompt_defect_suspected": False,
                "defect_reason": "",
                "common_sense_reason": "JSON 解析失败或调用失败：请人工检查。",
                "is_typical": False,
            }
            print(f"警告：JSON 解析失败或调用失败：{type(exc).__name__}: {exc}")
            llm_raw = ""

        # 5) 约束校验（仅提示，不强行改写字段）
        validate_secondary_category(parsed.get("secondary_category"))

        # 6) 打印 + 收集
        print("\n" + "=" * 80)
        print(f"样本 {idx + 1}")
        print(
            f"用户ID: {unit_row.get(user_col) if user_col else ''}    SessionID: {unit_row.get(session_col)}"
        )
        print("-" * 80)
        print(f"当前行触发工具名集合：{sorted(list(tool_names))}")
        print(
            f"映射触发意图(best_score={best_score})：{best_mapping.intent_desc} -> strategy_prefix={best_mapping.strategy_prefix}"
        )
        print(f"已动态挂载专家策略 docx：{os.path.basename(expert_docx_path)}")
        print(f"动态上下文字符总长度：{len(context_prompt)}")
        print("\nLLM 评价 JSON：")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print("=" * 80 + "\n")

        results.append(
            {
                "user_id": unit_row.get(user_col) if user_col else None,
                "session_id": unit_row.get(session_col),
                "tool_names": sorted(list(tool_names)),
                "mounted_expert_docx": expert_docx_path,
                "mapping_best_score": best_score,
                "llm_result": parsed,
                "error": None,
                # Milestone 8 导出需要对话历史文本
                "session_history_text": session_history_text,
            }
        )

    # 7) 保存结果，供 Milestone 7 使用
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Milestone 6 结果已保存到：{args.output}")
    except Exception as exc:
        print(f"警告：保存结果失败：{type(exc).__name__}: {exc}")

    # 8) Milestone 8（导出阶段）：导出 answer_issue="不合理" 的案例给人工修正
    try:
        export_unreasonable_cases_to_human_tasks(
            results=results,
            human_tasks_xlsx_path=args.human_tasks_xlsx,
        )
    except Exception as exc:
        print(f"警告：Milestone 8 导出失败：{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()

