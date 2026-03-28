#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step3.py

里程碑 Milestone 3：并发批处理与容错机制 (全量跑批)。

功能概述：
1. 复用 step1 与 step2 的逻辑，读取离线数据并聚合 Session。
2. 基于 step2 的 Prompt 与大模型调用方式，对所有 Session 进行批量审查。
3. 使用 ThreadPoolExecutor + threading.Semaphore 控制最大并发数（默认 5，可配置）。
4. 使用 tenacity 提供指数退避重试机制：
   - 针对 API 相关异常（网络错误、超时、限流等）自动重试，最多 3 次。
   - 失败会记录为 error_session，不中断主进程。
5. 使用 tqdm 展示整体进度条，便于观察跑批进度。
6. 将所有 Session 的审查结果保存到本地 JSON 文件，供后续 Milestone 4 生成 HTML 报告复用。

注意：
1. 本脚本只实现并发 + 容错与结果落盘，不做 HTML 报告。
2. 不修改 step1.py 与 step2.py 中已有的正确逻辑，只做复用与封装。
"""

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

import step1
import step2


# ===========================
# 并发与重试配置
# ===========================

DEFAULT_MAX_WORKERS = 5
DEFAULT_MAX_RETRIES = 3


def build_dataset(
    input_path: str,
) -> Tuple[Any, Any, str, Optional[str], Optional[str], Optional[str]]:
    """
    读取原始文件、进行清洗与会话聚合，返回：
        - df: 清洗后的 DataFrame
        - session_df: 会话聚合 DataFrame
        - session_col: sessionid 列名
        - user_col: userid 列名（可能为 None）
        - dialogue_col: dialoguesentence 列名（可能为 None）
        - rewrite_col: rewritequery 列名（可能为 None）
        - answer_col: answer 列名（可能为 None）
    """
    print(f"开始读取文件：{input_path}")
    df_raw = step1.read_input_file(input_path)
    print(f"读取完成，原始行数：{len(df_raw)}，列数：{len(df_raw.columns)}")

    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")

    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    dialogue_col = step1.detect_column(columns, step2.POSSIBLE_DIALOGUE_SENTENCE_COLUMNS)
    rewrite_col = step1.detect_column(columns, step2.POSSIBLE_REWRITE_QUERY_COLUMNS)
    answer_col = step1.detect_column(columns, step2.POSSIBLE_ANSWER_COLUMNS)

    print(f"检测到 sessionid 字段名：{session_col}")
    if user_col:
        print(f"检测到 userid 字段名：{user_col}")
    else:
        print("警告：未检测到 userid 字段，将仅按 sessionid 聚合会话。")

    if dialogue_col:
        print(f"检测到 dialoguesentence 字段名：{dialogue_col}")
    else:
        print("警告：未检测到 dialoguesentence 字段，Prompt 中将缺少用户原始问题。")

    if rewrite_col:
        print(f"检测到 rewritequery 字段名：{rewrite_col}")
    else:
        print("警告：未检测到 rewritequery 字段，Prompt 中将缺少重写问题。")

    if answer_col:
        print(f"检测到 answer 字段名：{answer_col}")
    else:
        print("警告：未检测到 answer 字段，Prompt 中将缺少机器人回答。")

    df = step1.basic_cleaning(df_raw, session_col=session_col)

    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    print(f"聚合完成，总会话数量：{len(session_df)}")

    return df, session_df, session_col, user_col, dialogue_col, rewrite_col, answer_col


def make_retry_decorator(max_attempts: int = DEFAULT_MAX_RETRIES):
    """
    构造 tenacity 的重试装饰器。
    """
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


def analyze_single_session_factory(
    client,
    df,
    dialogue_col: Optional[str],
    rewrite_col: Optional[str],
    answer_col: Optional[str],
    sop_text: str,
    semaphore: threading.Semaphore,
    max_retries: int = DEFAULT_MAX_RETRIES,
):
    """
    返回一个带有重试与并发控制的单 Session 分析函数。
    """

    retry_decorator = make_retry_decorator(max_attempts=max_retries)

    @retry_decorator
    def _call_with_retry(session_record: Dict[str, Any]) -> Dict[str, Any]:
        with semaphore:
            row_indices = session_record.get("row_indices") or []
            conversation_rows = step2.extract_conversation_rows_for_session(
                df=df,
                row_indices=row_indices,
                dialogue_col=dialogue_col,
                rewrite_col=rewrite_col,
                answer_col=answer_col,
            )
            result = step2.call_llm_for_session(
                client=client,
                session_record=session_record,
                conversation_rows=conversation_rows,
                sop_text=sop_text,
            )
            return {
                "user_id": session_record.get("user_id"),
                "session_id": session_record.get("session_id"),
                "row_indices": row_indices,
                "llm_result": result,
            }

    def wrapper(session_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        包一层安全调用：
        - 对于网络 / API 异常，自动重试，最多 max_retries 次。
        - 对于超过重试次数仍失败的情况，记录 error_session，不抛出到主线程。
        """
        try:
            return _call_with_retry(session_record)
        except Exception as exc:
            # 任何异常都记录下来，主进程不中断
            return {
                "user_id": session_record.get("user_id"),
                "session_id": session_record.get("session_id"),
                "row_indices": session_record.get("row_indices") or [],
                "error": f"{type(exc).__name__}: {exc}",
            }

    return wrapper


def run_batch_analysis(
    input_path: str,
    max_sessions: Optional[int],
    max_workers: int,
    max_retries: int,
    sop_path: str,
    output_path: str,
) -> None:
    """
    入口函数：执行全量（或部分） Session 的并发跑批分析。
    """
    # 1. 构建数据集
    df, session_df, _, _, dialogue_col, rewrite_col, answer_col = build_dataset(
        input_path
    )

    total_sessions = len(session_df)
    if total_sessions == 0:
        print("没有可用的会话数据，程序结束。")
        return

    if max_sessions is not None:
        total_to_run = max(1, min(max_sessions, total_sessions))
    else:
        total_to_run = total_sessions

    print(f"此次将分析的 Session 数量：{total_to_run}（总计 {total_sessions}）")

    # 2. 构建客户端与 SOP
    sop_text = step2.load_sop_text(sop_path)
    client = step2.build_client()

    # 3. 并发控制
    semaphore = threading.Semaphore(max_workers)
    analyze_single = analyze_single_session_factory(
        client=client,
        df=df,
        dialogue_col=dialogue_col,
        rewrite_col=rewrite_col,
        answer_col=answer_col,
        sop_text=sop_text,
        semaphore=semaphore,
        max_retries=max_retries,
    )

    results: List[Dict[str, Any]] = []

    print(
        f"开始并发跑批，大模型最大并发：{max_workers}，最大重试次数：{max_retries}。"
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx in range(total_to_run):
            session_record = session_df.iloc[idx].to_dict()
            futures.append(executor.submit(analyze_single, session_record))

        for future in tqdm(as_completed(futures), total=len(futures), desc="分析进度"):
            res = future.result()
            results.append(res)

    # 4. 统计错误 Session
    error_sessions = [r for r in results if "error" in r]
    print(
        f"跑批完成，成功 Session 数量：{len(results) - len(error_sessions)}，"
        f"失败 Session 数量：{len(error_sessions)}"
    )
    if error_sessions:
        print("部分失败 Session 示例：")
        for item in error_sessions[:5]:
            print(
                f"  user_id={item.get('user_id')} session_id={item.get('session_id')} error={item.get('error')}"
            )

    # 5. 结果落盘
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"全部 Session 的分析结果已保存到：{output_path}")
    except Exception as exc:
        print(f"警告：保存结果到 {output_path} 失败：{exc}")


def main() -> None:
    """
    命令行入口。
    """
    parser = argparse.ArgumentParser(
        description="途虎智能客服对话审查 Agent - Milestone 3 (并发批处理与容错机制)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入文件路径（与 step1/step2 相同，支持 .csv / .xls / .xlsx）。",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="本次最多分析的 Session 数量（默认 None 表示全量）。",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"最大并发线程数（默认 {DEFAULT_MAX_WORKERS}，建议在 5~10 之间）。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"单个 Session 调用失败时的最大重试次数（默认 {DEFAULT_MAX_RETRIES}）。",
    )
    parser.add_argument(
        "--sop",
        type=str,
        default="tuhu_sop.txt",
        help="本地 SOP 文本文件路径，默认当前目录下的 tuhu_sop.txt。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tuhu_session_analysis_step3.json",
        help="批量分析结果输出 JSON 文件名，默认 tuhu_session_analysis_step3.json。",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    run_batch_analysis(
        input_path=input_path,
        max_sessions=args.max_sessions,
        max_workers=max(1, args.max_workers),
        max_retries=max(1, args.max_retries),
        sop_path=args.sop,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

