# encoding: utf8

"""空间语义标注跑批：读取合格句对 → feature → schema → judge → 落盘汇总。

语料提取见 extract/sentence_extract.py，机械替换及两侧检查见 replace/sentence_replace.py。
正式 pair/synonym 图只消费替换阶段的句对，不展开候选、不重新筛选、不准备句法证据。
原句和替换句各用自己的关注词与三元组，特征/图式示例从 system_prompt/examples 按词选择。
single/alter_check 保留为显式诊断图，不是正式句对标注入口。

默认输入：results/location_pairs.jsonl（方位词）、results/direction_pairs.jsonl（趋向动词）。
--limit/--offset 按通过两侧检查的句对计数。结果为 JSONL、同名 CSV（可关闭）及 summary.json，
收尾按 (line_no, candidate.index) 排序，可与 all_in_one.py 对账。模型执行日志见 log/。

用法：
    python annotate --corpus results/location_pairs.jsonl --limit 5
    python annotate --corpus results/direction_pairs.jsonl --corpus-type 趋向动词 --limit 20
    python annotate --workers 4 --limit 200
    python annotate --no-llm --batch off --limit 5
    python annotate --models model-a model-b --workers 4
    python annotate --batch native --batch-resume --batch-cache results/run.batch-cache.jsonl
    python annotate --list-graphs

--models 保留模型独立执行与加权投票，缺票及平票转人工复核。--batch auto 优先使用原生 Batch，
不可用时明确回退同步；native 不允许回退。批处理仅包含特征、图式和判定波次，查表优先级不变。
完整命令与服务器迁移注意事项见 .docs/stage-migration.md。
"""

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import agent_log
import agents as agent_factory
import batch_inference as bi
import config
from replace import corpus_pairs as cp
import file_paths
from annotate import spatial_agents as sa
from annotate import spatial_graphs as sg
from annotate import spatial_tools as st
from annotate import voting

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = _PROJECT_ROOT / "results" / st.CORPUS_TYPE_CORPORA[st.DEFAULT_CORPUS_TYPE]
"""默认语料：默认语料类型（方位词）的那一份。按类型取用见 st.CORPUS_TYPE_CORPORA"""
DEFAULT_MODEL = sa.DEFAULT_MODEL
"""默认模型配置名（model_config/api_keys.json 里的键）"""


# --------------------------------------------------------------------------
# 一、结果落盘
# --------------------------------------------------------------------------


class ResultWriter:
    """结果落盘器：加锁追加 jsonl，并按各图自己的口径统计计数。

    多线程跑图时 save 节点会并发调用，故用锁串起"写文件 + 计数"；
    每条写完即关文件，长跑中途 tail 也能看到完整行。

    收尾（finish）会把文件按 (line_no, candidate.index) 重排（见 _sort_records）：
    并发下写进来的顺序是完成顺序，重排后才与 all_in_one.py 的输出同序、可逐行对账。
    同一份排好序的记录再顺手导出一份同名 .csv（见 __init__ 的 csv_columns），行序与 jsonl 一致。

    统计口径由图决定（spatial_graphs.GraphSpec.counter_names / tally）：
    pair 图统计两侧的 schema 来源，单句图只有一句——写死成 pair 的形状会让单句图
    静默统计出一堆 "none"。
    """

    def __init__(
        self,
        path: str | Path,
        *,
        counter_names: tuple[str, ...],
        tally,
        corpus_type: str | None = None,
        config_tables: dict[str, str] | None = None,
        reality_prefill: dict | None = None,
        csv_columns: tuple[tuple[str, str], ...] = (),
        model_name: str | None = None,
        overwrite: bool = False,
        input_records: int = 0,
        expanded_states: int = 0,
        progress_prefix: str = "",
    ):
        self.model_name = model_name
        self.progress_prefix = progress_prefix
        """进度行的前缀；多模型并行时各 writer 用它区分 [模型名]，单模型保持空串"""
        self.overwrite = bool(overwrite)
        """本次是否显式允许覆盖已有输出（--overwrite）；默认**拒绝**（见 __init__ 里的检查）"""
        self.input_records = int(input_records)
        """本次读进来的原句数（语料条数），只在汇总里用"""
        self.expanded_states = int(expanded_states)
        """本次展开出的候选状态数（= 落盘记录数的分母），只在汇总里用

        与 input_records 一起构成「原句 → 位置 → 候选」的转化链（D5）：光看 count 看不出
        一条原句展开成了几条、又卡在哪一步。"""
        """本次 --model 指定的模型配置名，追加到结果与汇总末尾；--no-llm 模式下为 None"""
        self.corpus_type = corpus_type
        """本次跑批的语料类型；只写进汇总，不进记录（记录形状不随本次改动变）"""
        self.config_tables = dict(config_tables or {})
        """本次实际读的四张配置表：{逻辑表名: 路径}，同样只进汇总（--alter-rules 换过表时很关键）"""
        self.reality_prefill = dict(reality_prefill or {})
        """闸门预填充的去重账（见 _prefill_reality）；同样只进汇总

        它解释的是「这一轮 reality 为什么比记录数少很多」——没有这项，事后看到调用数与记录数
        对不上只能靠猜。跑批是在**建好 writer 之后**才预填充的（图的接线校验要先跑，配置错不该
        等到花完预填充的钱才发现），所以这项由 main 在预填充之后补写到属性上。
        """
        self.agent_stats: dict[str, dict] = {}
        """各 agent 的调用统计：{名字: {"roundtrips", "failed", "cache_hits"[, "first_error"]}}
        （见 spatial_agents.BaseAgent.stats）

        run() 结束时才拿得到，同 reality_prefill：由 main 在收尾前补写到属性上。
        模型往返数是花费的直接口径（缓存命中省下的那部分不在这里，单列 cache_hits）；
        failed 是发出去但抛异常的调用数，**与 roundtrips 相加才是发出过的调用数**——
        整批被接口挡下时 roundtrips 是 0，光看它会误判成「一个请求都没发出去」（见
        _print_agent_stats）。失败过的 agent 另带 first_error（截断后的首个异常）。
        没跑到的 agent（--no-llm、或图没走到）报全 0，不是漏记。
        """
        self.csv_columns = tuple(csv_columns)
        """成对记录拍平 csv 的列序（见 spatial_graphs.GraphSpec.csv_columns）；空 = 本次不导

        由驱动按图与开关传入（--no-pairs-csv 关掉，非 pair 图本就传空）——writer 不认识图，
        也不该认识，收尾只照这份列序把已排序的记录导成同名 .csv（见 finish）。
        """
        self.extra_summary: dict = {}
        """调用方补写的汇总字段（多模型投票的 voting / models 块）

        由 main 在收尾前写入；finish 把它并进 .summary.json。单模型时为空，
        所以原有汇总形状与字段顺序不变。
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 拒绝覆盖已有输出（E1）：判断发生在**首次写入之前**——本构造是跑批里第一个碰输出
        # 文件的动作，等 write() 第一次追加时再查，文件里已经混进本轮的记录了。
        # 旧行为是不拒绝直接追加：显式 --out-file 指向旧结果时，旧记录、重复 id 与不同配置的
        # 结果会混在同一份文件里，收尾还会按 (line_no, candidate.index) 把整份重排，
        # 而 summary 的 count 只统计本轮写入，两份口径对不上。
        if not self.overwrite:
            with file_paths.open_exclusive(self.path):
                pass  # 只做独占创建：拿到一个空文件，之后的 append 都落在本轮自己的文件里
        else:
            self.path.write_text("", encoding="utf-8")  # 显式覆盖：先清空，免得混入历史记录
        self.lock = threading.Lock()
        # 常开追加句柄：write() 是长跑里的高频路径，每条都 open/close 在 Windows 上会
        # 把文件系统开销放大到与模型调用同量级。buffering=1 是行缓冲，写一行即 flush，
        # 保留「中途 tail 得到进度」的原行为；收尾排序前由 _close() 关闭。
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self.count = 0
        self.counter_names = tuple(counter_names)
        self.counters: dict[str, Counter] = {name: Counter() for name in self.counter_names}
        self.tally = tally
        self.started_at = time.time()

    def _close(self) -> None:
        """关闭常开追加句柄；可重复调用（已关闭时什么也不做）"""
        handle = getattr(self, "_handle", None)
        if handle is None:
            return
        with self.lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None

    def __del__(self):
        # 正常路径由 finish() -> _sort_records() 关闭；这里兜住异常退出或调用方中途放弃，
        # 免得 Windows 上留下一个占用结果文件的句柄。
        try:
            self._close()
        except Exception:
            pass

    def write(self, record: dict) -> None:
        record = {**record, "model_name": self.model_name}
        with self.lock:
            self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.count += 1
            self.tally(record, self.counters)
            if self.count % 10 == 0:
                print(f"  {self.progress_prefix}已写出 {self.count} 条", flush=True)

    def _sort_records(self) -> list[dict]:
        """把已落盘的 jsonl 按 (line_no, candidate.index) 重排（键见 sg.record_order_key）

        write() 是各 worker 在 save 节点并发追加（见模块 docstring），落盘顺序 = 完成顺序，
        与输入顺序无关。收尾统一重排，于是本脚本与 all_in_one.py 的 finish() 走同一段代码、
        产出同一个顺序，两份文件可逐行对账。all_in_one 本来就按输入顺序写，这里重排是兜底。

        流式落盘照旧：跑的过程中文件仍是完成顺序，tail 得到进度；只收尾重排一次。
        先写临时文件再原子替换，重排途中被打断也不会把结果文件截断。

        返回排好序的记录（没有文件或文件为空时返回空表）：收尾导 csv 就在这之后、直接拿这份
        内存里的记录写（见 finish）——csv 的行序于是**必然**等于重排后 jsonl 的行序，不必再读一遍
        文件，也不会有「两份文件次序不一致」的窗口。
        """
        # 必须先关闭写入句柄，Windows 上文件仍被打开时 rename 会失败；也保证缓冲全部落盘。
        self._close()
        if not self.path.exists():
            return []
        records = st.read_jsonl(self.path)
        if not records:
            return []
        if len(records) != self.count:
            # 条数对不上说明文件被中途截断或人工拼接改过；仍照常排，只当场说清楚
            print(f"警告：输出文件里有 {len(records)} 条，本次记录数为 {self.count}，请核对该文件")
        records.sort(key=sg.record_order_key)  # 稳定排序；键由构造保证唯一
        # 临时文件用 mkstemp 生成唯一名字，不再用确定的 `<输出>.sorting`：
        # 那个名字理论上可能正好是输入文件（或输入文件的别名），直接写会覆盖原始语料，
        # 且不受 --overwrite 控制。mkstemp 在目标同目录创建，replace 仍是原子的。
        fd, temp_name = tempfile.mkstemp(
            dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".sorting.tmp"
        )
        os.close(fd)
        tmp = Path(temp_name)
        try:
            st.write_jsonl(records, tmp)  # 复用唯一一份 jsonl 写法
            tmp.replace(self.path)  # Path.replace = os.replace，原子替换
        finally:
            tmp.unlink(missing_ok=True)
        print(f"  输出已按 (line_no, candidate.index) 排序（{len(records)} 条）", flush=True)
        return records

    def finish(self) -> dict:
        records = self._sort_records()
        csv_path = None
        if self.csv_columns and records:
            # 成对信息另导一份 csv 供 Excel 看（默认开；--no-pairs-csv 关）：拿刚排好的记录写，
            # 于是 csv 行序与 jsonl 行序同一份来源（见 _sort_records）
            # 路径走 st.pairs_csv_path 而不是 with_suffix(".csv")：后者在 --out-file result.csv
            # 这类路径上换出来还是它自己，收尾导 csv 会把刚写好的 jsonl 覆盖掉（E1）
            csv_path = st.pairs_csv_path(self.path)
            total, judged = st.write_pairs_csv(records, csv_path, self.csv_columns)
            print(
                f"  成对信息 csv：{csv_path}（{total} 行，其中 {judged} 行有同义判断）",
                flush=True,
            )
        # corpus_type / config_tables 是这次跑批「读的是哪一组表」的唯一凭据：user message
        # 不落盘（记录里只有 node 与耗时），事后要复现或核对口径就得靠这两项
        provenance = {"corpus_type": self.corpus_type} if self.corpus_type else {}
        if self.config_tables:
            provenance["config_tables"] = self.config_tables
        if self.reality_prefill:
            provenance["reality_prefill"] = self.reality_prefill
        if self.agent_stats:
            provenance["agent_stats"] = self.agent_stats
        summary = {
            "output_file": str(self.path),
            # csv 也是这次跑批的产物，与 output_file 并列记进汇总；没导（--no-pairs-csv 或非 pair 图）就不写
            **({"csv_file": str(csv_path)} if csv_path else {}),
            **provenance,
            # 数据契约（D3）：读取侧据此判断这份产物能不能直接解析，不必猜字段
            "schema_version": st.RESULT_SCHEMA_VERSION,
            # 分词口径（D4）：切分配置改了下标会位移，旧产物能不能复用就看这一项
            "segmentation": config.segmentation_signature(),
            # 「原句 → 候选 → 落盘记录」的转化链（D5）：count 只是最后一环，
            # 少了前两个数就看不出一条原句展开成了几条、中间掉了多少
            "input_records": self.input_records,
            "expanded_states": self.expanded_states,
            "count": self.count,
            "elapsed_seconds": round(time.time() - self.started_at, 1),
            **{name: dict(counter) for name, counter in self.counters.items()},
            "model_name": self.model_name,
            **self.extra_summary,
        }
        summary_path = self.path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["summary_file"] = str(summary_path)
        return summary


# --------------------------------------------------------------------------
# 二、命令行入口
# --------------------------------------------------------------------------


def _corpus_type_arg(value: str) -> str:
    """argparse 的 type：把类型规范化，认不出时报参数错（退出码 2）

    不用 type=st.normalize_corpus_type 直接挂上去：argparse 只会把异常类型名印出来，
    ValueError 的正文会丢掉，而这条消息正是要告诉人「有哪些类型可用」。
    """
    try:
        return st.normalize_corpus_type(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def add_batch_args(parser: argparse.ArgumentParser) -> None:
    """给跑批入口加上批量推理相关选项（spatial_runner 与 all_in_one 共用）

    默认 auto：模型配置声明了可用批量入口（OpenAI/OpenRouter，或显式 batch 配置）就
    走原生 Batch；没有就照旧同步执行。native 在不可用时直接报错，避免用户以为用了批量。
    """
    group = parser.add_argument_group("批量推理")
    group.add_argument(
        "--batch",
        choices=("off", "auto", "native"),
        default=config.BATCH_MODE,
        help=f"批量推理：auto（默认 {config.BATCH_MODE}，有原生 Batch 入口就用，否则同步回退）／native（必须用原生，"
        "不可用即报错）／off（完全走原来的同步链路）。原生入口按模型配置的 "
        "provider 或 batch 字段判定，见 batch_inference.batch_profile",
    )
    group.add_argument(
        "--batch-cache",
        default=None,
        help="批量结果 sidecar 路径（默认 <输出文件>.batch-cache.jsonl）；断点续跑时指向同一份",
    )
    group.add_argument(
        "--batch-resume",
        action="store_true",
        help="复用 --batch-cache 里已有的结果，并继续收 --batch-jobs.json 里未终结的 batch。"
        "默认每轮重新开始，避免旧结果混进新配置的产物",
    )
    group.add_argument(
        "--batch-poll-seconds",
        type=float,
        default=config.BATCH_POLL_SECONDS,
        help=f"轮询原生 batch 状态的间隔秒数（默认 {config.BATCH_POLL_SECONDS}）",
    )
    group.add_argument(
        "--batch-timeout-seconds",
        type=float,
        default=config.BATCH_TIMEOUT_SECONDS,
        help=f"等待单个原生 batch 完成的上限秒数（默认 {config.BATCH_TIMEOUT_SECONDS}，0 = 不限）；"
        "超时不会重复提交，batch ID 已落盘，可加 --batch-resume 继续收",
    )
    group.add_argument(
        "--batch-chunk-size",
        type=int,
        default=config.BATCH_CHUNK_SIZE,
        help=f"一个原生 batch 最多包含多少条请求（默认 {config.BATCH_CHUNK_SIZE}）；超出会分片提交",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="空间语义实验跑批：按指定的图把语料跑一遍，结果写入 jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法：")[-1],
    )
    add_batch_args(parser)
    parser.add_argument("--graph", default="pair", choices=sorted(sg.GRAPHS), help="跑哪张图（见 --list-graphs）")
    parser.add_argument("--list-graphs", action="store_true", help="打印可用的图与各自需要的 agent，然后退出")
    parser.add_argument(
        "--corpus-type",
        default=None,
        metavar="类型",
        type=_corpus_type_arg,
        help="语料类型：方位词（别名 localization/location/loc）或 趋向动词（别名 direction/dir/direction_verb）。"
        "决定读取该词性的特征、图式与判定配置，以及 system_prompt/examples 下的示例。"
        "默认方位词；**刻意不按语料文件名推断**——推断猜错时是静默读错一整组表，"
        "宁可要你显式说一句（语料名像另一类时会提醒一句，但不会替你改）。"
        "正式标注前请先运行 python replace 生成该类型的句对",
    )
    parser.add_argument("--corpus", default=None, help="句对 JSONL（pair/synonym 默认 results/location_pairs.jsonl 或 direction_pairs.jsonl）")
    parser.add_argument(
        "--alter-rules",
        default=None,
        help="仅 alter_check 诊断图使用；正式 pair/synonym 标注不替换，须将此参数传给 python replace",
    )
    parser.add_argument("--out-dir", default=None, help="输出目录（默认 results/<图的 prefix>）")
    parser.add_argument("--out-file", default=None, help="直接指定输出文件（默认按时间戳生成）")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出文件。默认**拒绝**：已有同名输出直接报错退出（E1）——"
        "旧行为是不拒绝直接追加，于是历史记录、重复 id 与不同配置的结果会混进同一份文件，"
        "收尾重排后与只统计本轮的 summary 对不上。覆盖前会先清空该文件，不会混入历史记录",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help=f"每个 agent **每条候选**的请求预算（默认 {config.CANDIDATE_MAX_REQUESTS}，取自"
        "config.CANDIDATE_MAX_REQUESTS；0 或负表示不限）。一条语料会展开出多个候选，同一位置的"
        "失败若由每个候选各自重试，重试次数会被候选数放大成整批的倍数；超预算即停，"
        "如实记为失败并转入人工复核（A3）",
    )
    parser.add_argument(
        "--batch-max-requests",
        type=int,
        default=None,
        help=f"整批共用的请求预算（默认 {config.BATCH_MAX_REQUESTS}，取自 config.BATCH_MAX_REQUESTS；"
        "0 或负表示不限）。所有 agent 共用一份计数，达到上限后余下的调用一律不再发出，"
        "记录照常落盘并转入人工复核（A3）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="最多处理多少条两侧已通过检查的句对（过滤后计数）；"
        "可用数量不足时会报告。默认 100",
    )
    parser.add_argument("--offset", type=int, default=0, help="跳过前多少条可用判定语料（过滤后计数）")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=DEFAULT_MODEL, help="model_config/api_keys.json 里的配置名（单模型模式）")
    model_group.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="配置名",
        help="多模型投票模式：指定 model_config/api_keys.json 里的多个配置名。"
        "每个模型独立跑完整张图（各自的 agent、缓存与原始结果），再按 voting_weight "
        "对每条候选做加权简单多数；平票记为 discarded。只给一个名字时等价于 --model。"
        "单个模型失败不会作废整轮：它的票按缺票计入并转人工复核，其余模型照常投票；"
        "只有全部模型都失败才报错。不能与 --no-llm 同用（投票必须有模型判定）",
    )
    parser.add_argument(
        "--model-workers",
        type=int,
        default=config.VOTING_MODEL_WORKERS,
        help=f"多模型投票时同时跑几个模型（默认 {config.VOTING_MODEL_WORKERS}，0 = 全部并行）。"
        "每个模型内部仍按 --workers 并发处理记录，总线程数约为两者相乘",
    )
    parser.add_argument("--workers", type=int, default=1, help="并发处理的记录数（LLM 调用是 IO 密集）")
    parser.add_argument(
        "--batch-inference",
        action="store_true",
        help="仅旧诊断图的句法准备使用；正式 pair/synonym 标注无句法输入，此开关无效",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="不调任何模型：只读取已有句对、查表与落盘；不重新替换或筛选",
    )
    parser.add_argument(
        "--feature-deep",
        action="store_true",
        help="feature_agent 用 deep agent（每类特征一个 subagent、主 agent 汇总）；"
        "默认是单次调用直接产出全部特征字段，一次省下约一个数量级的模型往返。"
        "deep 只在需要对照或单次调用抽得不好时开",
    )
    parser.add_argument(
        "--no-pairs-csv",
        action="store_true",
        help="不把成对信息（原句/替换句/图式/同义判断）另导一份 csv。"
        "默认**导出**到与结果 jsonl 同名、同目录的 .csv，行序与 jsonl 一致（收尾排序后同一份记录写就，"
        "见 ResultWriter.finish），供 Excel 查看；列序见 spatial_tools.PAIR_CSV_COLUMNS。"
        "只有 pair 图有这些成对字段，换别的图本就不导",
    )
    return parser.parse_args(argv)


def _print_graphs() -> None:
    print("可用的图（--graph <名字>）：")
    for spec in sg.GRAPHS.values():
        print(f"  {spec.name:<12}需要 agent：{'、'.join(spec.needs)}")
        print(f"  {' ' * 12}{spec.doc}")
        print(f"  {' ' * 12}产出：results/{spec.prefix}/")


def _warn_corpus_type_mismatch(corpus_path: Path, corpus_type: str) -> None:
    """语料文件名像**另一类**语料时提醒一句（只提醒，不改类型——类型只由 --corpus-type 决定）

    防的是 `--corpus results/direction_corpus.jsonl` 这类旧命令：不给类型就跑，四张表会静默
    按方位词那组读，结果看着像跑完了、口径却全错。这里只在文件名恰好命中另一类的默认语料名
    （含采样前缀 sampled_）时出声，不做任何推断——猜错时同样是静默读错一整组表。
    """
    name = corpus_path.name
    if name.startswith("sampled_"):
        name = name[len("sampled_") :]
    others = {corpus: kind for kind, corpus in st.CORPUS_TYPE_CORPORA.items() if kind != corpus_type}
    other_kind = others.get(name)
    if other_kind:
        print(
            f"警告：语料文件名 {corpus_path.name} 像是{other_kind}语料，但本次类型是{corpus_type}"
            f"（只由 --corpus-type 决定，不按文件名推断）；若确实是{other_kind}语料，"
            f"请加 --corpus-type {other_kind}"
        )


def _config_tables(agents) -> dict[str, str]:
    """本次实际读的配置表 {逻辑表名: 路径}（写进 .summary.json 的凭据）"""
    return {
        report["table"]: report["path"]
        for agent in agents.values()
        if (report := agent.config_report()) is not None
    }


def _print_config_tables(agents) -> None:
    """启动时逐张打印本次跑批读的配置表：路径 + 内容摘要；缺表/空表当场说清楚

    由 agent 自己报（见 spatial_agents.BaseAgent.config_report），不是这里按语料类型算路径：
    --alter-rules 把改写那张表换掉时，只有 agent 知道它实际读了哪张。
    图没建那个 agent 就不读那张表，因此不打印（不是漏印）。

    空表分两级：alter_rules 空 = **警告**（这条跑批跑不出东西，全部停在 alter_missed、
    后面五个 LLM agent 一次都不调）；另三张空 = **提示**（「空表走 LLM 兜底」是各表文件里写明的
    正常态，一律报警就是对正常跑批的误报）。

    表自己报出来的问题（agent 的 warnings）一律按**警告**打印，且与空不空无关：判定表里
    留空的格子就是这类——表填了一半，那些格没人裁定，跑批会照常降级交 LLM，但不该静悄悄过去。
    """
    reports = [report for agent in agents.values() if (report := agent.config_report()) is not None]
    if not reports:
        return
    print("配置表：")
    for report in reports:
        if report["missing"]:
            mark = "（文件不存在，已按空表处理）"
        else:
            mark = ""
        print(f"  {report['table']:<16}{report['path']}    {report['summary']}{mark}")
        for line in report.get("warnings") or []:
            print(f"      警告：{line}")
        if not report["empty"]:
            continue
        level = "警告" if report["table"] == "alter_rules" else "提示"
        # 说「没有已填内容」而不是「为空」：判定表可能只写了表头（标签列了一排、一格没填），
        # 那文件不空、summary 里也有标签数，印「为空」会与同一行自相矛盾
        print(f"      {level}：{report['table']} 没有已填内容 —— {report['degrade']}")
        if report["table"] == "alter_rules":
            print(f"            请把替换规则组填进 {report['path']}，或用 --alter-rules <路径> 指定另一张表")


def _print_agent_stats(stats: dict[str, dict]) -> None:
    """收尾打印各 agent 的模型往返数、失败数、超预算未发数与缓存命中数

    （写进汇总的那份，见 ResultWriter.agent_stats）

    这是花掉的钱的直接口径：往返数含工具往返与结构化输出那次；命中数是被按 message 缓存
    省下的重复问（命中不花往返）。全 0 的 agent 是没走到或根本没调模型（--no-llm）。

    失败数单独列，且**与往返数一样显眼**：往返数只数拿到回复的那些，一个请求都没成功时它是 0，
    而 0 次往返读起来正是「一个请求都没发出去」。整批被接口挡下（403 区域限制、欠费、连不上）
    时真相恰好相反——发是发了，全被拒。所以这里把失败数印在同一行，并在「一次都没成功却失败过」
    时额外把首个失败原因摊开：那种情况下收尾的这几行就是唯一的诊断入口，不该逼人去翻 jsonl。

    超预算未发数（budget_exceeded）与失败数是两件事：后者发出去了但被拒，前者**根本没发**。
    不印出来的话，超预算成片发生时调用账看起来只是「往返少」，看不出是预算掐断的。
    """
    if not stats:
        return
    parts = []
    budget = 0
    for name, item in stats.items():
        part = f"{name} {item['roundtrips']} 次往返"
        if item.get("failed"):
            part += f"／{item['failed']} 次失败"
        if item.get("budget_exceeded"):
            budget += item["budget_exceeded"]
            part += f"／{item['budget_exceeded']} 次超预算未发"
        if item.get("retries"):
            part += f"／{item['retries']} 次重试"
        if item.get("batch_items") or item.get("batch_hits"):
            part += f"／批处理登记 {item.get('batch_items', 0)} 项、回放 {item.get('batch_hits', 0)} 次"
        parts.append(part + f"／{item['cache_hits']} 次命中")
    total = sum(item["roundtrips"] for item in stats.values())
    failed = sum(item.get("failed", 0) for item in stats.values())
    hits = sum(item["cache_hits"] for item in stats.values())
    tail = f"，{failed} 次失败" if failed else ""
    print(f"模型调用账（共 {total} 次往返{tail}，缓存省下 {hits} 次）：" + "；".join(parts))
    if budget:
        print(
            f"提示：有 {budget} 次调用因请求预算未发出，记录里记为 source=error 并转入人工复核；"
            "需要放宽时调 --max-requests（每条候选）或 --batch-max-requests（整批）"
        )
    if total or not failed:
        return
    # 一次往返都没成却冒出失败：整批调用没一个成功的，这是最容易被误读成「没发请求」的形状
    reason = next(
        (item["first_error"] for item in stats.values() if item.get("first_error")),
        "（异常没留下可读的 repr）",
    )
    print(f"警告：本次没有任何一次模型调用成功，{failed} 次调用全部失败（请求已发出，被接口拒绝或没能连上）")
    print(f"      首次失败：{reason}")
    print("      逐条失败原因见各记录的 status_reason；若全部是同一条 403/超时，先查模型可达性再重跑")


def _prefill(records: list[dict], agents, spec: sg.GraphSpec, *, batch: bool = True) -> None:
    """跑批前把整批句子一次性分析好，并把**渲染后的句法证据**存进运行级证据表

    与旧的「只预热 LRU」不同：prepare_syntax 每片推理完立刻渲染入表（见
    spatial_tools.prepare_syntax），所以后面所有 agent/所有候选只读同一份文本：
    - 不会因为 LRU 淘汰而在拼消息时重新推理；
    - 多模型投票时几个模型共用同一份（LTP 与模型无关）；
    - 每个唯一句子只调用一次 LTP（后缀切分内部重跑不算在「按句调用」里）。
    哪些句子会被用到由图声明（spec.plan）。

    ``batch`` 只控制 LTP 一次前向的批大小：True 用 config.LTP_BATCH_CHUNK，
    False 用 chunk_size=1 逐句准备。两种方式都会在 worker 开跑前把证据表填好，
    因为消息侧只读 st.syntax_text，worker 里不再补算 LTP。
    """
    sentences = [s for s in spec.plan(records, agents) if s]
    if not sentences:
        return  # 正式 pair/synonym 标注没有句法输入，不能触发句法预填充
    chunk_size = config.LTP_BATCH_CHUNK if batch else 1
    stat = st.prepare_syntax(sentences, chunk_size=chunk_size)
    mode = "批量" if batch else "逐句"
    print(
        f"LTP {mode}准备：本图需 {stat['sentences']} 句，实际送入模型 {stat['inferred']} 句"
        f"（其余 {stat['cached']} 句已有证据）"
    )
    if stat["failed"]:
        print(f"警告：{stat['failed']} 句的句法证据准备失败，消息里将省略这些句子的证据")
        # 失败原因留在证据表里（spatial_tools.syntax_error），不在这里逐句打印
    missing = [s for s in sentences if s and not st.syntax_ready(s)]
    if missing:
        print(f"警告：{len(missing)} 句没有可用句法证据；worker 不会再补算 LTP（消息里省略该段）")


def _prefill_reality(records: list[dict], agents, spec: sg.GraphSpec, workers: int, *, verbose: bool = True) -> dict:
    """跑批前把闸门要判的原句一次判完（去重），随后节点按 message 读缓存

    与 _prefill 同一套路、同一个理由：同一件判定的重复请求应当在进线程池之前去掉。闸门判的
    只是原句（a 侧那个词），而一条语料按规则表能展开出 9~11 个候选状态、每个状态都要过一遍
    ——不去重就是几倍的调用（实测 1782 条记录只对应 200 个原句，其中 1283 条记录只对应 157 个）。
    注意「原句」这里要读成「原句 + 本位置那个词」：句内两个位置（上@5 与 上@3）同一句原话
    但是两个问题，prefill 按整条 user message 去重，不会把它们并掉。

    哪些原句会走到闸门由图自己声明（spec.gate_queries）：改写未命中的记录不经过闸门，
    也就不在这里判（规则表为空时正是整批都这样，那批一条都不该发）。

    失败不入缓存（见 BaseAgent.CACHEABLE_SOURCES），所以 failed 的那几个会在各自候选上重问——
    打印里如实交代，事后对不上账时能一眼看出差在哪。
    """
    if "reality" not in spec.needs:
        return {}
    stat = agents.need("reality").prefill(spec.gate_queries(records, agents), workers=workers)
    line = (
        f"reality 预填充：本图需判 {stat['unique']} 个原句"
        f"（去重前 {stat['queries']} 个待选状态），实际发出 {stat['asked']} 次"
    )
    if stat.get("reused"):
        line += f"；另有 {stat['reused']} 个状态直接复用记录里的 screening 判定"
    if stat["failed"]:
        line += f"；其中 {stat['failed']} 次失败未入缓存，会在各自候选上重试"
    if stat["skipped"]:
        line += f"；另有 {stat['skipped']} 个状态没有可判的原句/关注词，未发出"
    if verbose:
        print(line)
    return stat


def _validate_run_outputs(
    corpus_path: str | Path,
    out_path: str | Path,
    *,
    csv_columns: tuple[tuple[str, str], ...] = (),
    overwrite: bool = False,
    extra_outputs: tuple[str | Path, ...] = (),
) -> None:
    """在**首次写入之前**把这次跑批要写的几个路径一起查一遍（E1）

    查三件事，任一件不成立就在写盘之前退出：
      1. 输入与任何一个输出（jsonl / 成对 csv / summary / batch sidecar）不能是同一个文件——
         显式 `--out-file` 或 `--batch-cache` 与输入相同时，旧行为会把原始语料截断或混入结果。
      2. 几个输出之间不能互撞：`--out-file result.csv` 时 `with_suffix(".csv")` 换出来还是
         它自己，收尾导 csv 会覆盖刚写好的 jsonl（路径已由 st.pairs_csv_path 错开，这里复查）。
      3. 已有主产物默认拒绝（--overwrite 才允许），且拒绝发生在建文件之前；sidecar 允许复用。

    扩展名也在这里规定：jsonl 结果、csv 视图、summary.json 各归各位，不靠调用方自觉。
    """
    out_path = Path(out_path)
    # 真正的检查在 file_paths.validate_run_outputs：两个跑批入口共用同一份口径
    # 识别相对路径、符号链接与硬链接别名（见 file_paths.same_file_path）
    file_paths.validate_run_outputs(
        corpus_path,
        out_path,
        st.pairs_csv_path(out_path) if csv_columns else None,
        overwrite=overwrite,
        extra_outputs=extra_outputs,
    )


def _build_overrides(args, spec: sg.GraphSpec) -> dict[str, dict]:
    """命令行里会影响 agent 构造的注入项（--alter-rules / --feature-deep）

    与具体模型无关，多模型时每个模型各建一份同口径的 agent。
    """
    overrides: dict[str, dict] = {}
    if args.alter_rules and "alter" in spec.needs:
        overrides["alter"] = {"rules": st.load_alter_rules(args.alter_rules), "rules_path": args.alter_rules}
    if args.feature_deep and "feature" in spec.needs:
        overrides.setdefault("feature", {})["deep"] = True
    return overrides


def _start_agent_log(
    args,
    spec: sg.GraphSpec,
    corpus_type: str,
    corpus_path: Path,
    out_path: Path,
    *,
    model_key: str,
    extra: dict | None = None,
) -> agent_log.AgentLog:
    """为本轮跑批开一份 agent 执行日志（写不了也只是空对象，不影响标注）。

    元数据在跑批开始时就固定下来：图、模型、语料类型、输入输出、并发数、分词口径。
    多模型时 model_key 传合起来的说明，每个模型的原始输出另在 stage_start 事件里记。
    """
    meta = {
        "graph": spec.name,
        "model": model_key,
        "corpus_type": corpus_type,
        "corpus_file": str(corpus_path),
        "output_file": str(out_path),
        "workers": max(1, int(getattr(args, "workers", 1) or 1)),
        "no_llm": bool(getattr(args, "no_llm", False)),
        "schema_version": st.RESULT_SCHEMA_VERSION,
        "segmentation": config.segmentation_signature(),
        "command": " ".join(sys.argv),
        "pid": os.getpid(),
    }
    if extra:
        meta.update(extra)
    return agent_log.AgentLog(run_meta=meta)


def _attach_agent_log(agents, log: agent_log.AgentLog) -> None:
    """把同一份日志挂到一套 agent 上（每个 agent 的节点在 run 完成时写自己的记录）。"""
    for agent in agents.values():
        attach = getattr(agent, "attach_log", None)
        if callable(attach):
            attach(log)


def _build_model_agents(args, spec: sg.GraphSpec, corpus_type: str, model_key: str, overrides: dict[str, dict]):
    """按模型建一套 agent 并预热：配置错在启动时就报，不留给线程池去抢着建"""
    agents = sa.build_agents(
        spec.needs,
        model_key=model_key,
        llm=not args.no_llm,
        overrides=overrides,
        corpus_type=corpus_type,
        # 请求预算（A3）：两条上限各取 config，命令行给了就以命令行为准
        max_requests=config.CANDIDATE_MAX_REQUESTS if args.max_requests is None else args.max_requests,
        batch_max_requests=(
            config.BATCH_MAX_REQUESTS
            if args.batch_max_requests is None
            else args.batch_max_requests
        ),
    )
    for agent in agents.values():
        agent.warm()
    return agents


def _batch_cache_path(args, out_path: str | Path, model_key: str) -> Path:
    """单/多模型共用的 Batch sidecar 路径推导（实际写入与启动预检必须同源）"""
    return bi.resolve_cache_path(
        getattr(args, "batch_cache", None),
        out_path,
        model_key=model_key,
        multi_model=bool(getattr(args, "models", None) and len(args.models) > 1),
    )


def _execute_run(
    args,
    spec: sg.GraphSpec,
    corpus_type: str,
    corpus_path,
    records: list[dict],
    agents,
    out_path: str | Path,
    *,
    model_key: str,
    prefill_ltp: bool = False,
    verbose: bool = True,
    log: agent_log.AgentLog | None = None,
) -> dict:
    """用一套已建好的 agent 跑完整张图并落盘

    单模型与多模型的单模型分支共用这一段。``prefill_ltp`` 只表示**这个分支是否负责**
    准备 LTP 句法证据；批大小由 ``--batch-inference`` 决定。多模型模式下协调层统一
    准备一次，各模型分支传 False。
    ``log`` 是跑批入口建好的 agent 日志；本函数只补两个阶段事件，逐 agent 的记录由图节点写。
    """
    out_path = Path(out_path)
    if log is not None:
        log.event(
            "stage_start",
            stage="annotation",
            model=model_key,
            output_file=str(out_path),
            input_records=len(records),
            workers=max(1, int(args.workers)),
        )
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    writer = ResultWriter(
        out_path,
        counter_names=spec.counter_names,
        tally=spec.tally,
        corpus_type=corpus_type,
        config_tables=_config_tables(agents),
        # 成对信息 csv：默认导出（--no-pairs-csv 关），列序由图给（只有 pair 图有）
        csv_columns=csv_columns,
        model_name=model_key if not args.no_llm else None,
        overwrite=args.overwrite,
        progress_prefix="" if verbose else f"[{model_key}] ",
    )
    try:
        graph = spec.build(agents, writer)
        # 实际生效的请求策略（超时/重试/退避）写进汇总：事后看到某轮特别慢，
        # 能直接判断是模型慢还是重试多，不必翻 config.py 与模型配置去猜
        first_agent = next(iter(agents.values()), None)
        policy = getattr(first_agent, "retry_policy", None)
        if policy:
            writer.extra_summary["request_policy"] = dict(policy)
        if log is not None:
            # 把本轮 agent 日志的路径写进 .summary.json：事后拿到一份结果，能顺着汇总直接找到
            # 它是哪一轮跑出来的、每个 agent 当时怎么判的，不必在 log/ 里按时间猜。
            if getattr(log, "path", None) is not None:
                writer.extra_summary["agent_log_file"] = str(log.path)
            if getattr(log, "jsonl_path", None) is not None:
                writer.extra_summary["agent_log_jsonl"] = str(log.jsonl_path)

        if verbose:
            print(f"输入语料：{corpus_path}（{len(records)} 条）")
            print(f"输出文件：{out_path}")
            print(f"图：{spec.name}（{spec.doc}）")
            print("agent：" + "；".join(agent.describe() for agent in agents.values()))
            # 逐张打印本次实际读的配置表（含 --alter-rules 换过的那张）：缺表/空表当场说清楚。
            # 「表为空 → 一条都不改写却不提示」曾是最容易误读成「代码没接上」的情形，这一块就是
            # 把「配置缺内容」与「代码有问题」分开的地方
            _print_config_tables(agents)
            # 关注词表推得出来吗：推不出来（筛选配置改名、取词表的前缀对不上）就该在这里报，
            # 而不是等每条记录都静默回落成 match_tree[0]
            print("关注词表：" + "；".join(f"{kind} {len(words)} 个" for kind, words in st.load_focus_vocab().items()))
            if args.no_pairs_csv:
                print("成对信息 csv：本次不导出（--no-pairs-csv）")
            elif spec.csv_columns:
                print(f"成对信息 csv：收尾导出到 {st.pairs_csv_path(out_path)}（--no-pairs-csv 可关）")
            else:
                # 非 pair 图的记录里没有成对字段：默认开着也导不出东西，说清楚而不是静悄悄少一个产物
                print(f"提示：本图（{spec.name}）的记录不是成对形状，没有可导出的成对列，本次不导 csv")

        batch_session = None
        if (
            not args.no_llm
            and getattr(args, "batch", "off") == "native"
            and getattr(args, "feature_deep", False)
            and "feature" in spec.needs
        ):
            raise SystemExit(
                "错误：--batch native 不能与 --feature-deep 同用（deep agent 的工具回合无法交给原生 Batch）"
            )
        if not args.no_llm and getattr(args, "batch", "off") != "off":
            profile = bi.batch_profile(model_key)
            if profile.enabled:
                # 多模型投票时每个模型必须各用一份 sidecar，否则会把别的模型的结果回放进来。
                # 推导与启动前路径检查共用 _batch_cache_path，避免两处算出不同路径。
                cache_path = _batch_cache_path(args, out_path, model_key)
                batch_session = bi.BatchSession(
                    profile,
                    cache_path=cache_path,
                    poll_seconds=getattr(args, "batch_poll_seconds", config.BATCH_POLL_SECONDS),
                    timeout_seconds=getattr(args, "batch_timeout_seconds", config.BATCH_TIMEOUT_SECONDS),
                    chunk_size=getattr(args, "batch_chunk_size", config.BATCH_CHUNK_SIZE),
                    retry_policy=policy,
                    resume=bool(getattr(args, "batch_resume", False)),
                    strict_native=getattr(args, "batch", "auto") == "native",
                )
                if verbose:
                    print(f"批处理：{profile.describe()}")
                    if getattr(args, "feature_deep", False) and "feature" in spec.needs:
                        print("批处理：--feature-deep 的特征子 agent 只能同步回退（原生 Batch 跑不了工具回合）")
            elif getattr(args, "batch", "auto") == "native":
                raise SystemExit(f"错误：--batch native 不可用 —— {profile.describe()}")
            elif verbose:
                print(f"批处理：{profile.describe()}（回退同步执行）")

        if not args.no_llm:
            if prefill_ltp:
                # 句法证据必须在 worker 前准备好；--batch-inference 只决定 LTP 批大小。
                _prefill(records, agents, spec, batch=args.batch_inference)
            if batch_session is None:
                # 闸门的原句在这里一次判完（去重）。图已经建好——接线错在花钱之前就报出来，于是这份
                # 去重账补写到 writer 上，与语料类型、配置表一并进汇总（见 ResultWriter.reality_prefill）
                writer.reality_prefill = _prefill_reality(records, agents, spec, args.workers, verbose=verbose)
            elif verbose and "reality" in spec.needs:
                print("reality 预填充：批处理模式下由第一波批量请求统一提交（按 user message 去重）")

        # 一条语料的每个目标词位置各能改出好几个待选句，每个待选句一个状态、一条记录，
        # 所以状态数一般多于语料条数
        states = spec.init_states(records, agents, str(out_path))
        if verbose:
            print(f"处理状态：{len(records)} 条输入 → {len(states)} 个状态（正式标注不再展开句对）")
        # 转化链的两个分母（D5）补写到 writer 上、进汇总（与 reality_prefill / agent_stats 同一套路）
        writer.input_records = len(records)
        writer.expanded_states = len(states)
        started = time.time()

        if batch_session is not None:
            if verbose:
                print(f"批处理：{len(states)} 个状态进入波次执行（原生优先，失败回退同步）")
            # 批量波次会把同一条候选重放多次；候选预算必须跨波次保留同一份，
            # 否则每条候选每波都会拿到一份新预算，max_requests 形同虚设。
            scopes = {id(state): agents.new_budget_scope() for state in states}

            def run_batch_state(state):
                with agents.use_budget_scope(scopes[id(state)]):
                    return graph.invoke(state, {"recursion_limit": 60})

            bi.run_waves(
                states,
                run_batch_state,
                batch_session,
                fallback_workers=max(1, args.workers),
            )
            writer.extra_summary["batch"] = batch_session.summary()
            if verbose:
                batch_summary = batch_session.summary()
                print(
                    f"批处理账：{batch_summary['waves']} 波／{batch_summary['items_collected']} 项"
                    f"／成功 {batch_summary['items_completed']}／失败 {batch_summary['items_failed']}"
                    f"／回退同步 {batch_summary['fallback_items']}／缓存回放 {batch_summary['cache_hits']}"
                )
        else:

            def run_state(state):
                # 一条状态 = 一条候选，候选预算在这个作用域内累计（见 AgentSet.budget_scope）；
                # 作用域按线程隔离，--workers N 下各条候选互不串账
                with agents.budget_scope():
                    return graph.invoke(state, {"recursion_limit": 60})

            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                # 结果已在 save 节点落盘，这里只需驱动跑完；map 按输入顺序产出，异常会在此处抛出
                for _ in pool.map(run_state, states):
                    pass

        # 各 agent 的调用账（往返数、重试数与缓存命中数）：跑完才齐，收尾补写到 writer 上、进汇总
        writer.agent_stats = {name: agent.stats() for name, agent in agents.items()}
        summary = writer.finish()
        if verbose:
            print(
                f"\n完成 {summary['count']} 条记录（{len(records)} 条语料），用时 {summary['elapsed_seconds']} 秒"
                f"（总耗时 {round(time.time() - started, 1)} 秒）"
            )
            for name in spec.counter_names:
                print(f"{name}：{summary[name]}")
            _print_admission(summary)
            _print_agent_stats(writer.agent_stats)
            print(f"汇总文件：{summary['summary_file']}")
        if log is not None:
            log.event(
                "stage_end",
                stage="annotation",
                model=model_key,
                output_file=str(out_path),
                count=summary.get("count", 0),
                elapsed_seconds=summary.get("elapsed_seconds", 0),
            )
        return {"summary": summary, "writer": writer, "states": states}
    finally:
        writer._close()


def _run_single_model(args, spec, corpus_type, corpus_path, records, out_path: Path, model_key: str) -> dict:
    """原有单模型路径：行为、输出文件与汇总形状保持不变"""
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    cache_path = _batch_cache_path(args, out_path, model_key)
    sidecars = (
        bi.sidecar_paths(cache_path)
        if not args.no_llm and getattr(args, "batch", "off") != "off"
        else ()
    )
    # 先做无副作用的路径检查，再建模型/写任何文件：参数错了不该先花启动成本，
    # 更不能让 BatchSession 在检查前截断恰好被 --batch-cache 指到的语料。
    _validate_run_outputs(
        corpus_path,
        out_path,
        csv_columns=csv_columns,
        overwrite=args.overwrite,
        extra_outputs=sidecars,
    )
    overrides = _build_overrides(args, spec)
    agents = _build_model_agents(args, spec, corpus_type, model_key, overrides)
    log = _start_agent_log(args, spec, corpus_type, corpus_path, out_path, model_key=model_key)
    result = None
    try:
        _attach_agent_log(agents, log)
        log.agents_ready(agents)
        result = _execute_run(
            args,
            spec,
            corpus_type,
            corpus_path,
            records,
            agents,
            out_path,
            model_key=model_key,
            # 这个分支允许预填充；是否真的执行由 args.batch_inference 决定（见 _execute_run）。
            prefill_ltp=True,
            verbose=True,
            log=log,
        )
        return result
    except BaseException as exc:
        log.event("run_error", stage="single", error=repr(exc))
        raise
    finally:
        stats = result["writer"].agent_stats if result is not None else {}
        summary = log.finish(agent_stats=stats)
        if summary.get("log_file"):
            print(f"[Agent日志] 已写入 {summary['log_file']}（同名 .jsonl 为机器可读版本）")


def _run_multi_model(
    args,
    spec,
    corpus_type,
    corpus_path,
    records: list[dict],
    out_path: Path,
    model_names: list[str],
    weights: dict[str, int],
) -> dict:
    """多模型投票的日志外壳：建一份共享日志，跑完（或失败）都把汇总收尾。

    每个模型的原始输出、投票结果都写在同一份 agent-log 里，记录用 model 字段与
    stage_start 的 output_file 区分；这样一次 --models 只会留下一个时间戳、两份文件，
    不必在 log/ 里按模型再拼文件名。
    """
    log = _start_agent_log(
        args,
        spec,
        corpus_type,
        corpus_path,
        Path(out_path),
        model_key="、".join(model_names),
        extra={"models": list(model_names), "voting_weights": dict(weights)},
    )
    results: dict[str, dict] = {}
    try:
        return _run_multi_model_impl(
            args,
            spec,
            corpus_type,
            corpus_path,
            records,
            out_path,
            model_names,
            weights,
            log=log,
            results=results,
        )
    except BaseException as exc:
        log.event("run_error", stage="multi", error=repr(exc))
        raise
    finally:
        merged: dict[str, dict] = {}
        for name, result in results.items():
            writer = result.get("writer")
            for agent_name, item in ((getattr(writer, "agent_stats", None) or {}).items()):
                merged[f"{name}.{agent_name}"] = item
        summary = log.finish(agent_stats=merged)
        if summary.get("log_file"):
            print(f"[Agent日志] 已写入 {summary['log_file']}（同名 .jsonl 为机器可读版本）")


def _model_summary_entry(
    name: str,
    weights: dict[str, int],
    raw_paths: dict[str, Path],
    results: dict[str, dict],
    failed: dict[str, BaseException],
    model_records: dict[str, list[dict]],
) -> dict:
    """汇总里每个模型一条：跑成功的照旧，失败的标 status=failed 并留下错误与已参与票数

    失败模型也占一条，是为了让事后拿到 .summary.json 的人能一眼区分
    “这个模型没票”是它整轮都没跑起来，还是它跑了但一直弃权。
    """
    result = results.get(name)
    if result is not None:
        writer = result["writer"]
        return {
            "name": name,
            "weight": weights[name],
            "output_file": str(raw_paths[name]),
            "status": "ok",
            "count": result["summary"]["count"],
            "elapsed_seconds": result["summary"]["elapsed_seconds"],
            "agent_stats": writer.agent_stats,
            "reality_prefill": writer.reality_prefill,
            "request_policy": writer.extra_summary.get("request_policy"),
            "batch": writer.extra_summary.get("batch"),
        }
    exc = failed.get(name)
    return {
        "name": name,
        "weight": weights[name],
        "output_file": str(raw_paths[name]),
        "status": "failed",
        "error": repr(exc) if exc is not None else "未运行",
        "count": len(model_records.get(name) or []),
    }


def _run_multi_model_impl(
    args,
    spec,
    corpus_type,
    corpus_path,
    records: list[dict],
    out_path: Path,
    model_names: list[str],
    weights: dict[str, int],
    *,
    log: agent_log.AgentLog,
    results: dict[str, dict],
) -> dict:
    """多模型投票：每个模型独立跑完整张图，再逐候选加权投票

    并行边界在**模型**这一层：模型之间互不共享 agent、缓存与 writer；
    每个模型内部仍按 --workers 并发跑候选。原始结果各写自己的文件，
    最终投票结果写用户 --out-file 指定的那一个，下游读取方式与单模型一致。

    **单个模型失败不作废整轮**：失败模型已落盘的记录照常参与投票，没落盘的按缺票
    计（见 voting.aggregate_votes 的 missing_models / incomplete），受影响记录转人工
    复核；只有全部模型都失败、没有可投的票时才抛错。
    """
    if args.model_workers < 0:
        raise SystemExit("错误：--model-workers 不能为负数")
    model_workers = args.model_workers or len(model_names)
    raw_dir, raw_paths = voting.model_output_paths(out_path, model_names)
    csv_columns = spec.csv_columns if not args.no_pairs_csv else ()
    voting_csv_columns = csv_columns
    if csv_columns:
        # 最终 csv 在原有的成对列后追加投票审计列；原始各模型的 csv 仍用原列序
        voting_csv_columns = csv_columns + (
            ("投票状态", "voting.status"),
            ("获胜标签", "voting.winner_label"),
            ("获胜模型", "voting.winner_model"),
            ("有效权重", "voting.valid_weight"),
            ("配置权重", "voting.configured_weight"),
            ("缺票模型", "voting.missing_models"),
        )

    # 所有输出（最终投票文件 + 各模型原始文件及其 csv/summary）放进同一张清单，
    # 一次检查输入/输出别名与已有文件：分开逐组检查看不出“最终文件”和某个原始文件
    # 其实是同一个硬链接/软链接的情况。
    main_outputs: list[Path] = [out_path]
    if csv_columns:
        main_outputs.append(st.pairs_csv_path(out_path))
    main_outputs.append(out_path.with_suffix(".summary.json"))
    for raw_path in raw_paths.values():
        main_outputs.append(raw_path)
        if csv_columns:
            main_outputs.append(st.pairs_csv_path(raw_path))
        main_outputs.append(raw_path.with_suffix(".summary.json"))
    sidecars: list[Path] = []
    if not args.no_llm and getattr(args, "batch", "off") != "off":
        for name in model_names:
            # 执行阶段 _execute_run 拿到的是 raw_paths[name]，不是最终投票 out_path；
            # 默认 sidecar 跟着实际输出文件走，预检必须用同一个基路径。
            cache_path = _batch_cache_path(args, raw_paths[name], name)
            sidecars.extend(bi.sidecar_paths(cache_path))
    # sidecar 允许已存在（--batch-resume 要复用），但绝不能与语料或任一主产物同路径：
    # 否则每个模型建会话时会截断别人的结果/原始语料。
    file_paths.validate_output_paths([corpus_path], main_outputs + sidecars)
    if not args.overwrite:
        for output in main_outputs:
            if output.exists():
                raise ValueError(
                    f"输出文件 {output} 已存在（默认拒绝覆盖，免得混入历史记录与重复 id）；"
                    "请换一个输出路径，或显式加 --overwrite 覆盖它"
                )

    print(f"多模型投票：{len(model_names)} 个模型（权重 " + "，".join(f"{n}={weights[n]}" for n in model_names) + "）")
    print(f"最终输出：{out_path}")
    print(f"各模型原始结果：{raw_dir}/")
    print(f"模型并发：{model_workers}；每个模型内部 --workers={max(1, args.workers)}")

    overrides = _build_overrides(args, spec)
    agents_by_model = {}
    for name in model_names:
        print(f"  准备模型 {name}（voting_weight={weights[name]}）")
        agents_by_model[name] = _build_model_agents(args, spec, corpus_type, name, overrides)
        _attach_agent_log(agents_by_model[name], log)

    # 清单按「模型.agent」命名，同一个 agent 名在不同模型下各有一条，事后能看出
    # 这一轮到底建了几套、各自读哪份提示语与配置表。
    log.agents_ready(
        {
            f"{model_name}.{agent_name}": agent
            for model_name, model_agents in agents_by_model.items()
            for agent_name, agent in model_agents.items()
        }
    )

    first_agents = agents_by_model[model_names[0]]
    _print_config_tables(first_agents)
    print("关注词表：" + "；".join(f"{kind} {len(words)} 个" for kind, words in st.load_focus_vocab().items()))
    if args.no_pairs_csv:
        print("成对信息 csv：本次不导出（--no-pairs-csv）")
    elif spec.csv_columns:
        print(f"成对信息 csv：收尾导出到 {st.pairs_csv_path(out_path)}（--no-pairs-csv 可关）")
    else:
        print(f"提示：本图（{spec.name}）的记录不是成对形状，没有可导出的成对列，本次不导 csv")

    # LTP 是模型无关的：整批只准备一次，避免几个模型线程重复做同一批句法分析。
    # 证据必须在模型线程开跑前准备好（消息侧只读 st.syntax_text）；--batch-inference
    # 只决定 LTP 一次前向的批大小，关闭时逐句准备（chunk_size=1）。
    if not args.no_llm:
        _prefill(records, first_agents, spec, batch=args.batch_inference)

    started = time.time()
    results.clear()
    errors: list[tuple[str, BaseException]] = []
    log.event("voting_start", models=list(model_names), weights=dict(weights))
    with ThreadPoolExecutor(max_workers=max(1, model_workers)) as pool:
        futures = {
            pool.submit(
                _execute_run,
                args,
                spec,
                corpus_type,
                corpus_path,
                records,
                agents_by_model[name],
                raw_paths[name],
                model_key=name,
                prefill_ltp=False,
                verbose=False,
                log=log,
            ): name
            for name in model_names
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
                print(f"[{name}] 完成 {results[name]['summary']['count']} 条，"
                      f"用时 {results[name]['summary']['elapsed_seconds']} 秒")
            except BaseException as exc:  # 先收集，等其余模型收尾后再抛
                errors.append((name, exc))
                print(f"[{name}] 失败：{exc!r}")
    # 单个模型失败不再作废整轮：它的票按“缺票”记在 voting.missing_models 上，
    # 其余模型照常投票，受影响的记录由 aggregate_votes 标 incomplete 并转人工复核。
    # 只有**全部**模型都失败时才没有可投的票，那时才报错。
    failed: dict[str, BaseException] = dict(errors)
    if failed:
        if len(failed) == len(model_names):
            first_name, first_exc = errors[0]
            raise RuntimeError(
                f"多模型投票中全部模型运行失败（首个：{first_name}）：{first_exc!r}"
            ) from first_exc
        print(
            f"警告：{len(failed)}/{len(model_names)} 个模型失败，按缺票继续投票 —— "
            + "；".join(f"{name}: {exc!r}" for name, exc in errors)
        )
        log.event(
            "model_failed",
            models=list(failed),
            errors={name: repr(exc) for name, exc in errors},
            note="失败模型按缺票参与投票，其余模型继续",
        )

    model_records: dict[str, list[dict]] = {}
    for name in model_names:
        path = raw_paths[name]
        if not path.exists():
            if name in failed:
                # 还没开始落盘就失败了：整份缺票，aggregate_votes 会把它记进 missing_models
                continue
            raise RuntimeError(f"模型 {name} 没有产出结果文件：{path}")
        try:
            model_records[name] = st.read_jsonl(path)
        except (OSError, json.JSONDecodeError) as exc:
            if name in failed:
                # 失败那次的收尾可能留下一行半截记录：读不全就整份按缺票处理，
                # 不能因为一行坏数据把其余四个模型的成果一起扔掉
                print(f"警告：模型 {name} 的部分结果读不出来（{exc!r}），整份按缺票处理")
                continue
            raise
        if name in failed:
            print(f"模型 {name} 失败前已落盘 {len(model_records[name])} 条，按现有记录参与投票")

    final_records = voting.combine_model_records(
        spec.name,
        model_records,
        weights,
        model_names,
        output_file=out_path,
    )
    if not final_records:
        print("警告：所有模型都没有产出可投票的记录，未生成最终结果")
        log.event(
            "voting_end",
            status="empty",
            models=list(model_names),
            **({"failed_models": {name: repr(exc) for name, exc in failed.items()}} if failed else {}),
        )
        return {}

    # 状态数取第一个**跑成功**的模型：失败模型的 results 条目不存在（也可能恰好是 model_names[0]）
    first_states = next((results[name]["states"] for name in model_names if name in results), [])
    voting_writer = ResultWriter(
        out_path,
        counter_names=spec.counter_names,
        tally=spec.tally,
        corpus_type=corpus_type,
        config_tables=_config_tables(first_agents),
        csv_columns=voting_csv_columns,
        model_name="voting",
        overwrite=args.overwrite,
        input_records=len(records),
        expanded_states=len(first_states),
        progress_prefix="[voting] ",
    )
    voting_writer.extra_summary = {
        "voting": voting.summarize_voting(final_records),
        "model_names": list(model_names),
        "voting_weights": dict(weights),
        **({"failed_models": {name: repr(exc) for name, exc in failed.items()}} if failed else {}),
        **(
            {"agent_log_file": str(log.path), "agent_log_jsonl": str(log.jsonl_path)}
            if log.path is not None and log.jsonl_path is not None
            else {}
        ),
        # 失败模型也留一条（status=failed + error）：事后看汇总能直接分辨
        # “这个模型没票”是设计如此，还是它那一轮炸了
        "models": [
            _model_summary_entry(name, weights, raw_paths, results, failed, model_records)
            for name in model_names
        ],
    }
    for record in final_records:
        voting_writer.write(record)
    summary = voting_writer.finish()
    log.event(
        "voting_end",
        status="ok",
        models=list(model_names),
        count=summary.get("count", 0),
        voting=summary.get("voting"),
        output_file=str(out_path),
    )

    print(
        f"\n完成 {summary['count']} 条最终记录（{len(records)} 条语料），"
        f"用时 {summary['elapsed_seconds']} 秒（总耗时 {round(time.time() - started, 1)} 秒）"
    )
    for name in model_names:
        result = results.get(name)
        if result is None:
            print(
                f"  模型 {name}（权重 {weights[name]}）：运行失败，按缺票处理"
                f"（已落盘 {len(model_records.get(name) or [])} 条参与投票）"
            )
            continue
        stats = result["writer"].agent_stats
        roundtrips = sum(item.get("roundtrips", 0) for item in stats.values())
        call_failed = sum(item.get("failed", 0) for item in stats.values())
        retries = sum(item.get("retries", 0) for item in stats.values())
        print(
            f"  模型 {name}（权重 {weights[name]}）：{roundtrips} 次往返"
            + (f"／{call_failed} 次失败" if call_failed else "")
            + (f"／{retries} 次重试" if retries else "")
        )
    vote_stats = summary["voting"]
    print(
        f"投票：采纳 {vote_stats['accepted']} 条，舍弃 {vote_stats['discarded']} 条"
        f"（平票 {vote_stats['tie']}，无有效票 {vote_stats['no_valid_votes']}）；"
        f"不完整 {vote_stats['incomplete']} 条"
    )
    if vote_stats["labels"]:
        print("  采纳标签：" + "，".join(f"{k} {v}" for k, v in vote_stats["labels"].items()))
    if vote_stats["missing_by_model"]:
        print("  缺票模型：" + "，".join(f"{k} {v} 条" for k, v in vote_stats["missing_by_model"].items()))
    for name in spec.counter_names:
        print(f"{name}：{summary[name]}")
    _print_admission(summary)
    print(f"汇总文件：{summary['summary_file']}")
    return summary


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.list_graphs:
        _print_graphs()
        return

    selected_models = list(args.models) if args.models else [args.model]
    if len(set(selected_models)) != len(selected_models):
        raise SystemExit("错误：--models 里有重复的模型配置名")
    multi_model = len(selected_models) > 1
    if args.models and not multi_model:
        print(f"提示：--models 只给了一个模型（{selected_models[0]}），按单模型模式运行")
    if multi_model and args.no_llm:
        raise SystemExit("错误：--models 多模型投票必须有模型判定，不能与 --no-llm 同用")

    # 权重在花钱之前校验完：写错一个字段就当场退出，而不是跑完几个模型才发现权重没生效。
    # 单模型不读模型配置：--no-llm 的用途之一就是在没有 api_keys.json 的机器上验纯规则，
    # 这里不能因为一个用不到的投票权重把那条路堵死。
    weights: dict[str, int] = {}
    if multi_model:
        for name in selected_models:
            cfg = agent_factory.load_model_config(name)
            weights[name] = voting.normalize_weight(
                cfg.get(voting.VOTING_WEIGHT_FIELD), model_name=name
            )

    # args.corpus_type 已被 argparse 规范化（None 仍是 None），这里才落默认值：
    # 落到 None 上就分不清「没指定」与「显式指定方位词」，启动打印要说出类型是哪来的
    corpus_type = st.normalize_corpus_type(args.corpus_type)
    corpus_path = (
        Path(args.corpus)
        if args.corpus
        else _PROJECT_ROOT / "results" / st.CORPUS_TYPE_CORPORA[corpus_type]
    )
    print(f"语料类型：{corpus_type}（{'--corpus-type' if args.corpus_type else '默认'}）")
    print(f"配置目录：config_files/{st.CORPUS_TYPE_DIRS[corpus_type]}/")
    _warn_corpus_type_mismatch(corpus_path, corpus_type)

    spec = sg.get_graph(args.graph)
    if spec.name in ("pair", "synonym"):
        if not args.corpus:
            corpus_path = _PROJECT_ROOT / "results" / cp.PAIR_CORPORA[corpus_type]
        if args.alter_rules:
            raise SystemExit("错误：标注阶段不再替换句子；请将 --alter-rules 传给 python replace")
        records = cp.read_prepared_pairs(corpus_path, limit=args.limit, offset=args.offset)
        if any(record.get("corpus_type") != corpus_type for record in records):
            raise ValueError("句对的 corpus_type 与 --corpus-type 不一致，不能使用另一词性的标注配置")
    else:
        records = st.read_corpus(corpus_path, limit=args.limit, offset=args.offset, judge_only=True)
    if args.limit is not None and len(records) < args.limit:
        print(
            f"警告：可用判定语料只有 {len(records)} 条，少于计数限制 {args.limit} 条；"
            "已按实际数量运行"
            + (f"（已跳过 offset={args.offset} 条可用语料）" if args.offset else "")
        )
    if not records:
        print(f"没有从 {corpus_path} 读到可用于判定的记录")
        return

    if args.feature_deep and "feature" not in spec.needs:
        # 图不建 feature agent，这个开关就无从生效；静默忽略会让人以为 deep 跑过了
        print(f"提示：--feature-deep 对本图（{spec.name}）无效——它不建 feature agent")

    out_dir = Path(args.out_dir) if args.out_dir else _PROJECT_ROOT / "results" / spec.prefix
    out_path = Path(args.out_file) if args.out_file else st.build_output_path(out_dir, prefix=spec.prefix)

    if multi_model:
        _run_multi_model(args, spec, corpus_type, corpus_path, records, out_path, selected_models, weights)
    else:
        _run_single_model(args, spec, corpus_type, corpus_path, records, out_path, selected_models[0])


def _print_admission(summary: dict) -> None:
    """收尾把**处理成功率**与**数据准入率**并列打出来（A1）

    status_counts 说的是流程有没有跑完，而闸门判不出来时流程照样跑得完——那批记录会得到
    status=ok、空间性却根本没确认。两个数分开报，才不会把「跑完了」当成「能用了」。
    哪些维度卡住（reality/fluency/features/schema/judgement）也一并列出，
    review_required 那一行就是下一轮人工复核的入口。
    """
    admission = summary.get("admission_counts")
    if not admission:
        return
    total = sum(admission.get(key, 0) for key in ("admitted", "not_admitted"))
    if not total:
        return
    rate = admission.get("admitted", 0) / total
    print(f"数据准入：{admission.get('admitted', 0)}/{total} 条（{rate:.1%}）；"
          f"待人工复核 {admission.get('review_required', 0)} 条")
    for name in ("reality", "fluency", "features", "schema", "judgement"):
        counter = summary.get(f"{name}_admission")
        if counter:
            print(f"    {name}：" + "，".join(f"{key} {value}" for key, value in sorted(counter.items())))


if __name__ == "__main__":
    main()
