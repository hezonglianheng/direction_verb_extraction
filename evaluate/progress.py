# encoding: utf8

"""evaluate 的运行进度显示。

每个正在处理的模型一条进度：分母是去重后的逻辑请求数（与报告
``unique_requests`` 同口径），不是选中样本数。100% 表示每个请求都已拿到
预测或明确错误，不表示全部成功，也不表示报告已经写出。

TTY 下用 tqdm 动态条，多模型并行时各占一行、结束后释放位置；重定向到
文件 / 管道 / 测试的 ``StringIO`` 时退化为带模型前缀的整行里程碑
（0%、25%、50%、75%、100%），避免控制字符和刷屏。

进度是附属输出：任何显示异常都只关闭后续进度，不得中断评测、改变报告
或退出码。
"""

from __future__ import annotations

import os
import sys
import threading

try:  # requirements.txt 已固定 tqdm；缺失时只关闭动态条，不影响评测。
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

_MILESTONES = (25, 50, 75, 100)


if tqdm is not None:
    class _UnlockedTqdm(tqdm):
        """tqdm 4.67.3 的 ``refresh()`` 在 ``display()`` 抛异常时不会释放自己
        获取的全局写锁；多模型并行时会让其他模型永久阻塞。

        本类自己用 ``with self._lock`` 取得并释放锁，再让基类以 ``nolock=True``
        只负责绘制：跨线程（包括 tqdm 监控线程）仍然互斥，异常路径由上下文
        管理器保证释放，不会复现基类“先取锁、display()、再释放”的漏锁。
        """

        def refresh(self, nolock=False, lock_args=None):
            if nolock:
                return super().refresh(nolock=True)
            with self._lock:
                return super().refresh(nolock=True)

    # 子类的 get_lock() 默认会新建自己的锁；统一复用基类锁，确保监控线程、
    # tqdm.write() 和动态条刷新互相排斥。
    _UnlockedTqdm._lock = tqdm.get_lock()
else:  # pragma: no cover
    _UnlockedTqdm = None


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


class _SafeStream:
    """包住真实输出流：写入 / 刷新异常只标记断流，绝不让 tqdm 在持锁时抛出。"""

    def __init__(self, stream, manager: "ProgressManager"):
        self._stream = stream
        self._manager = manager

    def write(self, text):
        try:
            return self._stream.write(text)
        except Exception:
            self._manager._mark_broken()
            return len(text) if isinstance(text, str) else 0

    def flush(self):
        try:
            return self._stream.flush()
        except Exception:
            self._manager._mark_broken()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class ProgressManager:
    """进程内共享的进度显示与日志写入器，供多模型线程安全复用。"""

    def __init__(self, stream=None, *, enabled=True, interactive=None, bar_factory=None):
        self.stream = sys.stdout if stream is None else stream
        self.enabled = bool(enabled)
        self.broken = False
        detected = _is_tty(self.stream) if interactive is None else bool(interactive)
        self.interactive = bool(detected and (tqdm is not None or bar_factory is not None))
        self._bar_factory = bar_factory
        self._lock = threading.RLock()
        self._safe_stream = _SafeStream(self.stream, self)
        self._free_positions: list[int] = []
        self._next_position = 0
        self._live_tqdm = 0

    def start(self, model_key: str) -> "_ModelProgress":
        return _ModelProgress(self, str(model_key))

    def _mark_broken(self) -> None:
        self.broken = True
        self.enabled = False

    def write(self, message: str) -> None:
        """写一行诊断 / 里程碑；有动态条时先让 tqdm 清屏再写。"""
        with self._lock:
            try:
                if self.interactive and self._live_tqdm and tqdm is not None:
                    tqdm.write(message, file=self._safe_stream)
                else:
                    print(message, file=self._safe_stream, flush=True)
            except Exception:
                # 断管、流关闭或编码不支持：停用后续进度，但不改变评测结果。
                self._mark_broken()
                self.interactive = False

    def finalize(self) -> None:
        """解释器退出前处理已断开的 stdout。

        CPython 退出时会再次 flush ``sys.stdout``；若下游已关闭管道，二次
        flush 失败会把进程退出码改成 120。这里先把剩余缓冲导向 devnull，
        使 ``main()`` 的业务退出码保持不变。
        """
        with self._lock:
            # 只处理 CLI 自己管理的 stdout：库调用方注入的流可能有其他写入者，
            # 不能把共享 fd 静默改到 devnull。
            if not self.broken or self.stream is not sys.stdout:
                return
            try:
                self.stream.flush()
                return
            except Exception:
                pass
            try:
                devnull = os.open(os.devnull, os.O_WRONLY)
                try:
                    os.dup2(devnull, self.stream.fileno())
                finally:
                    os.close(devnull)
            except Exception:
                pass

    def _claim_position(self) -> int:
        if self._free_positions:
            return self._free_positions.pop(0)
        position = self._next_position
        self._next_position += 1
        return position

    def _release_position(self, position: int | None) -> None:
        if position is not None:
            self._free_positions.append(position)
            self._free_positions.sort()

    def _new_bar(self, total: int, position: int, model_key: str):
        desc = f"[{model_key}] 唯一请求"
        if self._bar_factory is not None:
            return self._bar_factory(total=total, desc=desc, position=position)
        return _UnlockedTqdm(
            total=total, desc=desc, position=position, leave=False,
            file=self._safe_stream, dynamic_ncols=True, mininterval=0.5,
        )


class _ModelProgress:
    """单个模型的进度句柄；update / close 只由持有该模型的线程调用。"""

    def __init__(self, manager: ProgressManager, model_key: str):
        self.manager = manager
        self.model_key = model_key
        self._total = 0
        self._done = 0
        self._last_done = -1
        self._last_milestone = -1
        self._started = False
        self._closed = False
        self._bar = None
        self._position: int | None = None
        self._live_tqdm = False

    def update(self, done, total) -> None:
        try:
            self._update(int(done), int(total))
        except Exception:
            self._disable()

    def _update(self, done: int, total: int) -> None:
        if self._closed or total <= 0:
            return
        manager = self.manager
        with manager._lock:
            if not self._started:
                self._started = True
                self._total = total
                if manager.enabled and manager.interactive:
                    self._position = manager._claim_position()
                    try:
                        self._bar = manager._new_bar(total, self._position, self.model_key)
                    except Exception:
                        manager._release_position(self._position)
                        self._position = None
                        manager.enabled = False
                        return
                    self._live_tqdm = manager._bar_factory is None
                    if self._live_tqdm:
                        manager._live_tqdm += 1
                elif manager.enabled and done == 0:
                    self._emit(0, total)
            done = max(0, min(done, self._total))
            if done <= self._last_done:
                return
            previous = max(self._last_done, 0)
            self._last_done = done
            self._done = done
            if not manager.enabled:
                return
            if self._bar is not None:
                self._bar.update(done - previous)
            else:
                self._emit(done, self._total)

    def _emit(self, done: int, total: int) -> None:
        percentage = min(100, (done * 100) // total)
        milestone = max((value for value in _MILESTONES if value <= percentage), default=0)
        if milestone <= self._last_milestone:
            return
        self._last_milestone = milestone
        self.manager.write(f"[{self.model_key}] 进度 {done}/{total}（{percentage}%）")

    def close(self) -> None:
        manager = self.manager
        with manager._lock:
            if self._closed:
                return
            self._closed = True
            if self._bar is not None:
                completed = self._total > 0 and self._done >= self._total
                self._discard_bar()
                # 小任务可能在 mininterval 内结束；leave=False 会清掉动态条，
                # 这里留一行终值，避免终端用户只看到 0%。
                if completed and manager.enabled:
                    manager.write(
                        f"[{self.model_key}] 进度 {self._done}/{self._total}（100%）")
            elif self._started and manager.enabled and not manager.interactive:
                total, done = self._total, self._done
                if total > 0 and done < total:
                    manager.write(
                        f"[{self.model_key}] 进度 {done}/{total}（{done * 100 // total}%），未完成")

    def _disable(self) -> None:
        with self.manager._lock:
            self.manager.enabled = False
            if self._bar is not None:
                self._discard_bar()

    def _discard_bar(self) -> None:
        try:
            self._bar.close()
        except Exception:
            pass
        if self._live_tqdm:
            self.manager._live_tqdm = max(0, self.manager._live_tqdm - 1)
        self.manager._release_position(self._position)
        self._bar = None
        self._position = None
