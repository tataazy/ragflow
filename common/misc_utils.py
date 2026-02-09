#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import asyncio
import base64
import functools
import hashlib
import logging
import os
import subprocess
import sys
import threading
import uuid
import multiprocessing

from concurrent.futures import ThreadPoolExecutor

import requests

def get_uuid():
    return uuid.uuid1().hex


def download_img(url):
    if not url:
        return ""
    response = requests.get(url)
    return "data:" + \
        response.headers.get('Content-Type', 'image/jpg') + ";" + \
        "base64," + base64.b64encode(response.content).decode("utf-8")


def hash_str2int(line: str, mod: int = 10 ** 8) -> int:
    return int(hashlib.sha1(line.encode("utf-8")).hexdigest(), 16) % mod

def convert_bytes(size_in_bytes: int) -> str:
    """
    Format size in bytes.
    """
    if size_in_bytes == 0:
        return "0 B"

    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    size = float(size_in_bytes)

    while size >= 1024 and i < len(units) - 1:
        size /= 1024
        i += 1

    if i == 0 or size >= 100:
        return f"{size:.0f} {units[i]}"
    elif size >= 10:
        return f"{size:.1f} {units[i]}"
    else:
        return f"{size:.2f} {units[i]}"


def once(func):
    """
    A thread-safe decorator that ensures the decorated function runs exactly once,
    caching and returning its result for all subsequent calls. This prevents
    race conditions in multi-thread environments by using a lock to protect
    the execution state.

    Args:
        func (callable): The function to be executed only once.

    Returns:
        callable: A wrapper function that executes `func` on the first call
                  and returns the cached result thereafter.

    Example:
        @once
        def compute_expensive_value():
            print("Computing...")
            return 42

        # First call: executes and prints
        # Subsequent calls: return 42 without executing
    """
    executed = False
    result = None
    lock = threading.Lock()
    def wrapper(*args, **kwargs):
        nonlocal executed, result
        with lock:
            if not executed:
                executed = True
                result = func(*args, **kwargs)
        return result
    return wrapper

@once
def pip_install_torch():
    device = os.getenv("DEVICE", "cpu")
    if device=="cpu":
        return
    logging.info("Installing pytorch")
    pkg_names = ["torch>=2.5.0,<3.0.0"]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkg_names])


# 线程池单例
_thread_pools = {}
_thread_pool_lock = threading.Lock()


def _get_or_create_thread_pool(pool_type="general"):
    """获取或创建线程池（单例模式）"""
    global _thread_pools
    
    if pool_type not in _thread_pools:
        with _thread_pool_lock:
            if pool_type not in _thread_pools:
                cpu_count = multiprocessing.cpu_count()
                
                # 根据环境变量获取最大工作线程数
                max_workers_env = os.getenv(f"THREAD_POOL_MAX_WORKERS_{pool_type.upper()}", 
                                          os.getenv("THREAD_POOL_MAX_WORKERS", "128"))
                
                try:
                    max_workers = int(max_workers_env)
                except ValueError:
                    # 根据任务类型设置默认线程数
                    if pool_type == "io_bound":
                        max_workers = min(cpu_count * 16, 512)  # IO密集型任务使用更多线程
                    elif pool_type == "cpu_bound":
                        max_workers = min(cpu_count * 2, 64)  # CPU密集型任务使用较少线程
                    else:
                        max_workers = min(cpu_count * 8, 256)  # 通用任务
                
                if max_workers < 1:
                    max_workers = 1
                
                # 创建线程池
                _thread_pools[pool_type] = ThreadPoolExecutor(
                    max_workers=max_workers, 
                    thread_name_prefix=f"RAGFlowPool_{pool_type}"
                )
                logging.info(f"Created thread pool {pool_type} with {max_workers} workers")
    
    return _thread_pools[pool_type]


def get_thread_pool_stats():
    """获取线程池统计信息"""
    stats = {}
    for pool_type, pool in _thread_pools.items():
        stats[pool_type] = {
            "max_workers": pool._max_workers,
            "queue_size": pool._work_queue.qsize() if hasattr(pool, "_work_queue") else 0
        }
    return stats


def shutdown_thread_pools():
    """关闭所有线程池"""
    global _thread_pools
    
    with _thread_pool_lock:
        for pool_type, pool in _thread_pools.items():
            try:
                pool.shutdown(wait=False)
                logging.info(f"Shutdown thread pool {pool_type}")
            except Exception as e:
                logging.error(f"Error shutting down thread pool {pool_type}: {e}")
        _thread_pools = {}


async def thread_pool_exec(func, *args, **kwargs):
    """在线程池中执行同步函数"""
    loop = asyncio.get_running_loop()
    
    # 确定线程池类型
    pool_type = kwargs.pop("_pool_type", "general")
    
    # 获取线程池
    pool = _get_or_create_thread_pool(pool_type)
    
    # 构建函数调用
    if kwargs:
        func = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(pool, func)
    return await loop.run_in_executor(pool, func, *args)
