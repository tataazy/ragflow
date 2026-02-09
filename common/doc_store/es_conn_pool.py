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
import logging
import time
from elasticsearch import Elasticsearch

from common import settings
from common.decorator import singleton

# 原始重试次数保留，仅微调等待逻辑（高并发下避免频繁重试）
ATTEMPT_TIME = 2
# 高并发核心参数（适配9.x客户端）
CONNECTIONS_PER_NODE = 100  # 每个ES节点的并发连接数（根据集群节点数调整，单节点设50-100）
REQUEST_TIMEOUT = 600      # 请求超时（高并发下延长，避免短超时失败）
MAX_RETRIES = 3            # 最大重试次数
RETRY_ON_TIMEOUT = True    # 超时后自动重试


@singleton
class ElasticSearchConnectionPool:

    def __init__(self):
        if hasattr(settings, "ES"):
            self.ES_CONFIG = settings.ES
        else:
            self.ES_CONFIG = settings.get_base_config("es", {})

        # 保留原始重试逻辑，微调等待策略（指数退避）
        connect_success = False
        for attempt in range(ATTEMPT_TIME):
            try:
                if self._connect():
                    connect_success = True
                    break
            except Exception as e:
                wait_time = 5 * (attempt + 1)
                logging.warning(f"{str(e)}. Waiting Elasticsearch {self.ES_CONFIG['hosts']} to be healthy, sleep {wait_time}s.")
                time.sleep(wait_time)

        if not connect_success or not self.es_conn or not self.es_conn.ping():
            msg = f"Elasticsearch {self.ES_CONFIG['hosts']} is unhealthy in {ATTEMPT_TIME*5}s."
            logging.error(msg)
            raise Exception(msg)

        # 版本检查：适配9.x（仅拒绝<8版本，保留原始逻辑结构）
        v = self.info.get("version", {"number": "9.2.1"})
        v_main = int(v["number"].split(".")[0])
        if v_main < 8:
            msg = f"Elasticsearch version must be ≥8, current: {v['number']}"
            logging.error(msg)
            raise Exception(msg)
        logging.info(f"ES连接成功（9.x客户端适配），版本：{v['number']}，单节点并发连接数：{CONNECTIONS_PER_NODE}")

    def _connect(self):
        """核心调整：适配9.2.1客户端参数，聚焦高并发"""
        auth_kwargs = {}
        if "username" in self.ES_CONFIG and "password" in self.ES_CONFIG:
            auth_kwargs["basic_auth"] = (self.ES_CONFIG["username"], self.ES_CONFIG["password"])

        # 9.2.1客户端参数（高并发优化）
        self.es_conn = Elasticsearch(
            hosts=self.ES_CONFIG["hosts"].split(","),
            verify_certs=self.ES_CONFIG.get("verify_certs", False),
            # 9.x高并发核心参数（替换原maxsize）
            connections_per_node=CONNECTIONS_PER_NODE,  # 每个节点的并发连接数（高并发关键）
            # 容错参数（提升高并发下的请求成功率）
            max_retries=MAX_RETRIES,                    # 最大重试次数
            retry_on_timeout=RETRY_ON_TIMEOUT,          # 超时自动重试
            # 超时配置（9.x用request_timeout替代原timeout）
            request_timeout=REQUEST_TIMEOUT,
            **auth_kwargs
        )

        if self.es_conn:
            self.info = self.es_conn.info()
            return True
        return False

    def get_conn(self):
        """保留原始逻辑，复用连接"""
        return self.es_conn

    def refresh_conn(self):
        """保留原始逻辑，适配9.x客户端的close/重连"""
        if self.es_conn and self.es_conn.ping():
            return self.es_conn
        else:
            if self.es_conn:
                self.es_conn.close()
            self._connect()
            return self.es_conn

    def __del__(self):
        """保留原始销毁逻辑"""
        if hasattr(self, "es_conn") and self.es_conn:
            self.es_conn.close()


ES_CONN = ElasticSearchConnectionPool()