import concurrent.futures
import functools
import itertools
import os
import time
import json
import msgpack
import redis
import refile
from loguru import logger
from tqdm import tqdm

_REDIS_URL = os.environ.get(
    "DATA3D_REDIS_URL",
    "redis://192.18.56.246:6666/0",
)
_REDIS_CONNECTION_POOL = redis.ConnectionPool.from_url(_REDIS_URL)


def retry(max_retry_times=10, retry_interval=10):
    def decorator(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            retry = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning("retry#{} exception:\n{}".format(retry, e))
                    retry += 1
                    if retry > max_retry_times:
                        raise e
                logger.info("retry {}/{} after {} seconds".format(retry, max_retry_times, retry_interval))
                time.sleep(retry_interval)

        return wrap

    return decorator


class RedisCachedIterator:

    _data_prefix = "d3dc"
    _ttl = 60 * 60 * 24 * 7  # 7 days

    def __init__(self, prefix, redis_url=None, **kwargs):
        self.prefix = ".".join([self._data_prefix, prefix])
        self._last_check_time = 0
        self.redis_url = _REDIS_URL if redis_url is None else redis_url

    @functools.lru_cache(maxsize=1)
    def get_client(self):
        return redis.Redis(connection_pool=_REDIS_CONNECTION_POOL)

    @property
    def _client(self):
        return self.get_client()

    @property
    def ttl(self):
        return self._client.ttl(self.prefix)

    def _update_ttl(self):
        now = time.time()
        if self._ttl and (now - self._last_check_time > self._ttl * 0.05):
            self._last_check_time = now
            current_ttl = self._client.ttl(self.prefix)
            current_ttl = current_ttl if current_ttl else 0
            # only update expire when current_ttl is less than 0.5 * target_ttl
            # in order to reduce the load of matser redis server
            result = current_ttl < self._ttl * 0.5
            if result:
                logger.debug("[TTL] prefix: {} update ttl {} -> {}".format(self.prefix, current_ttl, self._ttl))
                self._client.expire(self.prefix, self._ttl)

    def _set(self, key, val):
        self._client.hset(self.prefix, str(key), msgpack.dumps(val))

    @retry()
    def _get(self, key):
        self._update_ttl()
        return msgpack.loads(self._client.hget(self.prefix, str(key)), raw=False)

    def __len__(self):
        return self._get("__len")

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            items = []
            for i in itertools.islice(range(len(self)), idx.start, idx.stop, idx.step):
                items.append(self._get(i))
            return items
        else:
            return self._get(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)

    def exist(self):
        return self._client.hexists(self.prefix, "__len")

    def _insert_iterable(self, iterable, chunk_size=1000, **kwargs):
        pipe = self._client.pipeline()
        total = 0
        for idx, item in enumerate(tqdm(iterable)):
            pipe.hset(self.prefix, idx, msgpack.dumps(item))
            total += 1
            if idx % chunk_size == 0:
                pipe.expire(self.prefix, self._ttl)
                pipe.execute()
                pipe = self._client.pipeline()
        pipe.hset(self.prefix, "__len", msgpack.dumps(total))
        pipe.expire(self.prefix, self._ttl)
        pipe.execute()

class RedisCachedData:
    def __init__(self, path, oss_etag_helper, rebuild=False, **kwargs):
        self.path = path
        self.cache = RedisCachedIterator(oss_etag_helper.get_etag(self.path), **kwargs)
        if not self.cache.exist() or rebuild:
            logger.info("cannot find cache {}, building...".format(path))
            logger.info("cache key: {}".format(self.cache.prefix))
            self._init_cache()
        self.meta = self.cache._get("meta")

    def _init_cache(self):
        with refile.smart_open(self.path, "r") as rf:
            json_data = json.load(rf)

        frames = json_data.pop("frames")
        json_data["key_frame_idx"] = [i for i, x in enumerate(frames) if x["is_key_frame"]]
        self.cache._set("meta", json_data)
        self.cache._insert_iterable(frames)

    def __getitem__(self, key):
        if key == "frames":
            return self.cache
        else:
            return self.meta[key]


class OSSEtagHelper(RedisCachedIterator):
    _ttl = None

    def __init__(self, check_etag=True):
        super().__init__("oss_etag")
        self.check_etag = check_etag
        if self.check_etag:
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
            self.futures = []

    def _check_etag(self, path, etag):
        expected_etag = refile.smart_stat(path).extra["ETag"][1:-1]
        if expected_etag != etag:
            self._set(path, expected_etag)
            raise RuntimeError(f"ETag mismatch! {path} has been changed, please reload")

    def get_etag(self, path):
        if self._client.hexists(self.prefix, path):
            etag = self._get(path)
        else:
            etag = refile.smart_stat(path).extra["ETag"][1:-1]
            self._set(path, etag)
        if self.check_etag:
            future = self.executor.submit(self._check_etag, path, etag)
            self.futures.append(future)
        return etag

    def join(self):
        if not self.check_etag:
            return
        for f in self.futures:
            f.result()
