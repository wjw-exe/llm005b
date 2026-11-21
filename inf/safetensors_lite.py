# -*- coding: utf-8 -*-
"""safetensors 手写解析器（零第三方库，仅 numpy + 标准库）。

文件格式
--------
[0:8)   uint64 小端：header 的字节长度 N
[8:8+N) UTF-8 JSON header：{"tensor_name": {"dtype","shape","data_offsets"}, ...}
[8+N:)  原始张量字节区，data_offsets 是相对该区起点的 [begin, end)

支持 dtype：F32 / F16 / F64 / BF16 / I64 / I32 / I16 / I8 / U8 / BOOL。
对外统一转 float32（整型转 float32 便于直接进算子）。
"""
import json
import struct
from typing import Dict, Tuple

import numpy as np

_NP_DTYPE = {
    "F32": np.float32, "F16": np.float16, "F64": np.float64,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def read_header(path: str) -> Tuple[Dict[str, dict], int, int]:
    """返回 (header_dict, header_bytes, file_size)，并校验长度自洽。"""
    import os
    file_size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("文件不足 8 字节，不是合法 safetensors")
        (hlen,) = struct.unpack("<Q", raw)
        if hlen <= 0 or hlen > 100 * 1024 * 1024:
            raise ValueError(f"header 长度异常: {hlen}")
        body = f.read(hlen)
    if len(body) != hlen:
        raise ValueError("header 被截断（文件下载不完整？）")
    try:
        header = json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"header JSON 解析失败: {exc}") from exc
    header.pop("__metadata__", None)
    return header, hlen, file_size


def _bf16_to_f32(arr_u16: np.ndarray) -> np.ndarray:
    """BF16 -> FP32：低位补 16 个 0。"""
    u32 = arr_u16.astype(np.uint32) << 16
    return u32.view(np.float32)


def load_safetensors(path: str) -> Dict[str, np.ndarray]:
    """一次性载入全部张量，返回 {name: np.ndarray}（float32 或原始整型）。"""
    header, hlen, file_size = read_header(path)
    data_start = 8 + hlen
    total = 0
    for name, meta in header.items():
        total = max(total, int(meta["data_offsets"][1]))
    expect = data_start + total
    if file_size < expect:
        raise ValueError(
            f"文件不完整：实际 {file_size} 字节 < 期望 {expect} 字节，请重新下载")
    out: Dict[str, np.ndarray] = {}
    raw_all = np.memmap(path, dtype=np.uint8, mode="r")
    for name, meta in header.items():
        dtype = meta["dtype"]
        shape = tuple(int(x) for x in meta["shape"])
        begin, end = (int(x) for x in meta["data_offsets"])
        buf = np.asarray(raw_all[data_start + begin:data_start + end])
        if dtype == "BF16":
            arr = _bf16_to_f32(buf.view(np.uint16))
        else:
            if dtype not in _NP_DTYPE:
                raise ValueError(f"不支持的 dtype: {dtype}（张量 {name}）")
            arr = buf.view(_NP_DTYPE[dtype])
        arr = arr.reshape(shape) if shape else arr.reshape(())
        need = 1
        for s in shape:
            need *= s
        if arr.size != need:
            raise ValueError(
                f"张量 {name} 字节数与形状不符：shape={shape} 元素={arr.size}/{need}")
        if arr.dtype.kind == "f":
            out[name] = arr.astype(np.float32, copy=False)
        else:
            out[name] = arr
    return out


def describe(path: str) -> str:
    """生成人类可读的头信息摘要（用于日志与报告）。"""
    header, hlen, file_size = read_header(path)
    lines = [f"file      : {path}",
             f"size      : {file_size:,} bytes ({file_size / 1048576:.1f} MiB)",
             f"hdr_bytes : {hlen}",
             f"tensors   : {len(header)}"]
    total = 0
    for name in sorted(header):
        meta = header[name]
        shape = tuple(int(x) for x in meta["shape"])
        cnt = 1
        for s in shape:
            cnt *= s
        total += cnt
        lines.append(f"  {name:34s} {str(shape):18s} {meta['dtype']:5s} {cnt:>12,}")
    lines.append(f"total params (含 tied 重复存储): {total:,} = {total / 1e6:.2f} M")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    print(describe(sys.argv[1]))
