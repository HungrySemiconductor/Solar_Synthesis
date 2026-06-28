#!/usr/bin/env python3
"""
Surya-Bench 数据下载工具
=======================

从 AWS Open Data Registry 的 nasa-surya-bench 公开 S3 存储桶下载太阳观测数据。

数据来源: https://registry.opendata.aws/surya-bench/

数据说明
--------
- 数据内容: NASA SDO 的 AIA/HMI 观测数据 (Level-1.5 NetCDF)
- 时间范围: 2010-05-13 至 2024-07-31
- 文件格式: NetCDF (.nc)，包含多通道太阳图像序列
- 文件大小: ~310–590 MB/文件
- 目录结构: YYYY/MM/YYYYMMDD_HHMM.nc (每12分钟一个文件)
- 存储桶:   nasa-surya-bench (us-west-2, 公开访问无需 AWS 账号)

依赖
----
Python >= 3.7 (标准库) + curl (推荐, 下载大文件更稳定)

用法示例
--------
    # 0. 查看帮助
    python download_AWS.py --help

    # 1. 列出 2024年1月 的可用文件 (列出前20个)
    python download_AWS.py --list --year 2024 --month 1 --max-list 20

    # 2. 下载1个文件到当前目录 (测试用)
    python download_AWS.py --num 1 --output ./

    # 3. 下载指定日期的文件
    python download_AWS.py --num 5 --year 2024 --month 6 --day 15 --output ./data/

    # 4. 列出2024年所有可用月份
    python download_AWS.py --list-months --year 2024

    # 5. 下载单个指定文件
    python download_AWS.py --key "2024/06/20240615_0000.nc" --output ./

    # 6. 使用 urllib 后端 (无 curl 时自动回退)
    python download_AWS.py --num 1 --backend urllib --output ./

注意
----
- 单个文件约 300-600 MB，确保有足够磁盘空间
- 推荐安装 curl (https://curl.se/)，下载大文件更稳定、支持断点续传
- 按 Ctrl+C 可中断，已下载的部分会保留供续传
"""

import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
import argparse
import os
import sys
import time
import shutil
import subprocess
import shlex
from pathlib import Path

# Windows 控制台 UTF-8 修复
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────────────
S3_BUCKET = "nasa-surya-bench"
S3_REGION = "us-west-2"
S3_BASE_URL = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com"
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"

CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB (urllib 回退用)

# ── 检测可用后端 ──
_CURL_PATH = shutil.which("curl")
_HAS_CURL = _CURL_PATH is not None


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                      S3 列表操作                                   ║
# ╚═════════════════════════════════════════════════════════════════════╝

def list_objects(prefix="", max_keys=1000, delimiter=None):
    """列出 S3 存储桶中的对象 (单次请求)。

    参数
    ----
    prefix : str
        对象键前缀过滤 (如 "2024/01/")
    max_keys : int
        单次最多返回数量 (上限 1000)
    delimiter : str | None
        用于模拟目录层级, 如 "/"

    返回
    ----
    tuple[list[dict], bool, str | None, list[str]]
        (对象列表, 是否截断, 下一页标记, CommonPrefixes)
    """
    params = []
    if prefix:
        params.append(f"prefix={urllib.request.quote(prefix, safe='/')}")
    params.append(f"max-keys={max_keys}")
    if delimiter:
        params.append(f"delimiter={delimiter}")

    url = f"{S3_BASE_URL}/?{'&'.join(params)}"

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            root = ET.fromstring(resp.read())

        objects = []
        for content in root.findall(f"{S3_NS}Contents"):
            objects.append({
                "key": content.find(f"{S3_NS}Key").text,
                "size": int(content.find(f"{S3_NS}Size").text),
                "last_modified": content.find(f"{S3_NS}LastModified").text,
            })

        common_prefixes = []
        for cp in root.findall(f"{S3_NS}CommonPrefixes"):
            common_prefixes.append(cp.find(f"{S3_NS}Prefix").text)

        is_truncated = root.find(f"{S3_NS}IsTruncated")
        truncated = is_truncated is not None and is_truncated.text == "true"

        next_marker = None
        if truncated:
            marker = root.find(f"{S3_NS}NextMarker")
            if marker is not None:
                next_marker = marker.text
            else:
                token = root.find(f"{S3_NS}NextContinuationToken")
                if token is not None:
                    next_marker = token.text

        return objects, truncated, next_marker, common_prefixes

    except urllib.error.URLError as e:
        print(f"[错误] 无法访问 S3 存储桶: {e}")
        sys.exit(1)


# ── 便利查询函数 ──

def list_months(year):
    """列出指定年份下有哪些月份。"""
    print(f"正在查询 {year} 年的可用月份...")
    _, _, _, prefixes = list_objects(prefix=f"{year}/", max_keys=1000, delimiter="/")
    months = []
    for p in prefixes:
        parts = p.rstrip("/").split("/")
        if len(parts) >= 2:
            months.append(parts[1])
    return sorted(months)


def list_days(year, month):
    """列出指定年月下有哪些天。"""
    prefix = f"{year}/{month:0>2}/"
    print(f"正在查询 {prefix} 的可用日期...")
    _, _, _, prefixes = list_objects(prefix=prefix, max_keys=1000, delimiter="/")
    days = []
    for p in prefixes:
        parts = p.rstrip("/").split("/")
        if len(parts) >= 3:
            days.append(parts[2])
    return sorted(days)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                      下载功能                                      ║
# ╚═════════════════════════════════════════════════════════════════════╝

def format_size(size_bytes):
    """字节数 → 人类可读。"""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_speed(bytes_per_sec):
    """速度格式化。"""
    return f"{format_size(bytes_per_sec)}/s"


def _download_with_curl(url, local_path, show_progress=True):
    """使用 curl 下载, 自动断点续传。

    curl -C - 会自动检测本地文件大小并从断点继续。
    返回 True 表示成功, False 表示失败。
    """
    cmd = [
        _CURL_PATH,
        "-L",               # 跟随重定向
        "-C", "-",          # 断点续传: 自动检测已有文件大小
        "-o", str(local_path),
        "--retry", "3",     # curl 内置重试
        "--retry-delay", "3",
        "--connect-timeout", "30",
        "--max-time", "0",  # 不限速下载 (0 = 无限制)
        "--fail",           # HTTP 错误时返回失败
    ]
    if show_progress:
        cmd.extend(["--progress-bar"])  # curl 原生进度条
    else:
        cmd.extend(["-sS"])  # silent but show errors

    cmd.append(url)

    try:
        result = subprocess.run(
            cmd,
            stderr=subprocess.STDOUT,
            stdout=None if show_progress else subprocess.PIPE,
            text=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
    except subprocess.SubprocessError as e:
        print(f"\n  [curl错误] {e}")
        return False


def _download_with_urllib(url, local_path, remote_size, show_progress=True, max_retries=5):
    """使用 Python urllib 下载 (回退方案)。

    支持 HTTP Range 断点续传。
    返回 True 表示成功, False 表示失败。
    """
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait = min(2 ** attempt, 60)
            print(f"  第 {attempt} 次重试 (等待 {wait}s)...")
            time.sleep(wait)

        # 断点续传: 检查已有字节
        start_byte = 0
        if local_path.exists():
            start_byte = local_path.stat().st_size
            if remote_size and start_byte >= remote_size:
                return True  # 已完成

        try:
            req = urllib.request.Request(url)
            if start_byte > 0:
                req.add_header("Range", f"bytes={start_byte}-")

            with urllib.request.urlopen(req, timeout=300) as resp:
                content_length = resp.headers.get("Content-Length")
                content_range = resp.headers.get("Content-Range", "")

                if content_range:
                    total_size = int(content_range.rsplit("/", 1)[-1])
                elif content_length:
                    total_size = start_byte + int(content_length)
                else:
                    total_size = remote_size

                mode = "ab" if start_byte > 0 else "wb"

                with open(local_path, mode) as f:
                    downloaded = start_byte
                    start_time = time.time()
                    last_print = start_time

                    while True:
                        try:
                            chunk = resp.read(CHUNK_SIZE)
                        except (ConnectionResetError, ConnectionAbortedError,
                                TimeoutError, urllib.error.URLError,
                                socket.timeout) as e:
                            print(f"\n  [断连] {e}, 已保存 {format_size(downloaded)}")
                            break

                        if not chunk:
                            break

                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if show_progress and (now - last_print >= 1.0 or
                                              (total_size and downloaded >= total_size)):
                            elapsed = now - start_time
                            speed = (downloaded - start_byte) / elapsed if elapsed > 0 else 0
                            if total_size:
                                pct = downloaded / total_size * 100
                                bar_len = 40
                                filled = int(bar_len * downloaded / total_size)
                                bar = "#" * filled + "-" * (bar_len - filled)
                                remaining = total_size - downloaded
                                eta = remaining / speed if speed > 0 else 0
                                if eta < 3600:
                                    eta_str = f"ETA {eta:.0f}s"
                                else:
                                    eta_str = f"ETA {eta/60:.1f}min"
                                print(
                                    f"\r  {bar} {pct:5.1f}%  "
                                    f"{format_size(downloaded)}/{format_size(total_size)}  "
                                    f"{format_speed(speed)}  {eta_str}   ",
                                    end="", flush=True,
                                )
                            last_print = now

                    if total_size and downloaded >= total_size:
                        elapsed = time.time() - start_time
                        avg_speed = (downloaded - start_byte) / elapsed if elapsed > 0 else 0
                        print(f"\r  [OK] 完成: {format_size(downloaded)}  "
                              f"用时 {elapsed:.0f}s ({format_speed(avg_speed)})   ")
                        return True

        except urllib.error.URLError as e:
            print(f"\n  [错误] 网络错误: {e}")

    # 所有重试耗尽
    local_size = local_path.stat().st_size if local_path.exists() else 0
    if remote_size and local_size >= remote_size:
        return True
    return False


def download_file(key, output_dir, backend="auto", show_progress=True):
    """从 S3 下载单个文件, 保持目录结构。自动断点续传。

    参数
    ----
    key : str
        S3 对象键, 如 "2024/06/20240615_0000.nc"
    output_dir : str | Path
        输出根目录, 文件保存在 <output_dir>/<key>
    backend : str
        "auto" = 优先 curl, 无 curl 时回退 urllib
        "curl" = 仅使用 curl
        "urllib" = 仅使用 urllib
    show_progress : bool
        显示进度

    返回
    ----
    Path | None
        成功返回本地路径, 失败返回 None
    """
    output_dir = Path(output_dir)
    local_path = output_dir / key
    local_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{S3_BASE_URL}/{urllib.request.quote(key, safe='/')}"

    # ── 获取远程文件大小 ──
    remote_size = None
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                remote_size = int(content_length)
    except Exception:
        pass

    # ── 检查本地文件是否已完整 ──
    if local_path.exists() and remote_size:
        local_size = local_path.stat().st_size
        if local_size == remote_size:
            print(f"  [跳过] {key} 已完整 ({format_size(local_size)})")
            return local_path
        elif local_size > 0:
            print(f"  [续传] {key} 已有 {format_size(local_size)}/"
                  f"{format_size(remote_size)}, 继续下载")

    # ── 选择后端 ──
    use_curl = False
    if backend == "curl":
        if not _HAS_CURL:
            print("  [错误] --backend curl 但系统未安装 curl")
            return None
        use_curl = True
    elif backend == "urllib":
        use_curl = False
    elif backend == "auto":
        use_curl = _HAS_CURL

    # ── 执行下载 ──
    if use_curl:
        success = _download_with_curl(url, local_path, show_progress=show_progress)
    else:
        if backend == "auto":
            print("  [提示] 未检测到 curl, 使用 urllib (下载大文件可能较慢)")
        success = _download_with_urllib(
            url, local_path, remote_size, show_progress=show_progress
        )

    # ── 验证结果 ──
    if success:
        local_size = local_path.stat().st_size if local_path.exists() else 0
        if remote_size and local_size != remote_size:
            print(f"  [警告] 下载不完整: {format_size(local_size)}/{format_size(remote_size)}")
            # 不删除文件，留待续传
            return None
        return local_path
    else:
        if local_path.exists():
            local_size = local_path.stat().st_size
            print(f"  [失败] {key} - 已保存 {format_size(local_size)} (可续传)")
        return None


def download_files(objects, output_dir, num=None, backend="auto", show_progress=True):
    """下载多个文件。

    返回 (成功数, 失败数, 下载字节数)。
    """
    to_download = objects if (num is None or num <= 0) else objects[:num]

    total_bytes = sum(obj["size"] for obj in to_download)
    print(f"\n准备下载 {len(to_download)} 个文件, 总计 {format_size(total_bytes)}")
    print(f"输出目录: {Path(output_dir).resolve()}")
    print(f"下载后端: {backend}\n")

    success = 0
    failed = 0
    downloaded_bytes = 0

    for i, obj in enumerate(to_download, 1):
        key = obj["key"]
        size = obj["size"]
        print(f"[{i}/{len(to_download)}] {key} ({format_size(size)})")

        result = download_file(key, output_dir, backend=backend, show_progress=show_progress)
        if result:
            success += 1
            downloaded_bytes += size
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"下载完成: {success} 成功, {failed} 失败")
    if downloaded_bytes > 0:
        print(f"总下载量: {format_size(downloaded_bytes)}")
    print(f"{'='*60}")

    return success, failed, downloaded_bytes


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                      CLI 主入口                                    ║
# ╚═════════════════════════════════════════════════════════════════════╝

def main():
    parser = argparse.ArgumentParser(
        description="从 AWS Surya-Bench 公开数据集下载太阳观测数据 (NetCDF)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  %(prog)s --list --year 2024 --month 1              列出文件
  %(prog)s --num 1 --output ./                        下载1个文件(测试)
  %(prog)s --num 5 --year 2024 --month 6 --output ./data/
  %(prog)s --key "2024/06/20240615_0000.nc" --output ./
  %(prog)s --list-months --year 2024
  %(prog)s --num 1 --backend urllib --output ./       使用 urllib 后端
        """,
    )

    # 操作模式
    parser.add_argument("--list", action="store_true",
                        help="仅列出可用文件 (不下载)")
    parser.add_argument("--list-months", action="store_true",
                        help="列出指定年份的可用月份")
    parser.add_argument("--list-days", action="store_true",
                        help="列出指定年/月的可用日期 (需同时指定 --year --month)")

    # 筛选
    parser.add_argument("--year", type=int, default=None, help="年份, 如 2024")
    parser.add_argument("--month", type=int, default=None, help="月份 (1-12)")
    parser.add_argument("--day", type=int, default=None, help="日期 (1-31)")
    parser.add_argument("--key", type=str, default=None,
                        help="直接指定 S3 对象键, 如 '2024/06/20240615_0000.nc'")

    # 下载选项
    parser.add_argument("--num", type=int, default=None,
                        help="下载数量 (默认1个; 0=全部). 测试建议用1")
    parser.add_argument("--output", "--output-dir", dest="output_dir",
                        type=str, default="./", help="输出目录 (默认当前目录)")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "curl", "urllib"],
                        help="下载后端: auto(推荐)=curl优先, curl=仅curl, urllib=仅urllib")
    parser.add_argument("--max-list", type=int, default=50,
                        help="--list 显示的最大文件数 (默认50)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="静默模式")

    args = parser.parse_args()

    # ── list-months ──
    if args.list_months:
        if not args.year:
            parser.error("--list-months 需要指定 --year")
        months = list_months(args.year)
        print(f"\n{args.year} 年有数据的月份 ({len(months)} 个):")
        for m in months:
            print(f"  {args.year}/{m}")
        return

    # ── list-days ──
    if args.list_days:
        if not args.year or not args.month:
            parser.error("--list-days 需要同时指定 --year 和 --month")
        days = list_days(args.year, args.month)
        print(f"\n{args.year}/{args.month:0>2} 有数据的日期 ({len(days)} 天):")
        for d in days:
            print(f"  {args.year}/{args.month:0>2}/{d}")
        return

    # ── 构建前缀 ──
    prefix_parts = []
    if args.year:
        prefix_parts.append(str(args.year))
        if args.month:
            prefix_parts.append(f"{args.month:0>2}")
            if args.day:
                # YYYY/MM/YYYYMMDD_*.nc
                date_prefix = f"{args.year}{args.month:0>2}{args.day:0>2}_"
                prefix_parts.append(date_prefix)

    if prefix_parts and args.day:
        prefix = "/".join(prefix_parts)  # "2024/01/20240101_"
    elif prefix_parts:
        prefix = "/".join(prefix_parts) + "/"  # "2024/01/"
    else:
        prefix = ""

    # ── --key 直接下载 ──
    if args.key:
        print(f"下载指定文件: {args.key}")
        print(f"S3 URL: {S3_BASE_URL}/{args.key}")
        remote_size = None
        try:
            req = urllib.request.Request(
                f"{S3_BASE_URL}/{urllib.request.quote(args.key, safe='/')}",
                method="HEAD")
            with urllib.request.urlopen(req, timeout=30) as resp:
                cl = resp.headers.get("Content-Length")
                if cl:
                    remote_size = int(cl)
                    print(f"文件大小: {format_size(remote_size)}")
        except Exception:
            pass
        print()
        obj = {"key": args.key, "size": remote_size or 0, "last_modified": "N/A"}
        download_files([obj], args.output_dir, num=1,
                       backend=args.backend, show_progress=not args.quiet)
        return

    # ── 获取文件列表 ──
    print(f"S3 存储桶: s3://{S3_BUCKET}")
    if prefix:
        print(f"前缀过滤: {prefix}")
    print("正在获取文件列表...\n")

    list_limit = args.max_list if args.list else 1000
    objects, truncated, _, _ = list_objects(
        prefix=prefix, max_keys=list_limit, delimiter=None
    )

    if not objects:
        print("未找到匹配的文件。")
        if prefix:
            print("提示: 使用 --list-months --year YYYY 查看可用月份")
        return

    # ── --list 模式 ──
    if args.list:
        display = objects[:args.max_list]
        total_size = sum(obj["size"] for obj in display)

        print(f"找到 {len(objects)} 个文件 (显示前 {len(display)} 个):\n")
        print(f"{'序号':<6} {'文件名':<40} {'大小':<12} {'修改时间'}")
        print("-" * 90)
        for i, obj in enumerate(display, 1):
            fname = Path(obj["key"]).name
            print(f"{i:<6} {fname:<40} {format_size(obj['size']):<12} "
                  f"{obj['last_modified']}")
        print("-" * 90)
        print(f"显示文件总大小: {format_size(total_size)}")
        if truncated:
            print("注意: 结果被截断, 使用 --max-list N 查看更多")
        return

    # ── 下载模式 ──
    num = args.num if args.num is not None else 1
    objects.sort(key=lambda x: x["key"])
    print(f"找到 {len(objects)} 个匹配文件")

    download_files(objects, args.output_dir, num=num,
                   backend=args.backend, show_progress=not args.quiet)


if __name__ == "__main__":
    # 处理 urllib 中可能用到的 socket.timeout
    import socket
    main()
