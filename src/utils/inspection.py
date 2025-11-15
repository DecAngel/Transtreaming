import numpy as np
import torch


def inspect(obj, indent_level=0, prefix=None) -> str:
    """
    递归生成任意变量的YAML格式字符串描述

    Args:
        obj: 要描述的对象
        indent_level: 当前缩进级别
        prefix: 当前对象的前缀

    Returns:
        YAML格式的字符串描述
    """
    indent = "  " * indent_level
    obj_type = type(obj).__name__
    prefix = prefix if prefix is not None else ""
    children = []

    # 处理字典
    if isinstance(obj, dict):
        info = f"len={len(obj)}"
        for key, value in obj.items():
            children.append(inspect(value, indent_level + 1, prefix=f'{str(key)}: '))

    # 处理列表、元组、集合
    elif isinstance(obj, (list, tuple, set)):
        info = f"len={len(obj)}"

        # 转换列表并进行相似性判断
        items_list = list(obj) if not isinstance(obj, set) else list(obj)

        i = 0
        while i < len(items_list):
            current = items_list[i]
            similar_count = 0

            # 检查后续相似元素
            j = i + 1
            while j < len(items_list) and is_similar(current, items_list[j]):
                similar_count += 1
                j += 1

            if similar_count == 0:
                # 没有相似元素，正常处理
                item_str = inspect(current, indent_level + 1, prefix='- ')
                children.append(item_str)
                i += 1
            else:
                # 有相似元素，第一个正常显示，后面的省略
                first_item_str = inspect(current, indent_level + 1, prefix='- ')
                children.append(first_item_str)

                # 添加省略行
                similar_indent = "  " * (indent_level + 1)
                children.append(f"{similar_indent}... similar {similar_count} item{'s' if similar_count > 1 else ''}")

                i += similar_count + 1

    # 处理PyTorch Tensor
    elif isinstance(obj, torch.Tensor):
        if obj.numel() > 0:
            if obj.ndim > 0 and obj.size(-1) == 4:
                # A bbox tensor
                min_val = [f'{obj[..., i].min().item():.1e}' for i in range(4)]
                max_val = [f'{obj[..., i].max().item():.1e}' for i in range(4)]
                mean_val = [f'{obj[..., i].float().mean().item():.1e}' for i in range(4)]
                info = f"type={tensor_type(obj)}, shape={tuple(obj.shape)}, min={','.join(min_val)}, max={','.join(max_val)}, mean={','.join(mean_val)}"
            else:
                min_val = obj.min().item()
                max_val = obj.max().item()
                mean_val = obj.float().mean().item()
                info = f"type={tensor_type(obj)}, shape={tuple(obj.shape)}, min={min_val:.1e}, max={max_val:.1e}, mean={mean_val:.1e}"
        else:
            # Empty tensor
            info = f"type={tensor_type(obj)}, shape={tuple(obj.shape)}"

    # 处理NumPy数组
    elif isinstance(obj, np.ndarray):
        if obj.size > 0:
            if obj.ndim > 0 and obj.shape[-1] == 4:
                # A bbox ndarray
                min_val = [f'{np.min(obj[..., i]):.1e}' for i in range(4)]
                max_val = [f'{np.max(obj[..., i]):.1e}' for i in range(4)]
                mean_val = [f'{np.mean(obj[..., i], dtype=float):.1e}' for i in range(4)]
                info = f"type={ndarray_type(obj)}, shape={tuple(obj.shape)}, min={','.join(min_val)}, max={','.join(max_val)}, mean={','.join(mean_val)}"
            else:
                min_val = np.min(obj)
                max_val = np.max(obj)
                mean_val = np.mean(obj, dtype=float)
                info = f"type={ndarray_type(obj)}, shape={obj.shape}, min={min_val:.1e}, max={max_val:.1e}, mean={mean_val:.1e}"
        else:
            info = f"type={ndarray_type(obj)}, shape={obj.shape}"

    # 处理其他类型
    else:
        info = f"val={str(obj)}"

    return '\n'.join([f"{indent}{prefix}{obj_type}({info})"] + children)


def is_similar(a, b):
    """
    判断两个元素是否相似

    Args:
        a: 第一个元素
        b: 第二个元素

    Returns:
        bool: 是否相似
    """
    # 类型不同则不相似
    if type(a) != type(b):
        return False

    # 字典：所有键相同则相似
    if isinstance(a, dict):
        return set(a.keys()) == set(b.keys())

    # 列表、元组、集合：长度相同则相似
    elif isinstance(a, (list, tuple, set)):
        return len(a) == len(b)

    # PyTorch Tensor：形状相同则相似
    elif isinstance(a, torch.Tensor):
        return a.shape == b.shape

    # NumPy数组：形状相同则相似
    elif isinstance(a, np.ndarray):
        return a.shape == b.shape

    # 其他类型：直接判定为相似
    else:
        return True


def tensor_type(t: torch.Tensor) -> str:
    # 处理设备信息
    device = t.device

    if device.type == 'cuda':
        device_str = f"cuda:{device.index}/"
    elif device.type == 'cpu':
        # CPU设备省略为空
        device_str = ""
    elif device.type == 'mps':
        # Apple Silicon GPU
        device_str = f"mps:{device.index}/" if device.index is not None else "mps/"
    elif device.type == 'xla':
        # TPU设备
        device_str = f"xla:{device.index}/" if device.index is not None else "xla/"
    elif device.type == 'meta':
        # 元设备（用于形状推断）
        device_str = "meta/"
    elif device.type == 'hpu':
        # Habana设备
        device_str = f"hpu:{device.index}/" if device.index is not None else "hpu/"
    else:
        # 其他未知设备
        device_str = f"{device.type}:{device.index}/" if device.index is not None else f"{device.type}/"

    # 处理数据类型
    dtype_map = {
        torch.bool: "bool",
        torch.uint8: "uint8",
        torch.int8: "int8",
        torch.int16: "int16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.float16: "float16",
        torch.float32: "float32",
        torch.float64: "float64",
        torch.complex64: "complex64",
        torch.complex128: "complex128",
        torch.bfloat16: "bfloat16"
    }

    type_str = dtype_map.get(t.dtype, str(t.dtype).replace('torch.', ''))

    return f"{device_str}{type_str}"


def ndarray_type(arr: np.ndarray) -> str:
    """
    返回numpy数组的类型描述字符串

    Args:
        arr: numpy数组

    Returns:
        类型描述字符串，如 "int64", "float32", "bool"
    """
    # 处理数据类型
    dtype_map = {
        np.bool_: "bool",
        np.bool8: "bool",
        np.uint8: "uint8",
        np.uint16: "uint16",
        np.uint32: "uint32",
        np.uint64: "uint64",
        np.int8: "int8",
        np.int16: "int16",
        np.int32: "int32",
        np.int64: "int64",
        np.float16: "float16",
        np.float32: "float32",
        np.float64: "float64",
        np.complex64: "complex64",
        np.complex128: "complex128",
        np.object_: "object",
        np.str_: "str",
        np.bytes_: "bytes",
        np.datetime64: "datetime64",
        np.timedelta64: "timedelta64"
    }

    # 获取数据类型
    dtype = arr.dtype

    # 首先尝试从映射中获取类型名称
    if dtype in dtype_map:
        type_str = dtype_map[dtype]
    else:
        # 对于未知类型，使用dtype的字符串表示
        type_str = str(dtype)
        # 移除可能的numpy前缀
        if type_str.startswith('numpy.'):
            type_str = type_str[6:]

    return type_str
