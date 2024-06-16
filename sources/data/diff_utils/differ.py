
from .diff.common import Insert, Keep, Remove
from typing import List, Literal, NamedTuple, Union
# from .util import is_code, is_file_same
from .diff.myers import myers_diff

DiffKeyType = Literal["add", "remove", "modify", "same"]

# def project_diff(path_before: str, path_after: str) -> Dict[DiffKeyType, List[str]]:
#     """项目级别文件比较
#
#     Args:
#         path_before (str): 修改前文件夹路径
#         path_after (str): 修改后文件夹路径
#
#     Returns:
#         Dict[str, List[str]]: 项目比较结果
#     """
#     if not os.path.isdir(path_before):
#         raise RuntimeError(f"输入的路径不是文件夹：{path_before}")
#     if not os.path.isdir(path_after):
#         raise RuntimeError(f"输入的路径不是文件夹：{path_after}")
#
#     ret = {
#         'add': [],
#         'remove': [],
#         'modify': [],
#         'same': []
#     }
#
#     for (dirpath, dirnames, filenames) in os.walk(path_before):
#         rel_path = os.path.relpath(dirpath, path_before)
#         source_files = [f for f in filenames if is_code(f)]
#         for source_name in source_files:
#             rel_source_before = os.path.join(rel_path, source_name)
#             rel_source_after = os.path.join(rel_path, source_name)
#             abs_source_before = os.path.join(path_before, rel_path, source_name)
#             abs_source_after = os.path.join(path_after, rel_path, source_name)
#             if os.path.exists(abs_source_after):
#                 if is_file_same(abs_source_before, abs_source_after):
#                     ret['same'].append(rel_source_before)
#                 else:
#                     ret['modify'].append(rel_source_before)
#             else:
#                 ret['remove'].append(rel_source_before)
#
#     for (dirpath, dirnames, filenames) in os.walk(path_after):
#         rel_path = os.path.relpath(dirpath, path_after)
#         source_files = [f for f in filenames if is_code(f)]
#         for source_name in source_files:
#             rel_source_before = os.path.join(rel_path, source_name)
#             rel_source_after = os.path.join(rel_path, source_name)
#             abs_source_before = os.path.join(path_before, rel_path, source_name)
#             abs_source_after = os.path.join(path_after, rel_path, source_name)
#             if not os.path.exists(abs_source_before):
#                 ret['add'].append(rel_source_after)
#
#     return ret



def fileLineDiff(before, after) -> List[NamedTuple]:
    """进行文本行diff

    Args:
        file_before (str): 改动前文件
        file_after (str): 改动后文件

    Raises:
        RuntimeError: [description]
        RuntimeError: [description]

    Returns:
        List[NamedTuple]: 返回改动的行差异分析结果
    """    
    # if not os.path.isfile(file_before):
    #     raise RuntimeError(f"输入的路径不是文件：{file_before}")
    # if not os.path.isfile(file_after):
    #     raise RuntimeError(f"输入的路径不是文件：{file_after}")
    #
    # with open(file_before, 'r', encoding='utf-8') as f:
    #     lines_before = [l.rstrip() for l in f.readlines()]
    #
    # with open(file_after, 'r', encoding='utf-8') as f:
    #     lines_after = [l.rstrip() for l in f.readlines()]


    result = myers_diff(before, after)
    return result
    

def getModifiedBitmap(diffResult: List[Union[Keep, Insert, Remove]]) -> List[bool]:
    lineModified: List[bool] = []
    for elem in diffResult:
        if isinstance(elem, Keep):
            lineModified.append(False)
        elif isinstance(elem, Insert):
            lineModified.append(True)
    return lineModified
