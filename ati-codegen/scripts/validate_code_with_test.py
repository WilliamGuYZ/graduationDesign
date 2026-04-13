import json
import re
import traceback
from io import StringIO
import contextlib

def test_solution_code(solution_code, test_code, test_info):
    """
    使用 test_info 中的函数信息测试 solution 代码
    
    Args:
        solution_code: solution 的代码字符串
        test_code: test 的代码字符串
        test_info: 包含函数名、参数等信息的列表
    
    Returns:
        (success, error_message)
    """
    namespace = {}
    
    try:
        # 执行 solution 代码
        exec(solution_code, namespace)
        
        # 修改测试代码：移除 from solution import 语句
        lines = test_code.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # 跳过所有 from solution import 的行
            if 'from solution import' in line or 'import solution' in line:
                continue
            else:
                cleaned_lines.append(line)
        
        modified_test_code = '\n'.join(cleaned_lines)
        
        # 执行清理后的测试代码
        exec(modified_test_code, namespace)
        
        # 查找所有以 test_ 开头的函数
        test_functions = [name for name in namespace.keys() 
                         if name.startswith('test_') and callable(namespace[name])]
        
        if not test_functions:
            return False, "No test functions found"
        
        # 运行测试
        passed = 0
        failed = 0
        errors = []
        
        for test_name in test_functions:
            try:
                namespace[test_name]()
                passed += 1
            except AssertionError as e:
                failed += 1
                errors.append(f"{test_name}: AssertionError")
            except Exception as e:
                failed += 1
                errors.append(f"{test_name}: {type(e).__name__}")
        
        success = (failed == 0)
        error_msg = None if success else f"{failed} test(s) failed: {'; '.join(errors[:3])}"
        
        return success, error_msg
        
    except Exception as e:
        error_msg = f"Execution error: {type(e).__name__}: {str(e)}"
        return False, error_msg


def process_jsonl_file(input_file, output_file=None):
    """处理 JSONL 文件，只输出统计信息"""
    results = []
    total = 0
    passed = 0
    failed = 0
    error_types = {}
    
    print(f"正在处理文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                total += 1
                
                solution_code = data.get('solution', '')
                test_code = data.get('test', '')
                test_info = data.get('test_info', [])
                
                if not solution_code:
                    result = {'line': line_num, 'success': False, 'error': 'Missing solution code'}
                    failed += 1
                    error_types['Missing solution code'] = error_types.get('Missing solution code', 0) + 1
                elif not test_code:
                    result = {'line': line_num, 'success': False, 'error': 'Missing test code'}
                    failed += 1
                    error_types['Missing test code'] = error_types.get('Missing test code', 0) + 1
                else:
                    success, error = test_solution_code(solution_code, test_code, test_info)
                    
                    result = {
                        'line': line_num,
                        'success': success,
                        'error': error
                    }
                    
                    if success:
                        passed += 1
                    else:
                        failed += 1
                        # 统计错误类型
                        if error:
                            if 'Execution error' in error:
                                error_types['Execution error'] = error_types.get('Execution error', 0) + 1
                            elif 'No test functions' in error:
                                error_types['No test functions'] = error_types.get('No test functions', 0) + 1
                            else:
                                error_types['Test assertion failed'] = error_types.get('Test assertion failed', 0) + 1
                
                results.append(result)
                
                # 进度显示（每1000条显示一次）
                if total % 1000 == 0:
                    print(f"  已处理: {total} 条 (通过: {passed}, 失败: {failed})")
                
            except json.JSONDecodeError as e:
                results.append({
                    'line': line_num,
                    'success': False,
                    'error': f'JSON decode error: {e}'
                })
                failed += 1
                error_types['JSON decode error'] = error_types.get('JSON decode error', 0) + 1
    
    # 输出总结统计
    print(f"\n{'='*70}")
    print(f"验证结果统计")
    print(f"{'='*70}")
    print(f"总处理条数:     {total:,}")
    print(f"验证通过:       {passed:,} ({passed/total*100:.2f}%)" if total > 0 else "验证通过:       0")
    print(f"验证失败:       {failed:,} ({failed/total*100:.2f}%)" if total > 0 else "验证失败:       0")
    print(f"{'='*70}")
    
    if error_types:
        print(f"\n失败原因统计:")
        print(f"{'-'*70}")
        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type:<30} {count:>6,} 条 ({count/failed*100:.1f}%)" if failed > 0 else f"  {error_type:<30} {count:>6,} 条")
    
    # 保存详细结果
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"\n详细结果已保存到: {output_file}")
    
    print(f"{'='*70}\n")
    
    return total, passed, failed, results


def main():
    # ===== 修改这里的路径 =====
    INPUT_FILE = "data/raw/KodCode.jsonl"
    OUTPUT_FILE = "data/raw/KodCode_results.jsonl"
    # =========================
    
    process_jsonl_file(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()