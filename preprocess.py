# coding=utf-8
import sys
import re
import logging  # 使用日志记录进行调试

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class PreprocessorError(Exception):
    pass


class Macro:
    """存储宏定义的类"""

    def __init__(self, name, value, args=None, is_function_like=False):
        self.name = name
        self.value = value
        self.args = args if args is not None else []
        self.is_function_like = is_function_like

    def __repr__(self):
        if self.is_function_like:
            return f"Macro(name='{self.name}', args={self.args}, value='{self.value}')"
        else:
            return f"Macro(name='{self.name}', value='{self.value}')"


class BasicPreprocessor:
    def __init__(self):
        self.macros = {}
        # 条件编译状态栈，元素为 (directive, is_active, condition_already_met)
        self.conditional_stack = []
        self.line_mapping = []

    def _is_active(self):
        if not self.conditional_stack:
            return True
        return self.conditional_stack[-1][1]

    def _handle_line_continuation(self, code):
        """处理行连接符 '\' """
        return re.sub(r'\\\n', '', code)

    def _remove_comments(self, code):
        # 移除多行注释 /* ... */ (非贪婪)
        # 注意：这种简单的移除方式如果遇到跨行的注释，可能会影响后续 splitlines() 的准确性，
        # 但对于标准的注释风格通常有效。
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # 移除单行注释 // ...
        code = re.sub(r'//.*', '', code)
        return code

    def _parse_define(self, line):
        """解析 #define 指令"""
        # （此函数内部逻辑未改变）
        match_simple = re.match(r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+(.*)', line)
        match_func = re.match(r'#define\s+([a-zA-Z_][a-zA-Z0-9_]*)\((.*?)\)\s*(.*)', line)
        if match_func:
            name = match_func.group(1)
            args_str = match_func.group(2).strip()
            args = [arg.strip() for arg in args_str.split(',') if arg.strip()] if args_str else []
            value = match_func.group(3).strip()
            logging.debug(f"定义函数式宏: {name}({args}) = '{value}'")
            self.macros[name] = Macro(name, value, args, is_function_like=True)
        elif match_simple:
            name = match_simple.group(1)
            value = match_simple.group(2).strip()
            logging.debug(f"定义对象式宏: {name} = '{value}'")
            self.macros[name] = Macro(name, value, is_function_like=False)
        else:
            # 在抛出错误前记录行内容可能有助于调试
            logging.error(f"无法解析的 #define 语句: {line}")
            raise PreprocessorError(f"无法解析的 #define 语句: {line}")

    def _handle_undef(self, line):
        """处理 #undef 指令"""
        # （此函数内部逻辑未改变）
        match = re.match(r'#undef\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if match:
            name = match.group(1)
            if name in self.macros:
                logging.debug(f"取消定义宏: {name}")
                del self.macros[name]
            else:
                logging.warning(f"#undef 未定义的宏: {name}")
        else:
            raise PreprocessorError(f"无法解析的 #undef 语句: {line}")

    def _evaluate_condition(self, condition_str):
        condition_str = condition_str.strip()
        logging.debug(f"开始评估条件: '{condition_str}'")
        defined_match = re.match(r'defined\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)|defined\s+([a-zA-Z_][a-zA-Z0-9_]*)',
                                 condition_str)
        if defined_match:
            macro_name = defined_match.group(1) or defined_match.group(2)
            is_defined = macro_name in self.macros
            logging.debug(f"defined({macro_name}) -> {is_defined}")
            return is_defined

        evaluated_str = self._apply_object_macros(condition_str)
        logging.debug(f"应用对象宏后: '{evaluated_str}'")
        try:
            result = eval(evaluated_str, {"__builtins__": None}, {})
            is_true = bool(result)
            logging.debug(f"条件表达式 '{condition_str}' ('{evaluated_str}') 评估结果: {result} -> {is_true}")
            return is_true

        except (SyntaxError, NameError, TypeError, ValueError, Exception) as e:
            logging.warning(
                f"无法评估条件为有效表达式: '{condition_str}' ('{evaluated_str}') 错误: {type(e).__name__}: {e} -> False")
            return False

    def _handle_if(self, line):
        # （此函数内部逻辑未改变）
        condition_str = line[len("#if"):].strip()
        is_true = self._evaluate_condition(condition_str)
        parent_active = self._is_active()
        currently_active = parent_active and is_true
        self.conditional_stack.append(("#if", currently_active, is_true))
        logging.debug(
            f"#if {condition_str} -> {'Active' if currently_active else 'Inactive'}, Parent Active: {parent_active}, Condition True: {is_true}. Stack depth: {len(self.conditional_stack)}")

    def _handle_ifdef(self, line):
        # （此函数内部逻辑未改变）
        match = re.match(r'#ifdef\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if not match: raise PreprocessorError(f"无法解析 #ifdef: {line}")
        name = match.group(1)
        is_defined = name in self.macros
        parent_active = self._is_active()
        currently_active = parent_active and is_defined
        self.conditional_stack.append(("#ifdef", currently_active, is_defined))
        logging.debug(
            f"#ifdef {name} -> {'Active' if currently_active else 'Inactive'}, Parent Active: {parent_active}, Defined: {is_defined}. Stack depth: {len(self.conditional_stack)}")

    def _handle_ifndef(self, line):
        # （此函数内部逻辑未改变）
        match = re.match(r'#ifndef\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
        if not match: raise PreprocessorError(f"无法解析 #ifndef: {line}")
        name = match.group(1)
        is_defined = name in self.macros
        parent_active = self._is_active()
        currently_active = parent_active and (not is_defined)
        self.conditional_stack.append(("#ifndef", currently_active, not is_defined))
        logging.debug(
            f"#ifndef {name} -> {'Active' if currently_active else 'Inactive'}, Parent Active: {parent_active}, Defined: {is_defined}. Stack depth: {len(self.conditional_stack)}")

    def _handle_elif(self, line):
        # （此函数内部逻辑未改变）
        if not self.conditional_stack: raise PreprocessorError("#elif 没有匹配的 #if/#ifdef/#ifndef")
        directive, was_active, previous_condition_met = self.conditional_stack[-1]
        if directive not in ["#if", "#ifdef", "#ifndef", "#elif"]: raise PreprocessorError(
            f"#elif 跟在 {directive} 之后")
        parent_active = len(self.conditional_stack) == 1 or self.conditional_stack[-2][1]
        condition_str = line[len("#elif"):].strip()  # Moved here to always have it for logging
        if previous_condition_met or not parent_active:
            currently_active = False
            condition_result = False  # Don't evaluate if already met or parent inactive
        else:
            condition_result = self._evaluate_condition(condition_str)
            currently_active = condition_result
        self.conditional_stack[-1] = ("#elif", currently_active, previous_condition_met or condition_result)
        logging.debug(
            f"#elif {condition_str} -> {'Active' if currently_active else 'Inactive'}, Parent Active: {parent_active}, Previous Met: {previous_condition_met}, Condition True: {condition_result}. Stack depth: {len(self.conditional_stack)}")

    def _handle_else(self, line):
        # （此函数内部逻辑未改变）
        if not self.conditional_stack: raise PreprocessorError("#else 没有匹配的 #if/#ifdef/#ifndef")
        directive, was_active, previous_condition_met = self.conditional_stack[-1]
        if directive not in ["#if", "#ifdef", "#ifndef", "#elif"]: raise PreprocessorError(
            f"#else 跟在 {directive} 之后")
        parent_active = len(self.conditional_stack) == 1 or self.conditional_stack[-2][1]
        currently_active = parent_active and (not previous_condition_met)
        self.conditional_stack[-1] = ("#else", currently_active, True)  # Mark condition as met
        logging.debug(
            f"#else -> {'Active' if currently_active else 'Inactive'}, Parent Active: {parent_active}, Previous Met: {previous_condition_met}. Stack depth: {len(self.conditional_stack)}")

    def _handle_endif(self, line):
        # （此函数内部逻辑未改变）
        if not self.conditional_stack: raise PreprocessorError("#endif 没有匹配的 #if/#ifdef/#ifndef")
        self.conditional_stack.pop()
        logging.debug(f"#endif. Stack depth: {len(self.conditional_stack)}")

    def _apply_object_macros(self, line):
        changed = True
        processed_line = line
        max_passes = 10
        passes = 0
        applied_in_pass = set()
        while changed and passes < max_passes:
            changed = False
            current_line = processed_line
            applied_in_pass.clear()
            # 按名称长度排序，优先匹配长名称
            sorted_macros = sorted(
                [m for m in self.macros.values() if not m.is_function_like],
                key=lambda m: len(m.name),
                reverse=True
            )
            next_scan_pos = 0
            temp_line = ""
            while next_scan_pos < len(current_line):
                found_match = False
                for macro in sorted_macros:
                    pattern = r'\b' + re.escape(macro.name) + r'\b'
                    match = re.match(pattern, current_line[next_scan_pos:])
                    if match:
                        logging.debug(f"Applying object macro '{macro.name}' for value '{macro.value}'")
                        temp_line += macro.value
                        next_scan_pos += match.end()
                        applied_in_pass.add(macro.name)
                        changed = True
                        found_match = True
                        break
                if not found_match:
                    temp_line += current_line[next_scan_pos]
                    next_scan_pos += 1
            processed_line = temp_line
            passes += 1
        if passes == max_passes:
            logging.warning(f"可能存在宏替换循环或嵌套过深: {line}")
        return processed_line

    def _apply_function_macros(self, line):
        """
        该函数的扫描方式是：
        第一层 (轮次循环): 控制对整行进行重复扫描和处理的次数，直到稳定或达上限。
        在每一轮内部:
        先 扫描整行，查找 所有已知函数式宏的 所有调用出现位置 并收集起来。
        将所有找到的 调用出现位置 按顺序 排序。
        遍历 排序后的 调用出现位置列表。对于列表中的每一个调用实例：
        执行参数解析：扫描 从该调用实例的左括号后开始的部分文本，直到找到匹配的右括号，同时识别参数分隔符。
        根据参数解析结果，决定是否执行宏展开。
        在处理完当前轮次找到的所有调用后，开始下一轮，直到不再有变化。
        """
        processed_line = line
        max_passes = 10
        passes = 0
        changed = True
        while changed and passes < max_passes:
            changed = False
            current_line = processed_line
            potential_calls = []
            for name, macro in self.macros.items():
                if not macro.is_function_like: continue
                pattern = r'\b' + re.escape(name) + r'\s*\('
                for match in re.finditer(pattern, current_line):
                    potential_calls.append(
                        {'name': name, 'macro': macro, 'start': match.start(), 'arg_start': match.end()})
            potential_calls.sort(key=lambda x: x['start'])
            result_line = ""
            last_pos = 0
            for call in potential_calls:
                # 避免处理已经被前一个宏展开覆盖的文本区域
                if call['start'] < last_pos: continue
                name = call['name']
                macro = call['macro']
                start_index = call['start']  # 宏名称开始的索引
                arg_list_start = call['arg_start']  # 参数列表开始的索引 (左括号 '(' 后面的位置)
                logging.debug(f"检测到可能的函数宏调用: {name} at index {start_index}")
                # --- 参数解析的初始化 ---
                args = []  # 存储解析出的参数字符串列表
                balance = 1  # 括号平衡计数器，初始为 1，因为我们已经在宏名称后的 '(' 里面
                current_arg = ""  # 存储当前正在构建的参数字符串
                scan_pos = arg_list_start  # 扫描位置，从参数列表开始处（'(' 后面）开始扫描
                # 状态标志，用于处理字符串字面量、字符字面量和转义字符
                in_string = False  # 是否在双引号字符串内
                in_char = False  # 是否在单引号字符字面量内
                escape = False  # 前一个字符是否是转义符 '\'
                # --- 参数解析的核心循环 ---
                while scan_pos < len(current_line):
                    char = current_line[scan_pos]  # 当前正在检查的字符
                    if escape:
                        # 如果前一个是转义符，当前字符是转义序列的一部分，直接添加到当前参数，并取消转义状态
                        current_arg += char
                        escape = False
                    elif char == '\\':
                        # 如果当前字符是转义符 '\'，添加到当前参数，并设置转义状态
                        current_arg += char
                        escape = True
                    elif in_string:
                        # 如果当前在双引号字符串内，直接添加到当前参数
                        current_arg += char
                        if char == '"':
                            # 如果遇到了双引号，且不在转义状态，则字符串结束
                            in_string = False
                    elif in_char:
                        # 如果当前在单引号字符字面量内，直接添加到当前参数
                        current_arg += char
                        if char == "'":
                            # 如果遇到了单引号，且不在转义状态，则字符字面量结束
                            in_char = False
                    elif char == '"':
                        # 如果遇到了双引号，且不在字符串或字符字面量内，表示字符串开始
                        current_arg += char
                        in_string = True
                    elif char == "'":
                        # 如果遇到了单引号，且不在字符串或字符字面量内，表示字符字面量开始
                        current_arg += char
                        in_char = True
                    elif char == '(':
                        # 如果遇到了左括号，且不在字符串或字符字面量内，括号平衡加一
                        balance += 1
                        current_arg += char  # 括号本身也是参数内容的一部分
                    elif char == ')':
                        # 如果遇到了右括号，且不在字符串或字符字面量内，括号平衡减一
                        balance -= 1
                        if balance == 0:
                            # 如果平衡变为 0，说明找到了与宏名称后左括号匹配的右括号，参数列表解析结束
                            # 将当前累积的最后一个参数添加到 args 列表
                            # current_arg.strip() 移除参数前后的空白
                            # 'or not args' 处理无参宏 MACRO() 的情况，此时 current_arg 为空，args 也为空，需要添加一个空字符串参数
                            if current_arg.strip() or not args:
                                args.append(current_arg.strip())
                            break  # 退出参数解析循环
                        else:
                            # 如果平衡不为 0，说明这是嵌套括号的一部分，添加到当前参数
                            current_arg += char
                    elif char == ',' and balance == 1:
                        # 如果遇到了逗号，且不在字符串或字符字面量内，并且是顶层括号 (balance == 1)，说明是参数分隔符
                        args.append(current_arg.strip())  # 将当前累积的参数添加到 args 列表 (移除空白)
                        current_arg = ""  # 重置 current_arg，开始累积下一个参数
                    else:
                        # 其他所有字符 (字母、数字、运算符、不在字符串内的空白等)，都是当前参数的一部分
                        current_arg += char
                    scan_pos += 1  # 移动到下一个字符

                if balance != 0:
                    logging.warning(f"函数宏调用括号不匹配: {name} starting at {start_index}. Skipping.")
                    result_line += current_line[last_pos:start_index + len(name)]
                    last_pos = start_index + len(name)
                else:
                    arg_list_end = scan_pos
                    call_str = current_line[start_index: arg_list_end + 1]
                    logging.debug(f"解析参数: {args} for call {call_str}")
                    if len(args) != len(macro.args):
                        logging.error(
                            f"宏 '{name}' 参数数量不匹配: 需要 {len(macro.args)}, 得到 {len(args)}. Skipping expansion.")
                        result_line += current_line[last_pos:arg_list_end + 1]  # Append the unexpanded call
                        last_pos = arg_list_end + 1
                    else:
                        expansion = macro.value
                        for param_name, provided_arg in reversed(list(zip(macro.args, args))):
                            expansion = expansion.replace(param_name, provided_arg)
                        logging.debug(f"展开 '{call_str}' 为 '{expansion}'")
                        result_line += current_line[last_pos:start_index]
                        result_line += expansion
                        last_pos = arg_list_end + 1
                        changed = True  # Mark that a change happened in this pass
            result_line += current_line[last_pos:]  # Append any remaining text
            processed_line = result_line
            passes += 1
        if passes == max_passes:
            logging.warning(f"函数宏展开可能达到最大嵌套/迭代次数 for line: {line}")
        return processed_line

    def process(self, input_code):
        """
        是否是预处理指令？
            如果是：
                根据指令类型调用相应的处理函数，更新宏或条件栈状态。检查是否在活跃块中决定某些指令是否生效。忽略 #include。在活跃块中遇到未知指令则报错。
            如果不是：
                当前是否在活跃块中 (_is_active())？
                    如果是：
                        应用宏替换，将处理后的行添加到输出列表，记录输出行索引和原始行号的映射，输出行索引加一。
                    如果不是：跳过此行，不做任何处理或记录。
        处理完所有行后，检查条件栈是否为空。最后，将输出行列表合并成字符串，并返回该字符串和行号映射字典。
        """
        self.macros.clear()
        self.conditional_stack.clear()
        self.line_mapping = []

        try:
            code = self._handle_line_continuation(input_code)
            code = self._remove_comments(code)
            lines = code.splitlines()

            output_lines = []
            current_output_line_index = 0

            for i, line in enumerate(lines):
                original_lineno = i + 1
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                if stripped_line.startswith('#'):
                    logging.debug(f"处理指令 (原始 L{original_lineno}): {stripped_line}")
                    if stripped_line.startswith('#define'):
                        if self._is_active(): self._parse_define(stripped_line)
                    elif stripped_line.startswith('#undef'):
                        if self._is_active(): self._handle_undef(stripped_line)
                    elif stripped_line.startswith('#ifdef'):
                        self._handle_ifdef(stripped_line)
                    elif stripped_line.startswith('#ifndef'):
                        self._handle_ifndef(stripped_line)
                    elif stripped_line.startswith('#if'):
                        self._handle_if(stripped_line)
                    elif stripped_line.startswith('#elif'):
                        self._handle_elif(stripped_line)
                    elif stripped_line.startswith('#else'):
                        self._handle_else(stripped_line)
                    elif stripped_line.startswith('#endif'):
                        self._handle_endif(stripped_line)
                    elif stripped_line.startswith('#include'):
                        logging.debug(f"移除 #include (原始 L{original_lineno}): {stripped_line}")
                        pass
                    elif stripped_line.startswith('#error') or stripped_line.startswith(
                            '#warning') or stripped_line.startswith('#pragma'):
                        if self._is_active():
                            logging.warning(f"忽略不支持的指令 (原始 L{original_lineno}): {stripped_line}")
                    else:
                        if self._is_active():
                            raise PreprocessorError(f"未知的预处理指令 (原始 L{original_lineno}): {stripped_line}")
                        else:
                            logging.debug(f"忽略非活动块中的未知指令: {stripped_line}")

                elif self._is_active():
                    logging.debug(f"处理代码 (原始 L{original_lineno}): {line}")
                    processed_line = self._apply_object_macros(line)
                    processed_line = self._apply_function_macros(processed_line)
                    output_lines.append(processed_line)
                    self.line_mapping.append((current_output_line_index, original_lineno))
                    current_output_line_index += 1
                else:
                    logging.debug(f"跳过非活动代码 (原始 L{original_lineno}): {line}")
            if self.conditional_stack:
                raise PreprocessorError(f"文件结束时有未闭合的条件编译块: 最顶层是 {self.conditional_stack[-1][0]}")
            line_mapping_dict = {idx + 1: orig_ln for idx, orig_ln in self.line_mapping}
            return '\n'.join(output_lines), line_mapping_dict

        except PreprocessorError as e:
            print(f"预处理错误: {e}", file=sys.stderr)
            raise
        except Exception as e:
            print(f"预处理时发生意外错误: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            raise


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python preprocess.py <input_file.cpp>")
        sys.exit(1)

    input_file_path = sys.argv[1]

    try:
        # Read the input file content
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            code = infile.read()
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file_path}' 未找到", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"读取文件 '{input_file_path}' 时发生错误: {e}", file=sys.stderr)
        sys.exit(1)
    preprocessor = BasicPreprocessor()
    try:
        processed_code, line_map = preprocessor.process(code)
        print("--- 预处理后的代码 ---")
        print(processed_code)
        print("\n--- 行号映射 (处理后行号 -> 原始行号) ---")
        if line_map:
            for processed_ln in sorted(line_map.keys()):
                original_ln = line_map[processed_ln]
                print(f"  处理后 L{processed_ln} -> 原始 L{original_ln}")
        else:
            print("  (无有效代码行输出)")

    except Exception as e:
        print("\n预处理失败。", file=sys.stderr)
        sys.exit(1)
