from utils.toolbox import update_ui, update_ui_lastest_msg    # 刷新Gradio前端界面
from utils.toolbox import zip_folder, objdump, objload, promote_file_to_downloadzone
from .latex_toolbox import PRESERVE, TRANSFORM
from .latex_toolbox import set_forbidden_text, set_forbidden_text_begin_end, set_forbidden_text_careful_brace
from .latex_toolbox import reverse_forbidden_text_careful_brace, reverse_forbidden_text, convert_to_linklist, post_process
from .latex_toolbox import fix_content, find_main_tex_file, merge_tex_files, compile_latex_with_timeout

import os, shutil
import re
import numpy as np

pj = os.path.join


def split_subprocess(txt, project_folder, return_dict, opts):
    """
    break down latex file to a linked list,
    each node use a preserve flag to indicate whether it should
    be proccessed by GPT.
    """
    text = txt
    mask = np.zeros(len(txt), dtype=np.uint8) + TRANSFORM

    # 吸收title与作者以上的部分
    text, mask = set_forbidden_text(text, mask, r"^(.*?)\\maketitle", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"^(.*?)\\begin{document}", re.DOTALL)
    # 吸收iffalse注释
    text, mask = set_forbidden_text(text, mask, r"\\iffalse(.*?)\\fi", re.DOTALL)
    # 吸收在42行以内的begin-end组合
    text, mask = set_forbidden_text_begin_end(text, mask, r"\\begin\{([a-z\*]*)\}(.*?)\\end\{\1\}", re.DOTALL, limit_n_lines=42)
    # 吸收匿名公式
    text, mask = set_forbidden_text(text, mask, [ r"\$\$([^$]+)\$\$",  r"\\\[.*?\\\]" ], re.DOTALL)
    # 吸收其他杂项
    text, mask = set_forbidden_text(text, mask, [ r"\\section\{(.*?)\}", r"\\section\*\{(.*?)\}", r"\\subsection\{(.*?)\}", r"\\subsubsection\{(.*?)\}" ])
    text, mask = set_forbidden_text(text, mask, [ r"\\bibliography\{(.*?)\}", r"\\bibliographystyle\{(.*?)\}" ])
    text, mask = set_forbidden_text(text, mask, r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{wraptable\}(.*?)\\end\{wraptable\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, r"\\begin\{algorithm\}(.*?)\\end\{algorithm\}", re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{wrapfigure\}(.*?)\\end\{wrapfigure\}", r"\\begin\{wrapfigure\*\}(.*?)\\end\{wrapfigure\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{figure\}(.*?)\\end\{figure\}", r"\\begin\{figure\*\}(.*?)\\end\{figure\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{multline\}(.*?)\\end\{multline\}", r"\\begin\{multline\*\}(.*?)\\end\{multline\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{table\}(.*?)\\end\{table\}", r"\\begin\{table\*\}(.*?)\\end\{table\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{minipage\}(.*?)\\end\{minipage\}", r"\\begin\{minipage\*\}(.*?)\\end\{minipage\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{align\*\}(.*?)\\end\{align\*\}", r"\\begin\{align\}(.*?)\\end\{align\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\begin\{equation\}(.*?)\\end\{equation\}", r"\\begin\{equation\*\}(.*?)\\end\{equation\*\}"], re.DOTALL)
    text, mask = set_forbidden_text(text, mask, [r"\\includepdf\[(.*?)\]\{(.*?)\}", r"\\clearpage", r"\\newpage", r"\\appendix", r"\\tableofcontents", r"\\include\{(.*?)\}"])
    text, mask = set_forbidden_text(text, mask, [r"\\vspace\{(.*?)\}", r"\\hspace\{(.*?)\}", r"\\label\{(.*?)\}", r"\\begin\{(.*?)\}", r"\\end\{(.*?)\}", r"\\item "])
    text, mask = set_forbidden_text_careful_brace(text, mask, r"\\hl\{(.*?)\}", re.DOTALL)
    # reverse 操作必须放在最后
    text, mask = reverse_forbidden_text_careful_brace(text, mask, r"\\caption\{(.*?)\}", re.DOTALL, forbid_wrapper=True)
    text, mask = reverse_forbidden_text_careful_brace(text, mask, r"\\abstract\{(.*?)\}", re.DOTALL, forbid_wrapper=True)
    text, mask = reverse_forbidden_text(text, mask, r"\\begin\{abstract\}(.*?)\\end\{abstract\}", re.DOTALL, forbid_wrapper=True)
    root = convert_to_linklist(text, mask)

    # 最后一步处理，增强稳健性
    root = post_process(root)

    # 输出html调试文件，用红色标注处保留区（PRESERVE），用黑色标注转换区（TRANSFORM）
    with open(pj(project_folder, 'debug_log.html'), 'w', encoding='utf8') as f:
        segment_parts_for_gpt = []
        nodes = []
        node = root
        while True:
            nodes.append(node)
            show_html = node.string.replace('\n','<br/>')
            if not node.preserve:
                segment_parts_for_gpt.append(node.string)
                f.write(f'<p style="color:black;">#{node.range}{show_html}#</p>')
            else:
                f.write(f'<p style="color:red;">{show_html}</p>')
            node = node.next
            if node is None: break

    for n in nodes: n.next = None   # break
    return_dict['nodes'] = nodes
    return_dict['segment_parts_for_gpt'] = segment_parts_for_gpt
    return return_dict

class LatexPaperSplit():
    """
    break down latex file to a linked list,
    each node use a preserve flag to indicate whether it should
    be proccessed by GPT.
    """
    def __init__(self) -> None:
        self.nodes = None
        self.msg = "*{\\scriptsize\\textbf{警告：该PDF由GPT-Academic开源项目调用大语言模型+Latex翻译插件一键生成，" + \
            "版权归原文作者所有。翻译内容可靠性无保障，请仔细鉴别并以原文为准。" + \
            "项目Github地址 \\url{https://github.com/binary-husky/gpt_academic/}。"
        # 请您不要删除或修改这行警告，除非您是论文的原作者（如果您是论文原作者，欢迎加REAME中的QQ联系开发者）
        self.msg_declare = "为了防止大语言模型的意外谬误产生扩散影响，禁止移除或修改此警告。}}\\\\" 


    def merge_result(self, arr, mode, msg, buggy_lines=[], buggy_line_surgery_n_lines=10):
        """
        Merge the result after the GPT process completed
        """
        result_string = ""
        node_cnt = 0
        line_cnt = 0
        
        for node in self.nodes:
            if node.preserve:
                line_cnt += node.string.count('\n')
                result_string += node.string
            else:
                translated_txt = fix_content(arr[node_cnt], node.string)
                begin_line = line_cnt
                end_line = line_cnt + translated_txt.count('\n')

                # reverse translation if any error
                if any([begin_line-buggy_line_surgery_n_lines <= b_line <= end_line+buggy_line_surgery_n_lines for b_line in buggy_lines]):
                    translated_txt = node.string

                result_string += translated_txt
                node_cnt += 1
                line_cnt += translated_txt.count('\n')

        if mode == 'translate_zh':
            pattern = re.compile(r'\\begin\{abstract\}.*\n')
            match = pattern.search(result_string)
            if not match:
                # match \abstract{xxxx}
                pattern_compile = re.compile(r"\\abstract\{(.*?)\}", flags=re.DOTALL)
                match = pattern_compile.search(result_string)
                position = match.regs[1][0]
            else:
                # match \begin{abstract}xxxx\end{abstract}
                position = match.end()
            result_string = result_string[:position] + self.msg + msg + self.msg_declare + result_string[position:]
        return result_string


    def split(self, txt, project_folder, opts): 
        """
        break down latex file to a linked list,
        each node use a preserve flag to indicate whether it should
        be proccessed by GPT.
        P.S. use multiprocessing to avoid timeout error
        """
        import multiprocessing
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        p = multiprocessing.Process(
            target=split_subprocess, 
            args=(txt, project_folder, return_dict, opts))
        p.start()
        p.join()
        p.close()
        self.nodes = return_dict['nodes']
        self.sp = return_dict['segment_parts_for_gpt']
        return self.sp


class LatexPaperFileGroup():
    """
    use tokenizer to break down text according to max_token_limit
    """
    def __init__(self):
        self.file_paths = []
        self.file_contents = []
        self.sp_file_contents = []
        self.sp_file_index = []
        self.sp_file_tag = []

        # count_token
        from request_llm.bridge_all import model_info
        enc = model_info["gpt-3.5-turbo"]['tokenizer']
        def get_token_num(txt): return len(enc.encode(txt, disallowed_special=()))
        self.get_token_num = get_token_num

    def run_file_split(self, max_token_limit=1900):
        """
        use tokenizer to break down text according to max_token_limit
        """
        for index, file_content in enumerate(self.file_contents):
            if self.get_token_num(file_content) < max_token_limit:
                self.sp_file_contents.append(file_content)
                self.sp_file_index.append(index)
                self.sp_file_tag.append(self.file_paths[index])
            else:
                from ..crazy_utils import breakdown_txt_to_satisfy_token_limit_for_pdf
                segments = breakdown_txt_to_satisfy_token_limit_for_pdf(file_content, self.get_token_num, max_token_limit)
                for j, segment in enumerate(segments):
                    self.sp_file_contents.append(segment)
                    self.sp_file_index.append(index)
                    self.sp_file_tag.append(self.file_paths[index] + f".part-{j}.tex")
        print('Segmentation: done')

    def merge_result(self):
        self.file_result = ["" for _ in range(len(self.file_paths))]
        for r, k in zip(self.sp_file_result, self.sp_file_index):
            self.file_result[k] += r

    def write_result(self):
        manifest = []
        for path, res in zip(self.file_paths, self.file_result):
            with open(path + '.polish.tex', 'w', encoding='utf8') as f:
                manifest.append(path + '.polish.tex')
                f.write(res)
        return manifest


def Latex精细分解与转化(file_manifest, project_folder, llm_kwargs, plugin_kwargs, chatbot, history, system_prompt, mode='proofread', switch_prompt=None, opts=[]):
    import time, os, re
    from ..crazy_utils import request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency
    from .latex_actions import LatexPaperFileGroup, LatexPaperSplit

    #  <-------- 寻找主tex文件 ----------> 
    maintex = find_main_tex_file(file_manifest, mode)
    chatbot.append((f"定位主Latex文件", f'[Local Message] 分析结果：该项目的Latex主文件是{maintex}, 如果分析错误, 请立即终止程序, 删除或修改歧义文件, 然后重试。主程序即将开始, 请稍候。'))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
    time.sleep(3)

    #  <-------- 读取Latex文件, 将多文件tex工程融合为一个巨型tex ----------> 
    main_tex_basename = os.path.basename(maintex)
    assert main_tex_basename.endswith('.tex')
    main_tex_basename_bare = main_tex_basename[:-4]
    may_exist_bbl = pj(project_folder, f'{main_tex_basename_bare}.bbl')
    if os.path.exists(may_exist_bbl):
        shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge.bbl'))
        shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge_{mode}.bbl'))
        shutil.copyfile(may_exist_bbl, pj(project_folder, f'merge_diff.bbl'))

    with open(maintex, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
        merged_content = merge_tex_files(project_folder, content, mode)

    with open(project_folder + '/merge.tex', 'w', encoding='utf-8', errors='replace') as f:
        f.write(merged_content)

    #  <-------- 精细切分latex文件 ----------> 
    chatbot.append((f"Latex文件融合完成", f'[Local Message] 正在精细切分latex文件，这需要一段时间计算，文档越长耗时越长，请耐心等待。'))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面
    lps = LatexPaperSplit()
    res = lps.split(merged_content, project_folder, opts) # 消耗时间的函数

    #  <-------- 拆分过长的latex片段 ----------> 
    pfg = LatexPaperFileGroup()
    for index, r in enumerate(res):
        pfg.file_paths.append('segment-' + str(index))
        pfg.file_contents.append(r)

    pfg.run_file_split(max_token_limit=1024)
    n_split = len(pfg.sp_file_contents)

    #  <-------- 根据需要切换prompt ----------> 
    inputs_array, sys_prompt_array = switch_prompt(pfg, mode)
    inputs_show_user_array = [f"{mode} {f}" for f in pfg.sp_file_tag]

    if os.path.exists(pj(project_folder,'temp.pkl')):

        #  <-------- 【仅调试】如果存在调试缓存文件，则跳过GPT请求环节 ----------> 
        pfg = objload(file=pj(project_folder,'temp.pkl'))

    else:
        #  <-------- gpt 多线程请求 ----------> 
        gpt_response_collection = yield from request_gpt_model_multi_threads_with_very_awesome_ui_and_high_efficiency(
            inputs_array=inputs_array,
            inputs_show_user_array=inputs_show_user_array,
            llm_kwargs=llm_kwargs,
            chatbot=chatbot,
            history_array=[[""] for _ in range(n_split)],
            sys_prompt_array=sys_prompt_array,
            # max_workers=5,  # 并行任务数量限制, 最多同时执行5个, 其他的排队等待
            scroller_max_len = 40
        )

        #  <-------- 文本碎片重组为完整的tex片段 ----------> 
        pfg.sp_file_result = []
        for i_say, gpt_say, orig_content in zip(gpt_response_collection[0::2], gpt_response_collection[1::2], pfg.sp_file_contents):
            pfg.sp_file_result.append(gpt_say)
        pfg.merge_result()

        # <-------- 临时存储用于调试 ----------> 
        pfg.get_token_num = None
        objdump(pfg, file=pj(project_folder,'temp.pkl'))

    write_html(pfg.sp_file_contents, pfg.sp_file_result, chatbot=chatbot, project_folder=project_folder)

    #  <-------- 写出文件 ----------> 
    msg = f"当前大语言模型: {llm_kwargs['llm_model']}，当前语言模型温度设定: {llm_kwargs['temperature']}。"
    final_tex = lps.merge_result(pfg.file_result, mode, msg)
    objdump((lps, pfg.file_result, mode, msg), file=pj(project_folder,'merge_result.pkl'))

    with open(project_folder + f'/merge_{mode}.tex', 'w', encoding='utf-8', errors='replace') as f:
        if mode != 'translate_zh' or "binary" in final_tex: f.write(final_tex)
        

    #  <-------- 整理结果, 退出 ----------> 
    chatbot.append((f"完成了吗？", 'GPT结果已输出, 即将编译PDF'))
    yield from update_ui(chatbot=chatbot, history=history) # 刷新界面

    #  <-------- 返回 ----------> 
    return project_folder + f'/merge_{mode}.tex'


def remove_buggy_lines(file_path, log_path, tex_name, tex_name_pure, n_fix, work_folder_modified, fixed_line=[]):
    try:
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            log = f.read()
        import re
        buggy_lines = re.findall(tex_name+':([0-9]{1,5}):', log)
        buggy_lines = [int(l) for l in buggy_lines]
        buggy_lines = sorted(buggy_lines)
        buggy_line = buggy_lines[0]-1
        print("reversing tex line that has errors", buggy_line)

        # 重组，逆转出错的段落
        if buggy_line not in fixed_line:
            fixed_line.append(buggy_line)

        lps, file_result, mode, msg = objload(file=pj(work_folder_modified,'merge_result.pkl'))
        final_tex = lps.merge_result(file_result, mode, msg, buggy_lines=fixed_line, buggy_line_surgery_n_lines=5*n_fix)

        with open(pj(work_folder_modified, f"{tex_name_pure}_fix_{n_fix}.tex"), 'w', encoding='utf-8', errors='replace') as f:
            f.write(final_tex)

        return True, f"{tex_name_pure}_fix_{n_fix}", buggy_lines
    except:
        print("Fatal error occurred, but we cannot identify error, please download zip, read latex log, and compile manually.")
        return False, -1, [-1]


def 编译Latex(chatbot, history, main_file_original, main_file_modified, work_folder_original, work_folder_modified, work_folder, mode='default'):
    import os, time
    n_fix = 1
    fixed_line = []
    max_try = 32
    chatbot.append([f"正在编译PDF文档", f'编译已经开始。当前工作路径为{work_folder}，如果程序停顿5分钟以上，请直接去该路径下取回翻译结果，或者重启之后再度尝试 ...']); yield from update_ui(chatbot=chatbot, history=history)
    chatbot.append([f"正在编译PDF文档", '...']); yield from update_ui(chatbot=chatbot, history=history); time.sleep(1); chatbot[-1] = list(chatbot[-1]) # 刷新界面
    yield from update_ui_lastest_msg('编译已经开始...', chatbot, history)   # 刷新Gradio前端界面

    while True:
        import os
        may_exist_bbl = pj(work_folder_modified, f'merge.bbl')
        target_bbl = pj(work_folder_modified, f'{main_file_modified}.bbl')
        if os.path.exists(may_exist_bbl) and not os.path.exists(target_bbl):
            shutil.copyfile(may_exist_bbl, target_bbl)

        # https://stackoverflow.com/questions/738755/dont-make-me-manually-abort-a-latex-compile-when-theres-an-error
        yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 编译原始PDF ...', chatbot, history)   # 刷新Gradio前端界面
        ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_original}.tex', work_folder_original)

        yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 编译转化后的PDF ...', chatbot, history)   # 刷新Gradio前端界面
        ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_modified}.tex', work_folder_modified)
        
        if ok and os.path.exists(pj(work_folder_modified, f'{main_file_modified}.pdf')):
            # 只有第二步成功，才能继续下面的步骤
            yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 编译BibTex ...', chatbot, history)    # 刷新Gradio前端界面
            if not os.path.exists(pj(work_folder_original, f'{main_file_original}.bbl')):
                ok = compile_latex_with_timeout(f'bibtex  {main_file_original}.aux', work_folder_original)
            if not os.path.exists(pj(work_folder_modified, f'{main_file_modified}.bbl')):
                ok = compile_latex_with_timeout(f'bibtex  {main_file_modified}.aux', work_folder_modified)

            yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 编译文献交叉引用 ...', chatbot, history)  # 刷新Gradio前端界面
            ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_original}.tex', work_folder_original)
            ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_modified}.tex', work_folder_modified)
            ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_original}.tex', work_folder_original)
            ok = compile_latex_with_timeout(f'pdflatex -interaction=batchmode -file-line-error {main_file_modified}.tex', work_folder_modified)

            if mode!='translate_zh':
                yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 使用latexdiff生成论文转化前后对比 ...', chatbot, history) # 刷新Gradio前端界面
                print(    f'latexdiff --encoding=utf8 --append-safecmd=subfile {work_folder_original}/{main_file_original}.tex  {work_folder_modified}/{main_file_modified}.tex --flatten > {work_folder}/merge_diff.tex')
                ok = compile_latex_with_timeout(f'latexdiff --encoding=utf8 --append-safecmd=subfile {work_folder_original}/{main_file_original}.tex  {work_folder_modified}/{main_file_modified}.tex --flatten > {work_folder}/merge_diff.tex')

                yield from update_ui_lastest_msg(f'尝试第 {n_fix}/{max_try} 次编译, 正在编译对比PDF ...', chatbot, history)   # 刷新Gradio前端界面
                ok = compile_latex_with_timeout(f'pdflatex  -interaction=batchmode -file-line-error merge_diff.tex', work_folder)
                ok = compile_latex_with_timeout(f'bibtex    merge_diff.aux', work_folder)
                ok = compile_latex_with_timeout(f'pdflatex  -interaction=batchmode -file-line-error merge_diff.tex', work_folder)
                ok = compile_latex_with_timeout(f'pdflatex  -interaction=batchmode -file-line-error merge_diff.tex', work_folder)

        # <---------- 检查结果 ----------->
        results_ = ""
        original_pdf_success = os.path.exists(pj(work_folder_original, f'{main_file_original}.pdf'))
        modified_pdf_success = os.path.exists(pj(work_folder_modified, f'{main_file_modified}.pdf'))
        diff_pdf_success     = os.path.exists(pj(work_folder, f'merge_diff.pdf'))
        results_ += f"原始PDF编译是否成功: {original_pdf_success};" 
        results_ += f"转化PDF编译是否成功: {modified_pdf_success};" 
        results_ += f"对比PDF编译是否成功: {diff_pdf_success};" 
        yield from update_ui_lastest_msg(f'第{n_fix}编译结束:<br/>{results_}...', chatbot, history) # 刷新Gradio前端界面

        if diff_pdf_success:
            result_pdf = pj(work_folder_modified, f'merge_diff.pdf')    # get pdf path
            promote_file_to_downloadzone(result_pdf, rename_file=None, chatbot=chatbot)  # promote file to web UI
        if modified_pdf_success:
            yield from update_ui_lastest_msg(f'转化PDF编译已经成功, 即将退出 ...', chatbot, history)    # 刷新Gradio前端界面
            result_pdf = pj(work_folder_modified, f'{main_file_modified}.pdf') # get pdf path
            origin_pdf = pj(work_folder_original, f'{main_file_original}.pdf') # get pdf path
            if os.path.exists(pj(work_folder, '..', 'translation')):
                shutil.copyfile(result_pdf, pj(work_folder, '..', 'translation', 'translate_zh.pdf'))
            promote_file_to_downloadzone(result_pdf, rename_file=None, chatbot=chatbot)  # promote file to web UI
            # 将两个PDF拼接
            if original_pdf_success: 
                try:
                    from .latex_toolbox import merge_pdfs
                    concat_pdf = pj(work_folder_modified, f'comparison.pdf')
                    merge_pdfs(origin_pdf, result_pdf, concat_pdf)
                    promote_file_to_downloadzone(concat_pdf, rename_file=None, chatbot=chatbot)  # promote file to web UI
                except Exception as e:
                    pass
            return True # 成功啦
        else:
            if n_fix>=max_try: break
            n_fix += 1
            can_retry, main_file_modified, buggy_lines = remove_buggy_lines(
                file_path=pj(work_folder_modified, f'{main_file_modified}.tex'), 
                log_path=pj(work_folder_modified, f'{main_file_modified}.log'),
                tex_name=f'{main_file_modified}.tex',
                tex_name_pure=f'{main_file_modified}',
                n_fix=n_fix,
                work_folder_modified=work_folder_modified,
                fixed_line=fixed_line
            )
            yield from update_ui_lastest_msg(f'由于最为关键的转化PDF编译失败, 将根据报错信息修正tex源文件并重试, 当前报错的latex代码处于第{buggy_lines}行 ...', chatbot, history)   # 刷新Gradio前端界面
            if not can_retry: break

    return False # 失败啦


def write_html(sp_file_contents, sp_file_result, chatbot, project_folder):
    # write html
    try:
        import shutil
        from ..crazy_utils import construct_html
        from toolbox import gen_time_str
        ch = construct_html() 
        orig = ""
        trans = ""
        final = []
        for c,r in zip(sp_file_contents, sp_file_result): 
            final.append(c)
            final.append(r)
        for i, k in enumerate(final): 
            if i%2==0:
                orig = k
            if i%2==1:
                trans = k
                ch.add_row(a=orig, b=trans)
        create_report_file_name = f"{gen_time_str()}.trans.html"
        ch.save_file(create_report_file_name)
        shutil.copyfile(pj('./gpt_log/', create_report_file_name), pj(project_folder, create_report_file_name))
        promote_file_to_downloadzone(file=f'./gpt_log/{create_report_file_name}', chatbot=chatbot)
    except:
        from toolbox import trimmed_format_exc
        print('writing html result failed:', trimmed_format_exc())
