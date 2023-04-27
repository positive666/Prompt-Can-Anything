import re
import markdown
from typing import Set

from prompt_toolkit import prompt
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings

bindings = KeyBindings()

def write_results_to_file(history, file_name=None):
    """
        将对话记录history以Markdown格式写入文件中。如果没有指定文件名，则使用当前时间生成文件名。
    """
    import os
    import time
    if file_name is None:
        # file_name = time.strftime("chatGPT-log%Y-%m-%d-%H-%M-%S", time.localtime()) + '.md'
        file_name = 'chatGPT输出报告' + \
            time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.md'
    os.makedirs('./gpt_log/', exist_ok=True)
    with open(f'./gpt_log/{file_name}', 'w', encoding='utf8') as f:
        f.write('# chatGPT 分析报告\n')
        for i, content in enumerate(history):
            try:    # 这个bug没找到触发条件，暂时先这样顶一下
                if type(content) != str:
                    content = str(content)
            except:
                continue
            if i % 2 == 0:
                f.write('## ')
            f.write(content)
            f.write('\n\n')
    res = '以上材料已经被写入' + os.path.abspath(f'./gpt_log/{file_name}')
    print(res)
    return res

def create_keybindings(key: str = "c-@") -> KeyBindings:
    """
    Create keybindings for prompt_toolkit. Default key is ctrl+space.
    For possible keybindings, see: https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html#list-of-special-keys
    """

    @bindings.add(key)
    def _(event: dict) -> None:
        event.app.exit(result=event.app.current_buffer.text)

    return bindings


def create_session() -> PromptSession:
    return PromptSession(history=InMemoryHistory())


def create_completer(commands: list, pattern_str: str = "$") -> WordCompleter:
    return WordCompleter(words=commands, pattern=re.compile(pattern_str))


def get_input(
    session: PromptSession = None,
    completer: WordCompleter = None,
    key_bindings: KeyBindings = None,
) -> str:
    """
    Multiline input function.
    """
    return (
        session.prompt(
            completer=completer,
            multiline=True,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=key_bindings,
        )
        if session
        else prompt(multiline=True)
    )


async def get_input_async(
    session: PromptSession = None,
    completer: WordCompleter = None,
) -> str:
    """
    Multiline input function.
    """
    return (
        await session.prompt_async(
            completer=completer,
            multiline=True,
            auto_suggest=AutoSuggestFromHistory(),
        )
        if session
        else prompt(multiline=True)
    )


def get_filtered_keys_from_object(obj: object, *keys: str) -> Set[str]:
    """
    Get filtered list of object variable names.
    :param keys: List of keys to include. If the first key is "not", the remaining keys will be removed from the class keys.
    :return: List of class keys.
    """
    class_keys = obj.__dict__.keys()
    if not keys:
        return set(class_keys)

    # Remove the passed keys from the class keys.
    if keys[0] == "not":
        return {key for key in class_keys if key not in keys[1:]}
    # Check if all passed keys are valid
    if invalid_keys := set(keys) - class_keys:
        raise ValueError(
            f"Invalid keys: {invalid_keys}",
        )
    # Only return specified keys that are in class_keys
    return {key for key in keys if key in class_keys}
