# [step 1]>> 例如： API_KEY = "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" （此key无效）
API_KEY = "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"

# [step 2]>> 改为True应用代理，如果直接在海外服务器部署，此处不修改
USE_PROXY = True
if USE_PROXY:
    # 填写格式是 [协议]://  [地址] :[端口]，填写之前不要忘记把USE_PROXY改成True，如果直接在海外服务器部署，此处不修改
    # 例如    "socks5h://localhost:11284"
    # [协议] 常见协议无非socks5h/http; 例如 v2**y 和 ss* 的默认本地协议是socks5h; 而cl**h 的默认本地协议是http
    # [地址] 懂的都懂，不懂就填localhost或者127.0.0.1肯定错不了（localhost意思是代理软件安装在本机上）
    # [端口] 在代理软件的设置里找。虽然不同的代理软件界面不一样，但端口号都应该在最显眼的位置上

    # 代理网络的地址，打开你的科学上网软件查看代理的协议(socks5/http)、地址(localhost)和端口(11284)
    proxies = {
        #          [协议]://  [地址]  :[端口]
        "http":  "socks5h://localhost:4781",
        "https": "socks5h://localhost:4781",
    }
else:
    proxies = None

PROXIES = "socks5h://localhost:4781"

# 多线程函数插件中，默认允许多少路线程同时访问OpenAI。
# Free trial users的限制是每分钟3次，Pay-as-you-go users的限制是每分钟3500次。提高限制请查询：
# https://platform.openai.com/docs/guides/rate-limits/overview
DEFAULT_WORKER_NUM = 3

# [step 3]>> 以下配置可以优化体验，但大部分场合下并不需要修改
# 对话窗的高度
CHATBOT_HEIGHT = 1110

# 代码高亮
CODE_HIGHLIGHT = True

# 窗口布局
LAYOUT = "LEFT-RIGHT"  # "LEFT-RIGHT"（左右布局） # "TOP-DOWN"（上下布局）

# 发送请求到OpenAI后，等待多久判定为超时
TIMEOUT_SECONDS = 30

# 网页的端口, -1代表随机端口
WEB_PORT = -1

# 如果OpenAI不响应（网络卡顿、代理失败、KEY失效），重试的次数限制
MAX_RETRY = 2

# OpenAI模型选择是（gpt4现在只对申请成功的人开放）
LLM_MODEL = "gpt-3.5-turbo"

# OpenAI的API_URL
API_URL = "https://api.openai.com/v1/chat/completions"

# 设置并行使用的线程数
CONCURRENT_COUNT = 100

NUM_WORKS=2

# set model an model path 
GROUNED_MODEL_TYPE= {'S': "../Grounded-Segment-Anything-main/groundingdino_swint_ogc.pth",'L':None}
SAM_MODEL_TYPE= {'vit_h': "F:\\sam_vit_h_4b8939.pth"  ,'vit_l':None,'vit_b':None}
Tag2Text_Model_Path='F:\\tag2text_swin_14m.pth'#'weights/tag2text/tag2text_swin_14m.pth'
LAMA_MODEL_PATH='F:\\Inpaint-Anything\\big-lama'