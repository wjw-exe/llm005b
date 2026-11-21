@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
cd /d "%~dp0"
title llm005b - 0.05B 对话语言模型（纯 numpy 手写算子）

set "PY="
call :find_python
if not defined PY (
    echo.
    echo [错误] 没有找到可用的 Python 3.10+ 解释器。
    echo        请先安装 Python 3.10 或更高版本（安装时勾选 Add python.exe to PATH）：
    echo        https://www.python.org/downloads/windows/
    echo.
    pause
    exit /b 1
)

rem ---------- 依赖检查（只允许 numpy） ----------
"%PY%" -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [依赖] 未检测到 numpy，正在自动安装…
    "%PY%" -m pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
    "%PY%" -c "import numpy" >nul 2>&1
    if errorlevel 1 (
        echo [错误] numpy 安装失败，请手动执行： "%PY%" -m pip install numpy
        pause
        exit /b 1
    )
)
echo [就绪] Python: %PY%
"%PY%" -c "import numpy;print('[就绪] numpy: '+numpy.__version__+'  (本项目不使用 torch/transformers 等任何深度学习框架)')"

:menu
echo.
echo ============================================================
echo   llm005b  ·  对话语言模型（手写 numpy 算子，零框架）
echo ============================================================
echo   1  0.05B 指令模型对话  [首选 英文直连]  公开 50M 指令微调权重 + 手写算子
echo   2  自训练模型对话             工程内从零训练出的 46M 权重
echo   3  训练模型                   train.py 后台训练 + 日志
echo   4  梯度校验                   grad_check.py 数值梯度自检
echo   5  准备语料                   download_corpus.py + prepare.py
echo   6  查看训练日志               logs\train.log 末尾
echo   7  中文增强对话（公共翻译 API）  [需联网，公共接口较慢]
echo   0  退出
echo ============================================================
set "CHOICE="
set /p "CHOICE=请输入序号后回车（直接回车默认 1）: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" goto :chat_instruct
if "%CHOICE%"=="2" goto :chat
if "%CHOICE%"=="3" goto :train
if "%CHOICE%"=="4" goto :grad
if "%CHOICE%"=="5" goto :data
if "%CHOICE%"=="6" goto :log
if "%CHOICE%"=="7" goto :chat_instruct_zh
if "%CHOICE%"=="0" goto :eof
echo 无效输入，请重新选择。
goto :menu

:chat_instruct
echo.
if not exist "model_50m_instruct\model.safetensors" (
    echo [错误] 未找到指令模型权重：model_50m_instruct\model.safetensors
    echo        该目录应含 config.json / model.safetensors / tokenizer.json 等文件。
    pause
    goto :menu
)
echo [对话] 0.05B 指令微调模型启动中…（加载 212MB 权重约 2-3 秒）
"%PY%" -X utf8 chat.py --model instruct
echo.
pause
goto :menu

:chat_instruct_zh
echo.
if not exist "model_50m_instruct\model.safetensors" (
    echo [错误] 未找到指令模型权重：model_50m_instruct\model.safetensors
    echo        该目录应含 config.json / model.safetensors / tokenizer.json 等文件。
    pause
    goto :menu
)
echo ============================================================
echo  中文增强对话（公共翻译 API）
echo  链路：中文输入 -^> 译成英文 -^> 0.05B 指令模型推理 -^> 英文回答 -^> 回译中文回显
echo  提供方：MyMemory 免费接口（免 key，单次 500 字符，约 2-4 秒/次翻译）
echo  注意：需联网；公共接口较慢，每轮比纯英文多 2 次翻译等待；
echo        翻译失败会自动降级为原文直通，对话不会中断。
echo  切换：/translate off 回到纯英文直连   /raw on^|off 显示或隐藏英文原文
echo ============================================================
"%PY%" -X utf8 chat.py --model instruct --translate --provider mymemory
echo.
pause
goto :menu

:chat
echo.
echo [对话] 自训练模型启动中…（首次加载约 46M 参数需要几秒）
"%PY%" -X utf8 chat.py --model scratch
echo.
pause
goto :menu

:train
echo.
if not exist "data\train.bin" (
    echo [提示] 还没有编码好的语料 data\train.bin，将先执行语料准备。
    "%PY%" -X utf8 prepare.py
)
set "STEPS="
set /p "STEPS=请输入训练步数（直接回车默认 400 步）: "
if "%STEPS%"=="" set "STEPS=400"
echo [训练] 后台启动 %STEPS% 步训练（已有 ckpt 则自动续训），前台不会阻塞…
"%PY%" -X utf8 train.py --steps %STEPS% --batch 8 --lr 1.2e-3 --warmup 50 --resume --background
echo.
echo [训练] 日志文件：logs\train.log     （菜单 5 可查看末尾）
echo        训练完成后权重位于 ckpt\final.npz
pause
goto :menu

:grad
echo.
echo [梯度校验] 用有限差分对比手写反向传播…
"%PY%" -X utf8 grad_check.py
echo.
pause
goto :menu

:data
echo.
echo [语料] 步骤 1/2：从 hf-mirror.com 下载对话语料（带超时，失败自动换源）
"%PY%" -X utf8 tools\download_corpus.py --background
echo        进度见 logs\download.log，完成后按任意键继续步骤 2。
pause >nul
echo [语料] 步骤 2/2：训练 BPE 并编码
"%PY%" -X utf8 prepare.py
pause
goto :menu

:log
echo.
if not exist "logs\train.log" (
    echo 还没有训练日志，请先选择 2 训练模型。
) else (
    echo ---- logs\train.log 末尾 20 行 ----
    "%PY%" -X utf8 -c "import io;print(''.join(io.open(r'logs/train.log',encoding='utf-8',errors='replace').readlines()[-20:]))"
)
echo.
pause
goto :menu

rem ---------- 定位 Python 解释器 ----------
:find_python
if defined PYTHON if exist "%PYTHON%" set "PY=%PYTHON%"
if defined PY goto :eof

for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined PY set "PY=%%p"
)
if defined PY goto :eof

where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%p in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do (
        if not defined PY set "PY=%%p"
    )
)
if defined PY goto :eof

for %%d in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
) do (
    if not defined PY if exist %%d set "PY=%%~d"
)
if defined PY goto :eof

for /f "delims=" %%p in ('dir /b /s "C:\ruanjian\Marvis\MarvisAgent\*\runtime\python311\python.exe" 2^>nul') do (
    if not defined PY set "PY=%%p"
)
goto :eof
