@echo off
setlocal

chcp 65001 >nul
pushd "%~dp0" || exit /b 1

set "AI_CHAT_UTIL_CONFIG=%~dp0config.yml"

rem 分析対象のファイル1つ目を指定
set "input_file1="
if not "%~1"=="" (
	set "input_file1=%~1"
) else (
	set /p "input_file1=分析対象のファイル1つ目を指定してください（例: ..\..\..\work\test\data\test.pdf）: "
)

rem 分析対象のファイル2つ目を指定
set "input_file2="
if not "%~2"=="" (
	set "input_file2=%~2"
) else (
	set /p "input_file2=分析対象のファイル2つ目を指定してください（例: ..\..\..\work\test\data\test.png）: "
)

if "%input_file1%"=="" (
	echo [ERROR] input_file1 が未指定です。
	popd
	exit /b 1
)
if "%input_file2%"=="" (
	echo [ERROR] input_file2 が未指定です。
	popd
	exit /b 1
)

uv run -m ai_chat_util.cli --config "%AI_CHAT_UTIL_CONFIG%" analyze_files -i "%input_file1%" "%input_file2%" -p "指定したファイルを分析して"

set "RET=%ERRORLEVEL%"
popd
exit /b %RET%
