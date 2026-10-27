"""run_demo.bat 가 띄운 시연 화면을 서버가 뜬 뒤 브라우저로 연다.

Streamlit 은 headless 로 둔다 (첫 실행 때 이메일 입력을 기다리며 멈추지 않도록).
그래서 브라우저는 여기서 따로 연다. 60초 안에 서버가 안 뜨면 열지 않는다.
"""

import socket
import sys
import time
import webbrowser

URL = "http://localhost:8501"

for _ in range(60):
    try:
        socket.create_connection(("localhost", 8501), timeout=1).close()
    except OSError:
        time.sleep(1)
        continue
    webbrowser.open(URL)
    sys.exit(0)
sys.exit(1)
