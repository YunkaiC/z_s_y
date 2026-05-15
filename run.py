#!/bin/bash
# 争上游 - 启动服务器
# 使用方法: python run.py [port]
# 默认端口: 8000

import sys
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Install dependencies if needed
try:
    import fastapi
    import uvicorn
except ImportError:
    print("正在安装依赖...")
    os.system(f"{sys.executable} -m pip install --break-system-packages fastapi uvicorn")

# Get port
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000

# Update server.py port
import server
print(f"\n{'='*50}")
print(f"  争上游 - 局域网联机服务器")
print(f"  访问地址: http://<你的局域网IP>:{port}")
print(f"{'='*50}\n")

import uvicorn
uvicorn.run(server.app, host="0.0.0.0", port=port)
