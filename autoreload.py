""" A simple autoreload script using watchdog.

For some reason gradio's built-in autoreload doesn't seem to work properly hence this script.
It just uses watchdog to rerun app.py whenever any .py file (except this one) is modified.
"""
import subprocess
import sys
import time
import os
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler

process = None
def start_app():
    global process
    if process is not None:
        print("\n🔄 Terminating previous process...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Process didn't terminate, forcing kill...")
            process.kill()
            process.wait()
    
    print("🚀 Starting app.py...")
    process = subprocess.Popen([sys.executable, 'app.py'])

class ReloadHandler(PatternMatchingEventHandler):
    def __init__(self):
        # Monitor all .py files except autoreload.py
        super().__init__(
            patterns=['*.py'],
            ignore_patterns=['autoreload.py'],
            ignore_directories=True,
            case_sensitive=False
        )
        self.last_modified = {}
    
    def on_modified(self, event):
        # Debounce: ignore rapid successive modifications
        current_time = time.time()
        if event.src_path in self.last_modified:
            if current_time - self.last_modified[event.src_path] < 1:
                return
        
        self.last_modified[event.src_path] = current_time
        print(f"\n📝 Change detected in: {os.path.basename(event.src_path)}")
        start_app()

# Start the app initially
start_app()

# Setting up watchdog to monitor all .py files in the directory
observer = Observer()
handler = ReloadHandler()
observer.schedule(
    handler,
    path=".",
    recursive=True
)
observer.start()

try:
    print("\n👀 Watching for changes in .py files (excluding autoreload.py)...")
    print("Press Ctrl+C to stop\n")
    
    while True:
        time.sleep(1)
        
except KeyboardInterrupt:
    print("\n\n🛑 Shutting down...")
    observer.stop()
    observer.join()
    if process is not None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    print("✅ Cleanup complete")