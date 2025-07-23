#!/usr/bin/env python3
"""
Launcher for Lean 4 Streamlit Interfaces

Simple script to help users launch the appropriate Streamlit interface.
"""

import subprocess
import sys
import os

def main():
    print("🧮 LEAN 4 STREAMLIT INTERFACES")
    print("=" * 50)
    print()
    print("Choose which interface to launch:")
    print("1. 🧮 Combined Interface (recommended) - Panel viewer + Agentic editor")
    print("2. 🔍 Panel Viewer Only - VS Code-style proof state viewer")
    print("3. 🤖 Agentic Editor Only - Interactive editing interface")
    print()

    while True:
        try:
            choice = input("Enter choice (1-3) or 'q' to quit: ").strip()

            if choice == 'q':
                print("👋 Goodbye!")
                break

            elif choice == '1':
                print("🚀 Launching combined interface...")
                print("   Open your browser to: http://localhost:8501")
                print("   Press Ctrl+C to stop the server")
                print()
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
                break

            elif choice == '2':
                print("🚀 Launching panel viewer...")
                print("   Open your browser to: http://localhost:8501")
                print("   Press Ctrl+C to stop the server")
                print()
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_panel_viewer.py"])
                break

            elif choice == '3':
                print("🚀 Launching agentic editor...")
                print("   Open your browser to: http://localhost:8501")
                print("   Press Ctrl+C to stop the server")
                print()
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_agentic_editor.py"])
                break

            else:
                print("❌ Invalid choice. Please enter 1-3 or 'q'.")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error launching Streamlit: {e}")
            print("Make sure Streamlit is installed and you're in the correct directory.")
            break

if __name__ == "__main__":
    main()
