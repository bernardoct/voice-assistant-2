#!/usr/bin/env python3
"""
Entry point for Jetson server.
"""

import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jetson.server import main

if __name__ == "__main__":
    asyncio.run(main())
