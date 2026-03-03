#!/usr/bin/env python3
"""Quick version that looks back 7 days instead of 2"""
import os
import sys

# Modify the original script to look back 7 days
with open('production_websearch_detection.py', 'r') as f:
    content = f.read()

# Replace days_back=2 with days_back=7
content = content.replace('days_back: int = 2', 'days_back: int = 7')
content = content.replace('days_back=2', 'days_back=7')

exec(content)
