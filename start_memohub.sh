#!/bin/bash
# MemoHub OneClick起動スクリプト
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run memohub.py
