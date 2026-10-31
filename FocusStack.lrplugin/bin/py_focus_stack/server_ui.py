"""
Interactive Focus Stacking Web GUI Server
Allows real-time 100% ROI preview, interactive slider tuning, depth map visualization,
manual retouching brush override, and final 61MP 16-bit render.
"""
import os
import io
import sys
import time
import json
import base64
import subprocess
import cv2
import numpy as np
from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import http.client
import urllib.parse
import webbrowser
import threading

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<title>Open Focus Stack Pro — EG-MSD Interactive Tuning</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📷</text></svg>">
<style>
  :root {
    --bg: #121214;
    --card: #1c1c20;
    --border: #2e2e34;
    --accent: #0070f3;
    --accent-hover: #0060df;
    --text: #eaeaea;
    --muted: #888892;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    height: 100vh;
    overflow: hidden;
    user-select: none;
  }
  #sidebar {
    width: 380px;
    background: var(--card);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 18px;
    gap: 14px;
    overflow-y: auto;
  }
  h1 { font-size: 17px; font-weight: 600; display: flex; align-items: center; gap: 8px; }
  .badge { background: #2563eb22; color: #60a5fa; font-size: 11px; padding: 3px 8px; border-radius: 12px; font-weight: 500; }
  .control-group { display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 12px; font-weight: 600; color: var(--text); display: flex; justify-content: space-between; text-transform: uppercase; letter-spacing: 0.5px; }
  .val { color: var(--accent); font-family: monospace; font-size: 13px; }
  input[type="range"] {
    width: 100%;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .desc { font-size: 11px; color: var(--muted); line-height: 1.4; }
  
  #navigator-container {
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: #151518;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid var(--border);
  }
  #minimap-wrapper {
    position: relative;
    width: 100%;
    height: 130px;
    background: #0d0d0f;
    border-radius: 6px;
    overflow: hidden;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: crosshair;
  }
  #minimap-img {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
    pointer-events: none;
  }
  #minimap-box {
    position: absolute;
    border: 2px solid #38bdf8;
    background: rgba(56, 189, 248, 0.25);
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.7);
    border-radius: 2px;
    cursor: move;
    box-sizing: border-box;
  }
  .btn-center-crop {
    background: #27272a;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 5px 9px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    align-self: flex-start;
  }
  .btn-center-crop:hover { background: #3f3f46; }

  /* Retouching Panel Styles */
  .retouch-box {
    background: #151519;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .retouch-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .btn-tool {
    background: #27272a;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-tool:hover { background: #3a3a42; }
  .btn-tool.active {
    background: #2563eb;
    color: #fff;
    border-color: #3b82f6;
    box-shadow: 0 0 8px rgba(37, 99, 235, 0.4);
  }
  .btn-step {
    background: #27272a;
    color: var(--text);
    border: 1px solid var(--border);
    width: 28px;
    height: 24px;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 11px;
  }
  .btn-step:hover { background: #3a3a42; }
  .btn-blink {
    background: #202026;
    color: #93c5fd;
    border: 1px dashed #3b82f666;
    padding: 6px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
  }
  .btn-blink:active, .btn-blink.active {
    background: #1e3a8a;
    color: #fff;
    border-style: solid;
  }
  .btn-action-small {
    flex: 1;
    background: #222228;
    color: #a1a1aa;
    border: 1px solid var(--border);
    padding: 6px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
  }
  .btn-action-small:hover { background: #2f2f38; color: #fff; }

  /* Collapsible Algorithm Controls Section */
  .algo-collapsible {
    background: #151519;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.2s ease;
  }
  .algo-collapsible[open] {
    border-color: #3b82f644;
    background: #16161b;
  }
  .algo-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 10px 12px;
    cursor: pointer;
    background: #18181e;
    list-style: none;
    font-size: 11px;
    font-weight: 700;
    color: var(--text);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    user-select: none;
    transition: background 0.15s ease;
  }
  .algo-summary::-webkit-details-marker {
    display: none;
  }
  .algo-summary:hover {
    background: #22222a;
    color: #fff;
  }
  .summary-left {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .summary-chevron {
    font-size: 9px;
    color: var(--muted);
    transition: transform 0.2s ease;
    display: inline-block;
  }
  .algo-collapsible[open] .summary-chevron {
    transform: rotate(90deg);
    color: var(--accent);
  }
  .summary-badge {
    font-size: 10px;
    color: #60a5fa;
    background: #2563eb22;
    padding: 2px 7px;
    border-radius: 10px;
    font-weight: 600;
    text-transform: none;
    letter-spacing: normal;
  }
  .algo-collapsible-content {
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    border-top: 1px solid var(--border);
    background: #131317;
  }

  .btn-row { display: flex; flex-direction: column; gap: 8px; margin-top: auto; padding-top: 14px; border-top: 1px solid var(--border); }
  button {
    padding: 11px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all 0.15s ease;
  }
  .btn-render {
    background: #2563eb;
    color: #fff;
  }
  .btn-render:hover { background: #1d4ed8; }
  .btn-cancel {
    background: transparent;
    color: #ef4444;
    border: 1px solid #ef444444;
  }
  .btn-cancel:hover { background: #ef444422; }
  
  #main {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg);
    position: relative;
  }
  #toolbar {
    height: 48px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    font-size: 13px;
    color: var(--muted);
  }
  #live-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    font-weight: 600;
    color: #22c55e;
    background: rgba(34, 197, 94, 0.1);
    padding: 3px 8px;
    border-radius: 10px;
  }
  .pulse-dot {
    width: 6px;
    height: 6px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0% { transform: scale(0.9); opacity: 0.7; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.7; }
  }

  #viewport {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: radial-gradient(circle at center, #1e1e24 0%, #121214 100%);
    position: relative;
  }
  #canvas-container {
    position: relative;
    display: inline-block;
    box-shadow: 0 20px 40px rgba(0,0,0,0.6);
    border-radius: 4px;
    line-height: 0;
    overflow: hidden;
  }
  #canvas-container.mode-fit {
    cursor: zoom-in;
  }
  #canvas-container.mode-roi {
    cursor: grab;
  }
  #canvas-container.mode-roi.grabbing, #canvas-container.grabbing {
    cursor: grabbing !important;
  }
  #pan-overview-img {
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
    user-select: none;
    -webkit-user-drag: none;
    display: none;
    z-index: 1;
    image-rendering: auto;
  }
  #preview-img {
    max-width: calc(100vw - 380px);
    max-height: calc(100vh - 80px);
    width: auto;
    height: auto;
    border-radius: 4px;
    display: none;
    position: relative;
    z-index: 2;
    transition: opacity 0.15s ease;
    user-select: none;
    -webkit-user-drag: none;
  }
  #preview-img.loaded {
    display: block;
  }
  #preview-img.mode-fit {
    cursor: zoom-in;
  }
  #preview-img.mode-roi {
    cursor: grab;
  }
  #preview-img.mode-roi.grabbing {
    cursor: grabbing !important;
  }

  #retouch-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 4px;
    pointer-events: none;
    display: block;
    z-index: 10;
  }
  #viewport.brush-active {
    cursor: none;
  }
  #retouch-canvas.active {
    pointer-events: auto;
    cursor: none;
  }

  #brush-cursor {
    position: fixed;
    pointer-events: none;
    border: 1.5px solid #38bdf8;
    background: rgba(56, 189, 248, 0.18);
    border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.6), 0 0 8px rgba(0, 0, 0, 0.8);
    display: none;
    z-index: 1000;
    transform: translate(-50%, -50%);
    transition: border-color 0.15s ease, background-color 0.15s ease;
  }
  #brush-cursor.mode-erase {
    border-color: #f87171;
    background: rgba(248, 113, 113, 0.22);
  }
  #brush-cursor-inner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border: 1px dashed rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    pointer-events: none;
    box-sizing: border-box;
    display: none;
  }

  #pan-hint {
    position: absolute;
    top: 14px;
    right: 16px;
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 20px;
    font-size: 11px;
    color: #e4e4e7;
    pointer-events: none;
    display: none;
    z-index: 20;
  }

  #loading-badge {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(18, 18, 20, 0.85);
    backdrop-filter: blur(8px);
    color: #fff;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
    border: 1px solid var(--border);
    box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    display: none;
    z-index: 50;
  }
  #viewport-loader {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    pointer-events: none;
    z-index: 10;
    transition: opacity 0.25s ease;
  }
  .loader-spinner {
    width: 44px;
    height: 44px;
    border: 3.5px solid rgba(255, 255, 255, 0.1);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.85s cubic-bezier(0.5, 0, 0.5, 1) infinite;
  }
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  .loader-text {
    font-size: 13px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: 0.3px;
  }
  .loader-subtext {
    font-size: 11px;
    color: var(--muted);
  }

  .segmented-control {
    display: flex;
    background: #18181b;
    padding: 3px;
    border-radius: 8px;
    border: 1px solid var(--border);
    gap: 3px;
  }
  .seg-btn {
    flex: 1;
    padding: 6px 10px;
    border-radius: 6px;
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
  }
  .seg-btn:hover {
    color: var(--text);
    background: rgba(255, 255, 255, 0.05);
  }
  .seg-btn.active {
    background: #27272a;
    color: #ffffff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
  }
  .zoom-segmented .seg-btn, .bg-mode-segmented .seg-btn {
    padding: 6px 4px;
    font-size: 11px;
    gap: 2px;
  }

  #toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    background: #16a34a;
    color: #ffffff;
    padding: 14px 22px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 600;
    box-shadow: 0 10px 30px rgba(0,0,0,0.6);
    display: none;
    z-index: 1000;
    animation: fadeIn 0.2s ease-out;
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
  }

  #render-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(10, 10, 12, 0.88);
    backdrop-filter: blur(12px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: fadeIn 0.2s ease-out;
  }
  .render-modal-card {
    background: #18181b;
    border: 1px solid var(--border);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.7);
    border-radius: 14px;
    padding: 36px 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    max-width: 480px;
    width: 90%;
    gap: 14px;
  }
  .render-spinner-ring {
    width: 52px;
    height: 52px;
    border: 4px solid rgba(37, 99, 235, 0.2);
    border-top-color: #3b82f6;
    border-radius: 50%;
    animation: spin 0.8s cubic-bezier(0.5, 0, 0.5, 1) infinite;
    margin-bottom: 4px;
    box-sizing: border-box;
  }
  #render-title {
    font-size: 17px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.3px;
  }
  #render-desc {
    font-size: 13px;
    color: #a1a1aa;
    line-height: 1.5;
  }
  #render-timer {
    font-family: monospace;
    font-size: 12px;
    color: #60a5fa;
    background: rgba(37, 99, 235, 0.15);
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 600;
  }
  .render-bar-container {
    width: 100%;
    height: 6px;
    background: #27272a;
    border-radius: 3px;
    overflow: hidden;
    position: relative;
    margin: 4px 0;
  }
  .render-bar-fill {
    width: 0%;
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    border-radius: 3px;
    transition: width 0.25s ease;
  }
  .render-bar-indeterminate {
    width: 40%;
    height: 100%;
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    border-radius: 3px;
    position: absolute;
    animation: indeterminateBar 1.6s ease-in-out infinite;
  }
  @keyframes indeterminateBar {
    0% { left: -40%; }
    50% { left: 40%; }
    100% { left: 100%; }
  }
  #render-subnote {
    font-size: 11px;
    color: #71717a;
    line-height: 1.4;
  }

  .btn-dev-toggle {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border);
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 6px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 5px;
    transition: all 0.15s ease;
  }
  .btn-dev-toggle:hover {
    background: rgba(255, 255, 255, 0.1);
    color: var(--text);
  }
  .btn-dev-toggle.active {
    background: rgba(168, 85, 247, 0.18);
    border-color: #a855f7;
    color: #d8b4fe;
    box-shadow: 0 0 10px rgba(168, 85, 247, 0.25);
  }

  /* Shortcuts Cheatsheet Modal */
  #help-modal {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(10, 10, 12, 0.85);
    backdrop-filter: blur(12px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 9998;
    animation: fadeIn 0.15s ease-out;
  }
  .help-modal-card {
    background: #18181c;
    border: 1px solid var(--border);
    box-shadow: 0 25px 60px rgba(0, 0, 0, 0.75);
    border-radius: 14px;
    padding: 26px 30px;
    max-width: 560px;
    width: 90%;
    color: var(--text);
    display: flex;
    flex-direction: column;
    gap: 16px;
    position: relative;
  }
  .help-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
  }
  .help-title {
    font-size: 16px;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .help-close-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    font-size: 18px;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 6px;
    line-height: 1;
    transition: all 0.15s ease;
  }
  .help-close-btn:hover {
    color: #ffffff;
    background: #27272a;
  }
  .help-sections {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }
  .help-category {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }
  .help-category-title {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #60a5fa;
    margin-bottom: 2px;
  }
  .help-shortcut-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    gap: 8px;
  }
  .help-desc {
    color: #a1a1aa;
    font-size: 12px;
  }
  kbd {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 4px;
    box-shadow: 0 1px 0 rgba(0, 0, 0, 0.4);
    color: #f4f4f5;
    display: inline-block;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
    font-size: 11px;
    font-weight: 600;
    line-height: 1;
    padding: 3px 6px;
    white-space: nowrap;
  }
  .btn-help-trigger {
    background: #202024;
    border: 1px solid var(--border);
    color: #a1a1aa;
    padding: 5px 11px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
    display: flex;
    align-items: center;
    gap: 5px;
  }
  .btn-help-trigger:hover {
    background: #27272e;
    color: #ffffff;
    border-color: #3f3f46;
  }

  /* Initial Loading Screen */
  #init-loader {
    position: fixed;
    inset: 0;
    background: #101012;
    z-index: 10000;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: opacity 0.35s ease, visibility 0.35s ease;
  }
  #init-loader.hidden {
    opacity: 0;
    visibility: hidden;
    pointer-events: none;
    display: none !important;
  }
  .init-card {
    width: 440px;
    background: #18181c;
    border: 1px solid #282830;
    border-radius: 14px;
    padding: 36px 32px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    box-shadow: 0 20px 45px rgba(0,0,0,0.6);
  }
  .init-logo-ring {
    position: relative;
    width: 72px;
    height: 72px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 18px;
  }
  .init-logo-icon {
    font-size: 32px;
  }
  .init-spinner {
    position: absolute;
    inset: 0;
    border: 3px solid rgba(255, 255, 255, 0.08);
    border-top-color: #0070f3;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }
  .init-title {
    font-size: 19px;
    font-weight: 600;
    color: #fff;
    letter-spacing: -0.3px;
    margin-bottom: 4px;
  }
  .init-sub {
    font-size: 12px;
    color: #888892;
    margin-bottom: 24px;
  }
  .init-progress-track {
    width: 100%;
    height: 6px;
    background: #24242a;
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  .init-progress-fill {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #0070f3, #38bdf8);
    border-radius: 3px;
    transition: width 0.2s ease;
  }
  .init-status-row {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    font-weight: 500;
    color: #d1d1d6;
    margin-bottom: 6px;
  }
  #init-detail {
    width: 100%;
    text-align: left;
    font-size: 11px;
    color: #71717a;
    margin-bottom: 24px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .btn-cancel-init {
    background: #24242a;
    border: 1px solid #33333c;
    color: #a1a1aa;
    padding: 8px 22px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-cancel-init:hover {
    background: #2e2e36;
    color: #fff;
  }

  /* Drag and Drop Landing Screen */
  #drop-zone {
    position: fixed;
    inset: 0;
    background: radial-gradient(circle at center, #18181f 0%, #0d0d0f 100%);
    z-index: 10001;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 30px;
    box-sizing: border-box;
    transition: opacity 0.3s ease, visibility 0.3s ease;
  }
  #drop-zone.hidden {
    display: none !important;
  }
  .drop-card {
    width: 620px;
    max-width: 92vw;
    background: #18181c;
    border: 2px dashed #353540;
    border-radius: 18px;
    padding: 46px 38px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    box-shadow: 0 25px 65px rgba(0, 0, 0, 0.75);
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    cursor: pointer;
  }
  .drop-card:hover {
    border-color: #3b82f6;
    background: #1b1b22;
    box-shadow: 0 30px 75px rgba(0, 0, 0, 0.85), 0 0 35px rgba(59, 130, 246, 0.15);
  }
  .drop-card.drag-active {
    border-color: #60a5fa;
    border-style: solid;
    background: rgba(37, 99, 235, 0.09);
    box-shadow: 0 30px 80px rgba(0, 0, 0, 0.9), 0 0 50px rgba(59, 130, 246, 0.4);
    transform: scale(1.02);
  }
  .drop-icon-ring {
    width: 80px;
    height: 80px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: rgba(56, 189, 248, 0.08);
    border-radius: 50%;
    border: 1px solid rgba(56, 189, 248, 0.22);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
    transition: transform 0.25s ease;
  }
  .drop-card:hover .drop-icon-ring {
    transform: scale(1.06);
    background: rgba(56, 189, 248, 0.14);
    border-color: #38bdf8;
  }
  .drop-svg {
    width: 48px;
    height: 48px;
  }
  .drop-title {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.4px;
    margin-bottom: 4px;
  }
  .drop-tagline {
    font-size: 11px;
    font-weight: 600;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 18px;
  }
  .drop-hero-text {
    font-size: 16px;
    font-weight: 600;
    color: #e4e4e7;
    margin-bottom: 6px;
  }
  .drop-hint {
    font-size: 12px;
    color: #888892;
    margin-bottom: 28px;
  }
  .drop-actions {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
  }
  .btn-drop-action {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 600;
    padding: 11px 22px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s ease;
    border: none;
  }
  .btn-drop-primary {
    background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.15);
    box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
  }
  .btn-drop-primary:hover {
    background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(37, 99, 235, 0.45);
  }
  .btn-drop-secondary {
    background: #24242c;
    color: #e4e4e7;
    border: 1px solid #383844;
  }
  .btn-drop-secondary:hover {
    background: #2e2e38;
    color: #ffffff;
    border-color: #4a4a58;
    transform: translateY(-1px);
  }
  .drop-btn-icon {
    font-size: 15px;
  }
  .drop-formats-row {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 6px;
  }
  .drop-chip {
    font-size: 11px;
    font-weight: 500;
    color: #888892;
    background: #141418;
    border: 1px solid #282832;
    padding: 4px 10px;
    border-radius: 12px;
  }
  .drop-error-banner {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.4);
    color: #fca5a5;
    font-size: 12px;
    padding: 10px 18px;
    border-radius: 8px;
    margin-bottom: 20px;
    width: 100%;
    box-sizing: border-box;
    text-align: center;
  }
</style>
</head>
<body>

<div id="drop-zone">
  <div class="drop-card" onclick="onDropCardClick(event)">
    <div class="drop-icon-ring">
      <svg class="drop-svg" viewBox="0 0 64 64" fill="none">
        <path d="M14 20 L32 10 L50 20 L32 30 Z" stroke="#38bdf8" stroke-width="2.5" stroke-linejoin="round" fill="rgba(56, 189, 248, 0.16)"/>
        <path d="M14 32 L32 42 L50 32" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M14 44 L32 54 L50 44" stroke="#818cf8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <h2 class="drop-title">Open Focus Stack Pro</h2>
    <div class="drop-tagline">16-Bit Multi-Channel Fusion Engine</div>
    <div class="drop-hero-text">Drag &amp; Drop Bracketed Photos or Folder Here</div>
    <div class="drop-hint">Drop an entire photo folder or multiple bracketed focus frames to stack</div>

    <div class="drop-actions" onclick="event.stopPropagation()">
      <button type="button" class="btn-drop-action btn-drop-primary" onclick="browseFolder(event)">
        <span class="drop-btn-icon">📁</span> Browse Folder...
      </button>
      <button type="button" class="btn-drop-action btn-drop-secondary" onclick="browseFiles(event)">
        <span class="drop-btn-icon">🖼️</span> Select Photos...
      </button>
    </div>

    <div id="drop-error" class="drop-error-banner" style="display: none;"></div>

    <div class="drop-formats-row">
      <span class="drop-chip">16-Bit TIFF</span>
      <span class="drop-chip">JPEG</span>
      <span class="drop-chip">Sony ARW</span>
      <span class="drop-chip">Canon CR2 / CR3</span>
      <span class="drop-chip">Nikon NEF</span>
      <span class="drop-chip">DNG</span>
    </div>
  </div>
</div>

<div id="init-loader">
  <div class="init-card">
    <div class="init-logo-ring">
      <span class="init-logo-icon">📷</span>
      <div class="init-spinner"></div>
    </div>
    <h2 class="init-title">Open Focus Stack Pro</h2>
    <div class="init-sub">16-Bit Multi-Channel Fusion Engine</div>
    <div class="init-progress-track">
      <div id="init-progress-bar" class="init-progress-fill"></div>
    </div>
    <div class="init-status-row">
      <span id="init-stage">Initializing stacking engine...</span>
      <span id="init-pct">0%</span>
    </div>
    <div id="init-detail">Preparing bracketed frames...</div>
    <button type="button" class="btn-cancel-init" onclick="cancelStacking()">✕ Cancel Stacking</button>
  </div>
</div>

<div id="brush-cursor"><div id="brush-cursor-inner"></div></div>

<div id="sidebar">
  <h1>Open Focus Stack <span class="badge">EG-MSD 16-Bit Pro</span></h1>

  <!-- Utility Actions (Dev Mode & Shortcuts) -->
  <div style="display:flex; gap:8px;">
    <button type="button" id="btn-dev-mode" class="btn-dev-toggle" style="flex:1; justify-content:center; padding:6px 10px;" onclick="toggleDevMode()" title="Toggle Developer Mode / Diagnostic Signals">🛠️ Dev Mode</button>
    <button type="button" id="btn-help" class="btn-help-trigger" style="flex:1; justify-content:center; padding:6px 10px;" onclick="toggleHelpModal()" title="Keyboard Shortcuts (?)">⌨️ Shortcuts (?)</button>
  </div>

  <!-- Zoom Level (Top / Most Used) -->
  <div class="control-group">
    <label>Zoom Level</label>
    <div class="segmented-control zoom-segmented">
      <button type="button" id="btn-mode-fit" class="seg-btn active" onclick="setViewMode('fit')">Fit</button>
      <button type="button" id="btn-mode-50" class="seg-btn" onclick="setViewMode('50%')">50%</button>
      <button type="button" id="btn-mode-roi" class="seg-btn" onclick="setViewMode('100%')">100%</button>
      <button type="button" id="btn-mode-200" class="seg-btn" onclick="setViewMode('200%')">200%</button>
    </div>
  </div>

  <div id="navigator-container">
    <label id="navigator-label">Loupe Navigator <span class="val" id="val-pos">Fit (Click to 100%)</span></label>
    <div id="minimap-wrapper" onclick="onMinimapClick(event)">
      <img id="minimap-img" alt="Overview">
      <div id="minimap-box"></div>
    </div>
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <button class="btn-center-crop" onclick="resetCropCenter()">🎯 Reset to Center</button>
      <span class="desc" style="font-size:10px;">Drag box or image to pan</span>
    </div>
  </div>

  <!-- Source Frame Inspector & Blink (Most Used) -->
  <div class="control-group" id="grp-source-frame">
    <div style="display:flex; justify-content:space-between; align-items:center;">
      <label style="margin-bottom:0;">Source Frame Inspector</label>
      <span id="frame-indicator" class="val" style="font-size:11px;">Frame 1</span>
    </div>
    <div style="display:flex; gap:6px; align-items:center; margin-top:4px;">
      <button type="button" class="btn-step" onclick="stepFrame(-1)" title="Previous Frame (Left Arrow / ,)">◀</button>
      <input type="range" id="retouch-frame" min="1" max="1" value="1" oninput="onFrameSliderChange()" onmousedown="onSliderStartBlink()" onmouseup="onSliderStopBlink()" ontouchstart="onSliderStartBlink()" ontouchend="onSliderStopBlink()" style="flex:1;">
      <button type="button" class="btn-step" onclick="stepFrame(1)" title="Next Frame (Right Arrow / .)">▶</button>
    </div>
    <div id="frame-filename" style="font-size:10px; color:var(--muted); text-align:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; margin-top:3px;">DSC001.tif</div>
    <div style="display:flex; gap:6px; margin-top:6px;">
      <button type="button" id="btn-blink" class="btn-blink" style="flex:1;" onmousedown="startBlink()" onmouseup="stopBlink()" onmouseleave="stopBlink()" title="Hold \ to Blink Source Frame">👁️ Blink (\)</button>
      <button type="button" id="btn-peaking" class="btn-blink" style="flex:1;" onmousedown="startPeaking()" onmouseup="stopPeaking()" onmouseleave="stopPeaking()" title="Hold P to Blink Focus Peaking of Active Frame">🎯 Peaking (P)</button>
    </div>
  </div>

  <!-- Manual Retouching & Override Section (Most Used) -->
  <div class="retouch-box">
    <div class="retouch-header">
      <label style="color:#60a5fa; cursor:pointer;" onclick="toggleBrush()">🖌️ Retouch &amp; Override</label>
      <button type="button" id="btn-brush-toggle" class="btn-tool" onclick="toggleBrush()">Brush Tool (B)</button>
    </div>
    <div id="retouch-panel" style="display: none; flex-direction: column; gap: 8px;">
      <!-- Paint vs Erase Mode Switch -->
      <div class="segmented-control" style="margin-bottom: 2px;">
        <button type="button" id="btn-brush-paint" class="seg-btn active" onclick="setBrushToolMode('paint')">🖌️ Paint (B)</button>
        <button type="button" id="btn-brush-erase" class="seg-btn" onclick="setBrushToolMode('erase')">🧹 Erase (E)</button>
      </div>

      <div style="display:flex; justify-content:space-between; font-size:11px;">
        <span style="color:var(--muted);">Active Frame:</span>
        <span id="brush-frame-indicator" class="val" style="color:#60a5fa; font-weight:600;">Frame 1</span>
      </div>
      <div style="display:flex; justify-content:space-between; font-size:11px;">
        <span>Brush Size:</span>
        <span id="val-brush-size" class="val">50px</span>
      </div>
      <input type="range" id="brush-size" min="10" max="250" value="50" oninput="updateBrushSize()">

      <div style="display:flex; justify-content:space-between; font-size:11px;">
        <span>Brush Feather:</span>
        <span id="val-brush-feather" class="val">25%</span>
      </div>
      <input type="range" id="brush-feather" min="0" max="100" value="25" oninput="updateBrushFeather()">

      <div style="display:flex; justify-content:space-between; font-size:11px;">
        <span>Brush Opacity:</span>
        <span id="val-brush-opacity" class="val">100%</span>
      </div>
      <input type="range" id="brush-opacity" min="10" max="100" value="100" oninput="updateBrushOpacity()">

      <div style="display:flex; gap:8px; margin-top:4px;">
        <button type="button" class="btn-action-small" onclick="undoStroke()">↩️ Undo</button>
        <button type="button" class="btn-action-small" onclick="clearRetouch()">🗑️ Clear</button>
      </div>
    </div>
  </div>

  <!-- Collapsible Core Algorithm & Fusion Tuning Parameters -->
  <details class="algo-collapsible" id="grp-algo-collapsible">
    <summary class="algo-summary">
      <div class="summary-left">
        <span class="summary-chevron">▶</span>
        <span class="summary-title">⚙️ Algorithm &amp; Fusion Tuning</span>
      </div>
      <span class="summary-badge">EG-MSD</span>
    </summary>
    <div class="algo-collapsible-content">
      <div class="control-group" id="grp-bg-mode">
        <label>Background Mode <span class="val" id="val-bg_mode">Auto</span></label>
        <div class="segmented-control bg-mode-segmented">
          <button type="button" id="btn-bg-auto" class="seg-btn active" onclick="setBgMode('auto')">Auto</button>
          <button type="button" id="btn-bg-dark" class="seg-btn" onclick="setBgMode('dark')">Dark</button>
          <button type="button" id="btn-bg-bright" class="seg-btn" onclick="setBgMode('bright')">Bright</button>
          <button type="button" id="btn-bg-energy" class="seg-btn" onclick="setBgMode('energy')">Energy</button>
        </div>
        <div class="desc">Polarity gating for halo suppression: Dark (low-key), Bright (high-key), or Energy (mid-tone).</div>
      </div>

      <div class="control-group" id="grp-stack-order">
        <label>Stack Direction <span class="val" id="val-stack_order">Near ➔ Far</span></label>
        <div class="segmented-control">
          <button type="button" id="btn-order-near" class="seg-btn active" onclick="setStackOrder('near_to_far')">Near ➔ Far</button>
          <button type="button" id="btn-order-far" class="seg-btn" onclick="setStackOrder('far_to_near')">Far ➔ Near</button>
        </div>
        <div class="desc">Slice capture order. Near ➔ Far is standard for rails; Far ➔ Near for reverse in-camera bracketing sweeps.</div>
      </div>


      <div class="control-group" id="grp-halo-gamma">
        <label>Halo Suppression (γ) <span class="val" id="val-halo_gamma">0.45</span></label>
        <input type="range" id="halo_gamma" min="0.10" max="1.50" step="0.05" value="0.45" oninput="onSliderInput('halo_gamma')">
        <div class="desc">Exponential penalty on out-of-focus background light cones behind steep boundaries.</div>
      </div>

      <div class="control-group" id="grp-z-jump">
        <label>Z-Jump Sensitivity <span class="val" id="val-z_jump">5.0</span></label>
        <input type="range" id="z_jump" min="1.0" max="15.0" step="0.5" value="5.0" oninput="onSliderInput('z_jump')">
        <div class="desc">Depth jump threshold (ΔZ) triggering asymmetric occlusion gating around foreground edges.</div>
      </div>

      <div class="control-group" id="grp-anisotropy">
        <label>Bristle Anisotropy Boost <span class="val" id="val-anisotropy">0.50</span></label>
        <input type="range" id="anisotropy" min="0.0" max="1.50" step="0.05" value="0.50" oninput="onSliderInput('anisotropy')">
        <div class="desc">Eigenvalue coherence boost prioritizing directional thin structures (insect hairs, spider silk).</div>
      </div>

      <div class="control-group" id="grp-threshold">
        <label>Bokeh Noise Multiplier <span class="val" id="val-threshold">1.0x</span></label>
        <input type="range" id="threshold" min="0.2" max="3.0" step="0.1" value="1.0" oninput="onSliderInput('threshold')">
        <div class="desc">Multiplier over Poisson-Gaussian sensor noise floor to keep smooth bokeh noise-free.</div>
      </div>

      <div class="control-group" id="grp-specular">
        <label>Specular Bokeh Gate <span class="val" id="val-specular_thresh">0.94</span></label>
        <input type="range" id="specular_thresh" min="0.80" max="1.00" step="0.01" value="0.94" oninput="onSliderInput('specular_thresh')">
        <div class="desc">Luminance threshold suppressing false edge gradients from specular blur rings.</div>
      </div>
    </div>
  </details>

  <div class="btn-row">
    <button class="btn-render" id="btn-save" onclick="triggerFinalRender()">⚡ Render &amp; Import to Lightroom</button>
    <button class="btn-cancel" onclick="cancelStacking()">✕ Cancel Stacking</button>
  </div>
</div>

<div id="main">
  <div id="toolbar">
    <div style="display:flex; align-items:center; gap:12px;">
      <span id="img-info">Aligning frames...</span>
    </div>
    <div style="display:flex; align-items:center; gap:12px;">
      <div class="segmented-control" id="view-mode-selector">
        <button type="button" id="btn-view-image" class="seg-btn active" onclick="setVisualMode('image')">📷 Stacked Image</button>
        <button type="button" id="btn-view-depth" class="seg-btn dev-signal" style="display:none;" onclick="setVisualMode('depth')">🗺️ Depth Map (D)</button>
        <button type="button" id="btn-view-sal" class="seg-btn dev-signal" style="display:none;" onclick="setVisualMode('saliency')">⚡ Focus Saliency (S)</button>
      </div>
      <span id="live-tag"><span class="pulse-dot"></span> Live</span>
    </div>
  </div>
  <div id="viewport">
    <div id="viewport-loader">
      <div class="loader-spinner"></div>
      <div class="loader-text">Loading 61MP Preview Frames...</div>
      <div class="loader-subtext">Aligning and pre-rendering live multi-scale preview</div>
    </div>
    <div id="canvas-container" onmousedown="startPanDrag(event)">
      <img id="pan-overview-img" alt="Overview">
      <img id="preview-img" alt="Preview">
      <canvas id="retouch-canvas"></canvas>
    </div>
    <div id="pan-hint">🖐 Drag anywhere on the image to pan 100% crop</div>
    <div id="loading-badge">Computing live stack...</div>
  </div>
</div>

<div id="toast"></div>

<div id="render-modal">
  <div class="render-modal-card">
    <div class="render-spinner-ring"></div>
    <div id="render-title">⚡ Computing 16-Bit Focus Stack</div>
    <div id="render-desc">Streaming frames through multi-scale Laplacian pyramids...</div>
    <div id="render-timer">Elapsed: 0s</div>
    <div class="render-bar-container">
      <div id="render-bar" class="render-bar-fill"></div>
    </div>
    <div id="render-subnote">Processing 61MP TIFFs with float32 precision. This window will automatically close when the image imports into Lightroom.</div>
  </div>
</div>

<div id="help-modal" onclick="onHelpModalOverlayClick(event)">
  <div class="help-modal-card" onclick="event.stopPropagation()">
    <div class="help-header">
      <div class="help-title">⌨️ Keyboard Shortcuts Cheatsheet</div>
      <button type="button" class="help-close-btn" onclick="closeHelpModal()">✕</button>
    </div>
    <div class="help-sections">
      <div class="help-category">
        <div class="help-category-title">Navigation & Viewing</div>
        <div class="help-shortcut-row">
          <span class="help-desc">Toggle Fit / 100% Loupe</span>
          <kbd>Z</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Zoom In / Zoom Out</span>
          <span><kbd>+</kbd> <kbd>-</kbd></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Direct Zoom (Fit/50/100/200)</span>
          <span><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Toggle Depth Map</span>
          <kbd>D</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Toggle Focus Saliency</span>
          <kbd>S</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Jump Loupe to Point</span>
          <kbd>Click</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Pan Loupe View</span>
          <kbd>Drag</kbd>
        </div>
      </div>

      <div class="help-category">
        <div class="help-category-title">Retouch & Overrides</div>
        <div class="help-shortcut-row">
          <span class="help-desc">Paint Brush Tool</span>
          <kbd>B</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Eraser Brush Tool</span>
          <kbd>E</kbd>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Resize Brush (over image)</span>
          <span><kbd>Scroll</kbd> <em style="font-size:10px;color:var(--muted);">(or <kbd>[</kbd> <kbd>]</kbd>)</em></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Adjust Feather</span>
          <span><kbd>Shift+[</kbd> <kbd>Shift+]</kbd></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Blink Source Frame</span>
          <span><kbd>\</kbd> <em style="font-size:10px;color:var(--muted);">(hold)</em></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Blink Focus Peaking</span>
          <span><kbd>P</kbd> <em style="font-size:10px;color:var(--muted);">(hold)</em></span>
        </div>
        <div class="help-shortcut-row">
          <span class="help-desc">Step Source Frame</span>
          <span><kbd>←</kbd> <kbd>→</kbd> <em style="font-size:10px;color:var(--muted);">(or <kbd>,</kbd> <kbd>.</kbd>)</em></span>
        </div>
      </div>
    </div>

    <div style="display:flex; justify-content:space-between; align-items:center; border-top:1px solid var(--border); padding-top:12px; margin-top:2px;">
      <span style="font-size:11px; color:var(--muted);">Press <kbd>?</kbd> anytime to toggle cheatsheet • Press <kbd>Esc</kbd> to dismiss</span>
      <button type="button" class="btn-cancel-init" style="padding:4px 14px; font-size:11px;" onclick="closeHelpModal()">Close</button>
    </div>
  </div>
</div>

<script>
const ZOOM_LEVELS = ['fit', '50%', '100%', '200%'];
let isStandaloneMode = true;
let isDevMode = false;
let currentStackOrder = 'near_to_far';
let currentViewMode = 'fit';
let currentVisualMode = 'image'; // 'image' or 'depth'
let currentCx = 0.5;
let currentCy = 0.5;
let currentCropRect = [0.45, 0.45, 0.1, 0.1];
let previewDebounceTimer = null;
let currentAbortController = null;
let overviewImageSrc = "";
let overviewDepthSrc = "";
let overviewSaliencySrc = "";
let currentImageSrc = "";
let currentDepthSrc = "";
let currentSaliencySrc = "";
let isDraggingImage = false;
let isSpacePressed = false;
let dragStartX = 0;
let dragStartY = 0;
let isDraggingMinimap = false;

// Retouch Brush State
let isBrushActive = false;
let isPainting = false;
let numFrames = 1;
let frameNames = [];
let currentFrameIdx = 0; // 0-indexed
let brushRadius = 50;
let brushFeather = 25; // 0% (hard) to 100% (fully soft)
let brushOpacity = 1.0;
let strokeHistory = [];
let masterMaskCanvas = document.createElement('canvas');
let masterMaskCtx = masterMaskCanvas.getContext('2d');
let masterMaskInitialized = false;
let cachedFrameImages = {};
let inFlightFrameFetches = {};
let frameMaskCanvases = {};
let activeOverrideFrames = new Set();
let hasRetouchOverrides = false;
let lastMouseX = null;
let lastMouseY = null;
let isBlinking = false;
let isPeaking = false;
let cachedPeakingImages = {};
let currentAlgorithm = 'egmsd';
let brushToolMode = 'paint'; // 'paint' or 'erase'
let isSliderBlinking = false;
let brushWheelAccumulator = 0;
const BRUSH_SCROLL_MULTIPLIER = 0.15;

let currentBgMode = 'auto';

function setStackOrder(order) {
  if (currentStackOrder === order) return;
  currentStackOrder = order;
  const btnNear = document.getElementById('btn-order-near');
  const btnFar = document.getElementById('btn-order-far');
  const valEl = document.getElementById('val-stack_order');
  if (btnNear) btnNear.classList.toggle('active', order === 'near_to_far');
  if (btnFar) btnFar.classList.toggle('active', order === 'far_to_near');
  if (valEl) valEl.innerText = order === 'near_to_far' ? 'Near ➔ Far' : 'Far ➔ Near';
  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(fetchPreview, 50);
}

function toggleDevMode() {
  setDevMode(!isDevMode);
}

function setDevMode(active) {
  isDevMode = !!active;
  const btnDev = document.getElementById('btn-dev-mode');
  const devSignals = document.querySelectorAll('.dev-signal');
  if (btnDev) btnDev.classList.toggle('active', isDevMode);
  devSignals.forEach(el => {
    el.style.display = isDevMode ? '' : 'none';
  });
  if (!isDevMode && currentVisualMode !== 'image') {
    setVisualMode('image');
  }
  showToast(isDevMode ? "🛠️ Dev Mode: Intermediate signals enabled" : "Dev Mode disabled", 1500);
}

function setBgMode(mode) {
  currentBgMode = mode;
  ['auto', 'dark', 'bright', 'energy'].forEach(m => {
    const btn = document.getElementById('btn-bg-' + m);
    if (btn) btn.classList.toggle('active', m === mode);
  });
  const valEl = document.getElementById('val-bg_mode');
  if (valEl) {
    valEl.innerText = mode.charAt(0).toUpperCase() + mode.slice(1);
  }
  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(fetchPreview, 50);
}

function updateVal(id) {
  const el = document.getElementById(id);
  const valEl = document.getElementById('val-' + id);
  if (!el || !valEl) return;
  if (id === 'threshold') {
    valEl.innerText = parseFloat(el.value).toFixed(1) + 'x';
  } else if (id === 'halo_gamma' || id === 'anisotropy' || id === 'specular_thresh') {
    valEl.innerText = parseFloat(el.value).toFixed(2);
  } else if (id === 'z_jump') {
    valEl.innerText = parseFloat(el.value).toFixed(1);
  } else {
    valEl.innerText = el.value;
  }
}

function onSliderInput(id) {
  updateVal(id);
  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(() => {
    fetchPreview();
  }, 180);
}

function syncPanOverview() {
  const pImg = document.getElementById('preview-img');
  const oImg = document.getElementById('pan-overview-img');
  if (!pImg || !oImg || currentViewMode === 'fit' || !overviewImageSrc) {
    if (oImg) oImg.style.display = 'none';
    return;
  }
  let targetSrc = overviewImageSrc;
  if (currentVisualMode === 'depth' && overviewDepthSrc) targetSrc = overviewDepthSrc;
  else if (currentVisualMode === 'saliency' && overviewSaliencySrc) targetSrc = overviewSaliencySrc;

  if (oImg.src !== targetSrc) oImg.src = targetSrc;
  oImg.style.display = 'block';

  const dispW = pImg.offsetWidth || pImg.naturalWidth;
  const dispH = pImg.offsetHeight || pImg.naturalHeight;
  const [rx, ry, rw, rh] = currentCropRect;
  if (rw > 0 && rh > 0 && dispW > 0 && dispH > 0) {
    const fullW = dispW / rw;
    const fullH = dispH / rh;
    const originX = -rx * fullW;
    const originY = -ry * fullH;
    oImg.style.width = Math.round(fullW) + 'px';
    oImg.style.height = Math.round(fullH) + 'px';
    oImg.style.left = Math.round(originX) + 'px';
    oImg.style.top = Math.round(originY) + 'px';
    oImg.style.transform = 'translate(0px, 0px)';
  }
}

function setViewMode(mode) {
  const normMode = (mode === 'roi') ? '100%' : mode;
  if (currentViewMode === normMode) return;
  currentViewMode = normMode;

  const btnFit = document.getElementById('btn-mode-fit');
  const btn50 = document.getElementById('btn-mode-50');
  const btnRoi = document.getElementById('btn-mode-roi');
  const btn200 = document.getElementById('btn-mode-200');

  if (btnFit) btnFit.classList.toggle('active', normMode === 'fit');
  if (btn50) btn50.classList.toggle('active', normMode === '50%');
  if (btnRoi) btnRoi.classList.toggle('active', normMode === '100%');
  if (btn200) btn200.classList.toggle('active', normMode === '200%');

  const nav = document.getElementById('navigator-container');
  const panHint = document.getElementById('pan-hint');
  const img = document.getElementById('preview-img');
  const cCont = document.getElementById('canvas-container');
  const oImg = document.getElementById('pan-overview-img');
  const navLabel = document.getElementById('navigator-label');

  if (nav) nav.style.display = 'flex';

  if (normMode !== 'fit') {
    if (panHint) panHint.style.display = 'block';
    if (img) img.className = 'loaded mode-roi';
    if (cCont) cCont.className = 'mode-roi';
    if (navLabel) {
      const t = normMode === '50%' ? '50% Zoom' : (normMode === '200%' ? '200% Loupe' : '100% Loupe');
      navLabel.innerHTML = `${t} Navigator <span class="val" id="val-pos">${Math.round(currentCx * 100)}%, ${Math.round(currentCy * 100)}%</span>`;
    }
    updateMinimapBox();
    syncPanOverview();
  } else {
    if (panHint) panHint.style.display = 'none';
    if (img) img.className = 'loaded mode-fit';
    if (cCont) cCont.className = 'mode-fit';
    if (oImg) oImg.style.display = 'none';
    if (navLabel) {
      navLabel.innerHTML = `Overview Navigator <span class="val" id="val-pos">Fit (Click to 100%)</span>`;
    }
    updateMinimapBox();
  }
  fetchPreview();
}

function zoomIn() {
  let idx = ZOOM_LEVELS.indexOf(currentViewMode === 'roi' ? '100%' : currentViewMode);
  if (idx === -1) idx = 0;
  if (idx < ZOOM_LEVELS.length - 1) {
    setViewMode(ZOOM_LEVELS[idx + 1]);
  }
}

function zoomOut() {
  let idx = ZOOM_LEVELS.indexOf(currentViewMode === 'roi' ? '100%' : currentViewMode);
  if (idx === -1) idx = 0;
  if (idx > 0) {
    setViewMode(ZOOM_LEVELS[idx - 1]);
  }
}

function setVisualMode(vmode) {
  if (currentVisualMode === vmode) return;
  currentVisualMode = vmode;
  document.getElementById('btn-view-image').classList.toggle('active', vmode === 'image');
  document.getElementById('btn-view-depth').classList.toggle('active', vmode === 'depth');
  const btnSal = document.getElementById('btn-view-sal');
  if (btnSal) btnSal.classList.toggle('active', vmode === 'saliency');
  updateViewportSource();
}

function toggleVisualMode() {
  const modes = ['image', 'depth', 'saliency'];
  let idx = modes.indexOf(currentVisualMode);
  setVisualMode(modes[(idx + 1) % modes.length]);
}

function updateViewportSource() {
  if (isBlinking || isPeaking) return;
  const img = document.getElementById('preview-img');
  let target = currentImageSrc;
  if (currentVisualMode === 'depth') target = currentDepthSrc;
  else if (currentVisualMode === 'saliency') target = currentSaliencySrc;
  if (target && img.src !== target) {
    img.src = target;
  }
  syncPanOverview();
  redrawRetouchCanvas();
}

function resetCropCenter() {
  currentCx = 0.5;
  currentCy = 0.5;
  updateMinimapBox();
  fetchPreview();
}

function showToast(msg, duration) {
  const toast = document.getElementById('toast');
  if (toast) {
    toast.innerText = msg;
    toast.style.display = 'block';
    setTimeout(() => { toast.style.display = 'none'; }, duration || 3000);
  }
}

function showLoadingBadge(show) {
  const badge = document.getElementById('loading-badge');
  if (badge) badge.style.display = show ? 'block' : 'none';
  const img = document.getElementById('preview-img');
  const oImg = document.getElementById('pan-overview-img');
  if (show) {
    if (img && img.classList.contains('loaded')) {
      img.style.filter = 'brightness(0.50) contrast(0.95)';
      img.style.opacity = '0.70';
    }
    if (oImg) {
      oImg.style.filter = 'brightness(0.50) contrast(0.95)';
      oImg.style.opacity = '0.70';
    }
  } else {
    if (img) {
      img.style.filter = '';
      img.style.opacity = '1';
    }
    if (oImg) {
      oImg.style.filter = '';
      oImg.style.opacity = '1';
    }
  }
}

function updateMinimapBox(overrideCx, overrideCy) {
  const wrapper = document.getElementById('minimap-wrapper');
  const mImg = document.getElementById('minimap-img');
  const box = document.getElementById('minimap-box');
  const posLabel = document.getElementById('val-pos');

  if (typeof overrideCx !== 'number' || isNaN(overrideCx)) overrideCx = undefined;
  if (typeof overrideCy !== 'number' || isNaN(overrideCy)) overrideCy = undefined;

  const activeCx = (overrideCx !== undefined) ? overrideCx : currentCx;
  const activeCy = (overrideCy !== undefined) ? overrideCy : currentCy;

  if (posLabel) {
    if (currentViewMode === 'fit') {
      posLabel.innerText = 'Fit (Click to 100%)';
    } else {
      posLabel.innerText = Math.round(activeCx * 100) + '%, ' + Math.round(activeCy * 100) + '%';
    }
  }

  if (!mImg || !mImg.complete || mImg.naturalWidth === 0) return;

  const wRatio = mImg.offsetWidth / mImg.naturalWidth;
  const hRatio = mImg.offsetHeight / mImg.naturalHeight;
  const scale = Math.min(wRatio, hRatio);

  const renderedW = mImg.naturalWidth * scale;
  const renderedH = mImg.naturalHeight * scale;
  const offsetX = (wrapper.offsetWidth - renderedW) / 2;
  const offsetY = (wrapper.offsetHeight - renderedH) / 2;

  if (currentViewMode === 'fit') {
    box.style.display = 'block';
    box.style.left = offsetX + 'px';
    box.style.top = offsetY + 'px';
    box.style.width = renderedW + 'px';
    box.style.height = renderedH + 'px';
    box.style.cursor = 'crosshair';
    box.style.background = 'transparent';
    box.style.boxShadow = 'none';
    return;
  }

  box.style.cursor = 'move';
  box.style.background = 'rgba(56, 189, 248, 0.22)';
  box.style.boxShadow = '0 0 10px rgba(56, 189, 248, 0.7)';
  let [rx, ry, rw, rh] = currentCropRect;
  if (overrideCx !== undefined) {
    rx = Math.max(0, Math.min(1.0 - rw, activeCx - rw / 2));
    ry = Math.max(0, Math.min(1.0 - rh, activeCy - rh / 2));
  }
  box.style.display = 'block';
  box.style.left = (offsetX + rx * renderedW) + 'px';
  box.style.top = (offsetY + ry * renderedH) + 'px';
  box.style.width = Math.max(16, rw * renderedW) + 'px';
  box.style.height = Math.max(16, rh * renderedH) + 'px';
}

function onMinimapClick(e) {
  const wrapper = document.getElementById('minimap-wrapper');
  const mImg = document.getElementById('minimap-img');
  if (!mImg || mImg.naturalWidth === 0) return;

  const rect = mImg.getBoundingClientRect();
  const clickX = e.clientX - rect.left;
  const clickY = e.clientY - rect.top;

  const u = Math.max(0, Math.min(1, clickX / rect.width));
  const v = Math.max(0, Math.min(1, clickY / rect.height));

  currentCx = u;
  currentCy = v;

  if (currentViewMode === 'fit') {
    setViewMode('100%');
    return;
  }

  updateMinimapBox();

  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(fetchPreview, 50);
}

function fetchPreview() {
  showLoadingBadge(true);

  if (currentAbortController) {
    currentAbortController.abort();
  }
  currentAbortController = new AbortController();

  const mode = currentViewMode;
  const vp = document.getElementById('viewport');
  const vw = vp ? Math.max(400, Math.round(vp.clientWidth - 40)) : 1600;
  const vh = vp ? Math.max(300, Math.round(vp.clientHeight - 40)) : 1000;

  const params = new URLSearchParams({
    method: 'egmsd',
    z_jump: document.getElementById('z_jump') ? document.getElementById('z_jump').value : '5.0',
    halo_gamma: document.getElementById('halo_gamma') ? document.getElementById('halo_gamma').value : '0.45',
    anisotropy: document.getElementById('anisotropy') ? document.getElementById('anisotropy').value : '0.50',
    threshold: document.getElementById('threshold') ? document.getElementById('threshold').value : '1.0',
    specular_thresh: document.getElementById('specular_thresh') ? document.getElementById('specular_thresh').value : '0.94',
    bg_mode: currentBgMode,
    stack_order: currentStackOrder,
    mode: mode,
    cx: currentCx.toFixed(4),
    cy: currentCy.toFixed(4),
    vw: vw,
    vh: vh
  });

  fetch('/api/preview?' + params.toString(), { signal: currentAbortController.signal })
    .then(r => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    })
    .then(data => {
      const loader = document.getElementById('viewport-loader');
      if (loader) loader.style.display = 'none';

      currentImageSrc = data.image;
      currentDepthSrc = data.depth;
      currentSaliencySrc = data.saliency || data.depth;

      if (currentBgMode === 'auto' && data.effective_bg_mode) {
        const valEl = document.getElementById('val-bg_mode');
        if (valEl) {
          const effCap = data.effective_bg_mode.charAt(0).toUpperCase() + data.effective_bg_mode.slice(1);
          valEl.innerText = `Auto (${effCap})`;
        }
      }

      if (data.num_frames && data.num_frames > 0) {
        numFrames = data.num_frames;
        frameNames = data.frame_names || [];
        const slider = document.getElementById('retouch-frame');
        if (slider) {
          slider.max = numFrames;
          updateFrameLabel();
        }
      }

      const img = document.getElementById('preview-img');
      img.onload = function() {
        img.className = 'loaded ' + (currentViewMode !== 'fit' ? 'mode-roi' : 'mode-fit');
        img.style.display = 'block';
        img.style.transform = '';
        const rCanvas = document.getElementById('retouch-canvas');
        if (rCanvas) rCanvas.style.transform = '';
        const cCont = document.getElementById('canvas-container');
        if (cCont) cCont.className = (currentViewMode !== 'fit' ? 'mode-roi' : 'mode-fit');
        const oImg = document.getElementById('pan-overview-img');
        if (oImg) {
          oImg.style.transform = '';
          oImg.style.filter = '';
          oImg.style.opacity = '1';
        }
        showLoadingBadge(false);
        syncCanvasDimensions();
        syncPanOverview();
      };
      
      updateViewportSource();

      if (mode === 'fit') {
        overviewImageSrc = data.image;
        overviewDepthSrc = data.depth;
        overviewSaliencySrc = data.saliency || data.depth;
        const mImg = document.getElementById('minimap-img');
        if (mImg && mImg.src !== overviewImageSrc) {
          mImg.onload = updateMinimapBox;
          mImg.src = overviewImageSrc;
        }
      } else if (!overviewImageSrc) {
        const ovParams = new URLSearchParams(params);
        ovParams.set('mode', 'fit');
        fetch('/api/preview?' + ovParams.toString())
          .then(r => r.json())
          .then(ovData => {
            overviewImageSrc = ovData.image;
            overviewDepthSrc = ovData.depth;
            overviewSaliencySrc = ovData.saliency || ovData.depth;
            const mImg = document.getElementById('minimap-img');
            if (mImg && mImg.src !== overviewImageSrc) {
              mImg.onload = updateMinimapBox;
              mImg.src = overviewImageSrc;
            }
          })
          .catch(() => {});
      }

      if (data.crop_rect) {
        currentCropRect = data.crop_rect;
        updateMinimapBox();
        syncPanOverview();
      }

      document.getElementById('img-info').innerText = data.info;
    })
    .catch(err => {
      if (err.name === 'AbortError') return;
      showLoadingBadge(false);
      const loader = document.getElementById('viewport-loader');
      if (loader) loader.style.display = 'none';
      console.error("Preview error:", err);
      document.getElementById('img-info').innerText = "Preview notice: " + err;
    });
}

function syncCanvasDimensions() {
  const img = document.getElementById('preview-img');
  const canvas = document.getElementById('retouch-canvas');
  if (!img || !img.naturalWidth) return;

  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.style.width = img.offsetWidth + 'px';
  canvas.style.height = img.offsetHeight + 'px';

  if (!masterMaskInitialized) {
    const aspect = img.naturalHeight / img.naturalWidth;
    masterMaskCanvas.width = 2400;
    masterMaskCanvas.height = Math.round(2400 * aspect);
    masterMaskCtx.clearRect(0, 0, masterMaskCanvas.width, masterMaskCanvas.height);
    masterMaskInitialized = true;
  }
  redrawRetouchCanvas();
}

// Retouch Brush & Controls
function toggleBrush() {
  isBrushActive = !isBrushActive;
  const btn = document.getElementById('btn-brush-toggle');
  const panel = document.getElementById('retouch-panel');
  const canvas = document.getElementById('retouch-canvas');
  const cursor = document.getElementById('brush-cursor');
  const vp = document.getElementById('viewport');

  if (btn) btn.classList.toggle('active', isBrushActive);
  if (panel) panel.style.display = isBrushActive ? 'flex' : 'none';
  if (canvas) canvas.classList.toggle('active', isBrushActive);
  if (vp) vp.classList.toggle('brush-active', isBrushActive);
  if (cursor) cursor.style.display = 'none';

  if (isBrushActive) {
    syncCanvasDimensions();
    prefetchFrameImage(currentFrameIdx);
    updateFrameLabel();
    if (lastMouseX !== null && lastMouseY !== null) {
      const vpRect = vp ? vp.getBoundingClientRect() : (canvas ? canvas.getBoundingClientRect() : null);
      if (vpRect && lastMouseX >= vpRect.left && lastMouseX <= vpRect.right &&
          lastMouseY >= vpRect.top && lastMouseY <= vpRect.bottom) {
        if (cursor) cursor.style.display = 'block';
        updateBrushCursor(lastMouseX, lastMouseY);
      }
    }
  }
}

function updateFrameLabel() {
  const lbl = document.getElementById('frame-indicator');
  const fnLbl = document.getElementById('frame-filename');
  const bfi = document.getElementById('brush-frame-indicator');
  if (lbl) lbl.innerText = `Frame ${currentFrameIdx + 1} / ${numFrames}`;
  if (bfi) {
    if (brushToolMode === 'erase') {
      bfi.innerText = `Erasing to Stack`;
      bfi.style.color = '#f87171';
    } else {
      bfi.innerText = `Frame ${currentFrameIdx + 1} / ${numFrames}`;
      bfi.style.color = '#60a5fa';
    }
  }
  if (fnLbl && frameNames[currentFrameIdx]) {
    fnLbl.innerText = frameNames[currentFrameIdx];
  }
}

function setBrushToolMode(mode) {
  brushToolMode = (mode === 'erase') ? 'erase' : 'paint';
  const btnPaint = document.getElementById('btn-brush-paint');
  const btnErase = document.getElementById('btn-brush-erase');
  const cursor = document.getElementById('brush-cursor');
  if (btnPaint) btnPaint.classList.toggle('active', brushToolMode === 'paint');
  if (btnErase) btnErase.classList.toggle('active', brushToolMode === 'erase');
  if (cursor) cursor.classList.toggle('mode-erase', brushToolMode === 'erase');
  updateFrameLabel();
  if (!isBrushActive) {
    toggleBrush();
  }
}

function onSliderStartBlink() {
  if (isBlinking || isPeaking) return;
  isSliderBlinking = true;
  isBlinking = true;
  const c = document.getElementById('retouch-canvas');
  if (c) c.style.display = 'none';
  const img = document.getElementById('preview-img');
  prefetchFrameImage(currentFrameIdx).then(fImg => {
    if (isSliderBlinking && fImg) {
      img.src = fImg.src;
    }
  });
}

function onSliderStopBlink() {
  if (!isSliderBlinking) return;
  isSliderBlinking = false;
  isBlinking = false;
  const c = document.getElementById('retouch-canvas');
  if (c && isBrushActive) c.style.display = 'block';
  updateViewportSource();
}

function stepFrame(delta) {
  let next = currentFrameIdx + delta;
  if (next < 0) next = 0;
  if (next >= numFrames) next = numFrames - 1;
  if (currentFrameIdx === next) return;
  currentFrameIdx = next;
  const rf = document.getElementById('retouch-frame');
  if (rf) rf.value = currentFrameIdx + 1;
  updateFrameLabel();

  if (isBlinking || isSliderBlinking) {
    const img = document.getElementById('preview-img');
    prefetchFrameImage(currentFrameIdx).then(fImg => {
      if ((isBlinking || isSliderBlinking) && fImg) img.src = fImg.src;
    });
  } else if (isPeaking) {
    const img = document.getElementById('preview-img');
    prefetchPeakingImage(currentFrameIdx).then(pImg => {
      if (isPeaking && pImg) img.src = pImg.src;
    });
  } else {
    prefetchFrameImage(currentFrameIdx);
  }
}

function onFrameSliderChange() {
  currentFrameIdx = parseInt(document.getElementById('retouch-frame').value) - 1;
  updateFrameLabel();

  if (isBlinking || isSliderBlinking) {
    const img = document.getElementById('preview-img');
    prefetchFrameImage(currentFrameIdx).then(fImg => {
      if ((isBlinking || isSliderBlinking) && fImg) img.src = fImg.src;
    });
  } else if (isPeaking) {
    const img = document.getElementById('preview-img');
    prefetchPeakingImage(currentFrameIdx).then(pImg => {
      if (isPeaking && pImg) img.src = pImg.src;
    });
  } else {
    prefetchFrameImage(currentFrameIdx);
  }
}

function updateBrushSize() {
  brushRadius = parseInt(document.getElementById('brush-size').value);
  document.getElementById('val-brush-size').innerText = brushRadius + 'px';
  updateBrushCursor();
}

function changeBrushSize(delta) {
  brushRadius = Math.max(10, Math.min(250, brushRadius + delta));
  document.getElementById('brush-size').value = brushRadius;
  document.getElementById('val-brush-size').innerText = brushRadius + 'px';
  updateBrushCursor();
}

function updateBrushFeather() {
  brushFeather = parseInt(document.getElementById('brush-feather').value);
  document.getElementById('val-brush-feather').innerText = brushFeather + '%';
  updateBrushCursor();
}

function changeBrushFeather(delta) {
  brushFeather = Math.max(0, Math.min(100, brushFeather + delta));
  const el = document.getElementById('brush-feather');
  if (el) el.value = brushFeather;
  const valEl = document.getElementById('val-brush-feather');
  if (valEl) valEl.innerText = brushFeather + '%';
  updateBrushCursor();
}

function updateBrushOpacity() {
  brushOpacity = parseInt(document.getElementById('brush-opacity').value) / 100.0;
  document.getElementById('val-brush-opacity').innerText = Math.round(brushOpacity * 100) + '%';
}

function updateBrushCursor(x, y) {
  const cursor = document.getElementById('brush-cursor');
  if (!cursor || !isBrushActive) return;
  cursor.style.width = (brushRadius * 2) + 'px';
  cursor.style.height = (brushRadius * 2) + 'px';
  if (x !== undefined && y !== undefined) {
    cursor.style.left = x + 'px';
    cursor.style.top = y + 'px';
  } else if (lastMouseX !== null && lastMouseY !== null) {
    cursor.style.left = lastMouseX + 'px';
    cursor.style.top = lastMouseY + 'px';
  }
  const inner = document.getElementById('brush-cursor-inner');
  if (inner) {
    const solidRatio = Math.max(0, 1.0 - (brushFeather / 100.0));
    if (brushFeather > 0 && brushFeather < 100) {
      inner.style.display = 'block';
      const innerDiam = Math.max(2, Math.round(brushRadius * 2 * solidRatio));
      inner.style.width = innerDiam + 'px';
      inner.style.height = innerDiam + 'px';
    } else {
      inner.style.display = 'none';
    }
  }
}

function getFrameCacheParams(frameIdx) {
  const vp = document.getElementById('viewport');
  const vw = vp ? Math.max(400, Math.round(vp.clientWidth - 40)) : 1600;
  const vh = vp ? Math.max(300, Math.round(vp.clientHeight - 40)) : 1000;
  const key = `${frameIdx}_${currentViewMode}_${currentCx.toFixed(4)}_${currentCy.toFixed(4)}_${vw}_${vh}`;
  return { key, vw, vh };
}

function prefetchFrameImage(frameIdx) {
  const { key, vw, vh } = getFrameCacheParams(frameIdx);
  if (cachedFrameImages[key]) return Promise.resolve(cachedFrameImages[key]);
  if (inFlightFrameFetches[key]) return inFlightFrameFetches[key];

  const params = new URLSearchParams({
    index: frameIdx,
    mode: currentViewMode,
    cx: currentCx.toFixed(4),
    cy: currentCy.toFixed(4),
    vw: vw,
    vh: vh
  });

  inFlightFrameFetches[key] = fetch('/api/frame?' + params.toString())
    .then(r => r.json())
    .then(data => {
      const img = new Image();
      return new Promise(resolve => {
        img.onload = () => {
          cachedFrameImages[key] = img;
          delete inFlightFrameFetches[key];
          resolve(img);
        };
        img.onerror = () => {
          delete inFlightFrameFetches[key];
          resolve(null);
        };
        img.src = data.frame;
      });
    })
    .catch(() => {
      delete inFlightFrameFetches[key];
      return null;
    });

  return inFlightFrameFetches[key];
}

function startBlink() {
  isBlinking = true;
  document.getElementById('btn-blink').classList.add('active');
  const img = document.getElementById('preview-img');
  prefetchFrameImage(currentFrameIdx).then(fImg => {
    if (isBlinking && fImg) {
      img.src = fImg.src;
      document.getElementById('retouch-canvas').style.opacity = '0';
    }
  });
}

function stopBlink() {
  if (!isBlinking) return;
  isBlinking = false;
  document.getElementById('btn-blink').classList.remove('active');
  document.getElementById('retouch-canvas').style.opacity = '1';
  updateViewportSource();
}

function prefetchPeakingImage(frameIdx) {
  const vp = document.getElementById('viewport');
  const vw = vp ? Math.max(400, Math.round(vp.clientWidth - 40)) : 1600;
  const vh = vp ? Math.max(300, Math.round(vp.clientHeight - 40)) : 1000;
  const key = `${frameIdx}_${currentViewMode}_${currentCx.toFixed(4)}_${currentCy.toFixed(4)}_${vw}_${vh}`;
  if (cachedPeakingImages[key]) return Promise.resolve(cachedPeakingImages[key]);

  const params = new URLSearchParams({
    index: frameIdx,
    mode: currentViewMode,
    cx: currentCx.toFixed(4),
    cy: currentCy.toFixed(4),
    vw: vw,
    vh: vh,
    peaking: 'true'
  });

  return fetch('/api/frame?' + params.toString())
    .then(r => r.json())
    .then(data => {
      const img = new Image();
      return new Promise(resolve => {
        img.onload = () => {
          cachedPeakingImages[key] = img;
          resolve(img);
        };
        img.src = data.frame;
      });
    })
    .catch(() => null);
}

function startPeaking() {
  isPeaking = true;
  const btn = document.getElementById('btn-peaking');
  if (btn) btn.classList.add('active');
  const img = document.getElementById('preview-img');
  prefetchPeakingImage(currentFrameIdx).then(pImg => {
    if (isPeaking && pImg) {
      img.src = pImg.src;
      document.getElementById('retouch-canvas').style.opacity = '0';
    }
  });
}

function stopPeaking() {
  if (!isPeaking) return;
  isPeaking = false;
  const btn = document.getElementById('btn-peaking');
  if (btn) btn.classList.remove('active');
  document.getElementById('retouch-canvas').style.opacity = '1';
  updateViewportSource();
}

// Canvas Painting & Compositing
const retouchCanvas = document.getElementById('retouch-canvas');
const retouchCtx = retouchCanvas.getContext('2d');

function syncMasterFromFrameMasks() {
  if (!masterMaskInitialized || !masterMaskCanvas.width || !masterMaskCanvas.height) return;
  const w = masterMaskCanvas.width;
  const h = masterMaskCanvas.height;
  const masterData = masterMaskCtx.createImageData(w, h);
  const m = masterData.data;

  const frameKeys = Object.keys(frameMaskCanvases);
  if (frameKeys.length === 0) {
    masterMaskCtx.clearRect(0, 0, w, h);
    hasRetouchOverrides = false;
    return;
  }

  const frameEntries = [];
  for (let k = 0; k < frameKeys.length; k++) {
    const fIdx = parseInt(frameKeys[k]);
    const canvas = frameMaskCanvases[fIdx];
    if (canvas) {
      frameEntries.push({
        idx: fIdx,
        data: canvas.getContext('2d').getImageData(0, 0, w, h).data
      });
    }
  }

  let totalActive = 0;
  for (let i = 0; i < m.length; i += 4) {
    let bestFrame = 0;
    let bestAlpha = 0;
    for (let f = 0; f < frameEntries.length; f++) {
      const a = frameEntries[f].data[i + 3];
      if (a > bestAlpha) {
        bestAlpha = a;
        bestFrame = frameEntries[f].idx;
      }
    }
    if (bestAlpha > 0) {
      totalActive++;
      m[i] = bestFrame;       // R: Frame index
      m[i + 1] = bestAlpha;   // G: Alpha opacity (0..255)
      m[i + 2] = bestFrame;   // B: Frame index
      m[i + 3] = 255;         // A: 255 opaque to prevent premultiplied alpha corruption
    }
  }
  masterMaskCtx.putImageData(masterData, 0, 0);
  hasRetouchOverrides = (totalActive > 0);
}

function rebuildFrameMasksFromMaster() {
  frameMaskCanvases = {};
  activeOverrideFrames.clear();
  hasRetouchOverrides = false;
  if (!masterMaskCanvas.width || !masterMaskCanvas.height) return;
  const maskData = masterMaskCtx.getImageData(0, 0, masterMaskCanvas.width, masterMaskCanvas.height);
  const data = maskData.data;
  const framesFound = new Set();
  for (let i = 0; i < data.length; i += 4) {
    const alpha = (data[i + 3] === 255) ? data[i + 1] : data[i + 3];
    if (alpha > 0) {
      framesFound.add(data[i]);
      hasRetouchOverrides = true;
    }
  }
  framesFound.forEach(fIdx => {
    activeOverrideFrames.add(fIdx);
    const c = document.createElement('canvas');
    c.width = masterMaskCanvas.width;
    c.height = masterMaskCanvas.height;
    const ctx = c.getContext('2d');
    const fData = ctx.createImageData(c.width, c.height);
    const fd = fData.data;
    for (let i = 0; i < data.length; i += 4) {
      const alpha = (data[i + 3] === 255) ? data[i + 1] : data[i + 3];
      if (alpha > 0 && data[i] === fIdx) {
        fd[i] = 255;
        fd[i + 1] = 255;
        fd[i + 2] = 255;
        fd[i + 3] = alpha;
      }
    }
    ctx.putImageData(fData, 0, 0);
    frameMaskCanvases[fIdx] = c;
  });
}

function redrawRetouchCanvas() {
  const img = document.getElementById('preview-img');
  if (!img || !img.naturalWidth) return;

  retouchCtx.clearRect(0, 0, retouchCanvas.width, retouchCanvas.height);

  if (masterMaskCanvas.width === 0 || !hasRetouchOverrides) return;

  const [rx, ry, rw, rh] = currentCropRect;
  const sx = rx * masterMaskCanvas.width;
  const sy = ry * masterMaskCanvas.height;
  const sw = rw * masterMaskCanvas.width;
  const sh = rh * masterMaskCanvas.height;

  if (currentVisualMode === 'depth') {
    // Render depth representation of painted masks
    if (!window._offscreenDepthCanvas) {
      window._offscreenDepthCanvas = document.createElement('canvas');
      window._offscreenDepthCtx = window._offscreenDepthCanvas.getContext('2d');
    }
    const offCanvas = window._offscreenDepthCanvas;
    const offCtx = window._offscreenDepthCtx;
    if (offCanvas.width !== masterMaskCanvas.width || offCanvas.height !== masterMaskCanvas.height) {
      offCanvas.width = masterMaskCanvas.width;
      offCanvas.height = masterMaskCanvas.height;
    }
    const maskData = masterMaskCtx.getImageData(0, 0, masterMaskCanvas.width, masterMaskCanvas.height);
    const outData = offCtx.createImageData(masterMaskCanvas.width, masterMaskCanvas.height);
    const m = maskData.data;
    const o = outData.data;
    const nF = Math.max(1, numFrames - 1);

    for (let i = 0; i < m.length; i += 4) {
      const alpha = (m[i + 3] === 255) ? m[i + 1] : m[i + 3];
      if (alpha > 0) {
        const fIdx = m[i]; // red channel contains frame index
        const g = Math.round((fIdx * 255.0) / nF);
        o[i] = g;
        o[i + 1] = g;
        o[i + 2] = g;
        o[i + 3] = alpha;
      }
    }
    offCtx.putImageData(outData, 0, 0);
    retouchCtx.drawImage(offCanvas, sx, sy, sw, sh, 0, 0, retouchCanvas.width, retouchCanvas.height);
  } else {
    // Render RGB blend of painted frame(s) onto canvas
    if (!window._retouchLayerCanvas) {
      window._retouchLayerCanvas = document.createElement('canvas');
      window._retouchLayerCtx = window._retouchLayerCanvas.getContext('2d');
    }
    const layerCanvas = window._retouchLayerCanvas;
    const layerCtx = window._retouchLayerCtx;
    if (layerCanvas.width !== retouchCanvas.width || layerCanvas.height !== retouchCanvas.height) {
      layerCanvas.width = retouchCanvas.width;
      layerCanvas.height = retouchCanvas.height;
    }

    const frameKeys = Object.keys(frameMaskCanvases);
    for (const fStr of frameKeys) {
      const fIdx = parseInt(fStr);
      const maskCanvas = frameMaskCanvases[fIdx];
      if (!maskCanvas) continue;

      const { key: cacheKey } = getFrameCacheParams(fIdx);
      const fImg = cachedFrameImages[cacheKey];

      if (fImg) {
        layerCtx.clearRect(0, 0, layerCanvas.width, layerCanvas.height);
        layerCtx.save();
        layerCtx.drawImage(fImg, 0, 0, layerCanvas.width, layerCanvas.height);
        layerCtx.globalCompositeOperation = 'destination-in';
        layerCtx.drawImage(maskCanvas, sx, sy, sw, sh, 0, 0, layerCanvas.width, layerCanvas.height);
        layerCtx.restore();

        retouchCtx.drawImage(layerCanvas, 0, 0);
      } else {
        prefetchFrameImage(fIdx).then(loadedImg => {
          if (loadedImg) redrawRetouchCanvas();
        });
      }
    }
  }
}

let lastPaintX = null;
let lastPaintY = null;

function getCanvasCoords(e) {
  const rect = retouchCanvas.getBoundingClientRect();
  const w = Math.max(1, rect.width);
  const h = Math.max(1, rect.height);
  const fx = Math.max(0, Math.min(1, (e.clientX - rect.left) / w));
  const fy = Math.max(0, Math.min(1, (e.clientY - rect.top) / h));
  return { fx, fy, cssWidth: w, cssHeight: h };
}

function paintAtCoords(fx, fy, cssWidth) {
  if (!masterMaskInitialized) {
    syncCanvasDimensions();
  }
  const [rx, ry, rw, rh] = currentCropRect;
  const u = rx + fx * rw;
  const v = ry + fy * rh;
  const mx = u * masterMaskCanvas.width;
  const my = v * masterMaskCanvas.height;
  const mRad = (brushRadius / cssWidth) * rw * masterMaskCanvas.width;

  const solidRatio = Math.max(0, Math.min(0.999, 1.0 - (brushFeather / 100.0)));

  if (brushToolMode === 'erase') {
    // ERASER MODE: Erase manual retouch overrides from per-frame masks
    const grad = masterMaskCtx.createRadialGradient(mx, my, 0, mx, my, Math.max(1, mRad));
    grad.addColorStop(0, `rgba(0, 0, 0, ${brushOpacity})`);
    if (solidRatio > 0.01) {
      grad.addColorStop(solidRatio, `rgba(0, 0, 0, ${brushOpacity})`);
    }
    grad.addColorStop(1, 'rgba(0, 0, 0, 0)');

    for (const fIdx in frameMaskCanvases) {
      const fCtx = frameMaskCanvases[fIdx].getContext('2d');
      fCtx.save();
      fCtx.globalCompositeOperation = 'destination-out';
      fCtx.fillStyle = grad;
      fCtx.beginPath();
      fCtx.arc(mx, my, Math.max(1, mRad), 0, Math.PI * 2);
      fCtx.fill();
      fCtx.restore();
    }
    return;
  }

  // Update per-frame mask canvas
  if (!frameMaskCanvases[currentFrameIdx]) {
    const fmc = document.createElement('canvas');
    fmc.width = masterMaskCanvas.width;
    fmc.height = masterMaskCanvas.height;
    frameMaskCanvases[currentFrameIdx] = fmc;
  }
  const fCtx = frameMaskCanvases[currentFrameIdx].getContext('2d');
  fCtx.save();
  const fGrad = fCtx.createRadialGradient(mx, my, 0, mx, my, Math.max(1, mRad));
  fGrad.addColorStop(0, `rgba(255, 255, 255, ${brushOpacity})`);
  if (solidRatio > 0.01) {
    fGrad.addColorStop(solidRatio, `rgba(255, 255, 255, ${brushOpacity})`);
  }
  fGrad.addColorStop(1, `rgba(255, 255, 255, 0)`);
  fCtx.fillStyle = fGrad;
  fCtx.beginPath();
  fCtx.arc(mx, my, Math.max(1, mRad), 0, Math.PI * 2);
  fCtx.fill();
  fCtx.restore();

  // Erase from other frame masks so latest paint wins
  for (const otherIdx in frameMaskCanvases) {
    if (parseInt(otherIdx) !== currentFrameIdx) {
      const oCtx = frameMaskCanvases[otherIdx].getContext('2d');
      oCtx.save();
      oCtx.globalCompositeOperation = 'destination-out';
      oCtx.fillStyle = fGrad;
      oCtx.beginPath();
      oCtx.arc(mx, my, Math.max(1, mRad), 0, Math.PI * 2);
      oCtx.fill();
      oCtx.restore();
    }
  }

  activeOverrideFrames.add(currentFrameIdx);
  hasRetouchOverrides = true;
}

function paintInterpolated(fx, fy, cssWidth) {
  if (lastPaintX === null) {
    paintAtCoords(fx, fy, cssWidth);
    lastPaintX = fx;
    lastPaintY = fy;
    redrawRetouchCanvas();
    return;
  }
  const dfx = fx - lastPaintX;
  const dfy = fy - lastPaintY;
  const dist = Math.hypot(dfx * cssWidth, dfy * cssWidth);
  const stepSize = Math.max(2, brushRadius / 4.0);
  const steps = Math.ceil(dist / stepSize);

  for (let s = 1; s <= steps; s++) {
    const t = s / steps;
    const ifx = lastPaintX + dfx * t;
    const ify = lastPaintY + dfy * t;
    paintAtCoords(ifx, ify, cssWidth);
  }
  lastPaintX = fx;
  lastPaintY = fy;
  redrawRetouchCanvas();
}

function undoStroke() {
  if (strokeHistory.length > 0) {
    const prev = strokeHistory.pop();
    masterMaskCtx.putImageData(prev, 0, 0);
    rebuildFrameMasksFromMaster();
    redrawRetouchCanvas();
    showToast("Undid last brush stroke", 1500);
  }
}

function clearRetouch() {
  if (confirm("Clear all manual retouching overrides?")) {
    masterMaskCtx.clearRect(0, 0, masterMaskCanvas.width, masterMaskCanvas.height);
    frameMaskCanvases = {};
    activeOverrideFrames.clear();
    hasRetouchOverrides = false;
    strokeHistory = [];
    redrawRetouchCanvas();
    showToast("Retouching cleared", 1500);
  }
}

// Viewport Mouse Events for Painting & Hover-Constrained Brush Cursor
function handleBrushMove(e) {
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  if (isBrushActive) {
    const cursor = document.getElementById('brush-cursor');
    if (cursor && cursor.style.display !== 'block') {
      cursor.style.display = 'block';
    }
    updateBrushCursor(e.clientX, e.clientY);
  }
}

retouchCanvas.addEventListener('mouseenter', handleBrushMove);
retouchCanvas.addEventListener('mousemove', handleBrushMove);

retouchCanvas.addEventListener('mouseleave', function(e) {
  if (!isPainting) {
    const vp = document.getElementById('viewport');
    if (vp) {
      const rect = vp.getBoundingClientRect();
      if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
        const cursor = document.getElementById('brush-cursor');
        if (cursor) cursor.style.display = 'none';
      }
    } else {
      const cursor = document.getElementById('brush-cursor');
      if (cursor) cursor.style.display = 'none';
    }
  }
});

retouchCanvas.addEventListener('mousedown', function(e) {
  if (isSpacePressed) {
    startPanDrag(e);
    return;
  }
  if (!isBrushActive || e.button !== 0) return;
  isPainting = true;
  strokeHistory.push(masterMaskCtx.getImageData(0, 0, masterMaskCanvas.width, masterMaskCanvas.height));
  if (strokeHistory.length > 20) strokeHistory.shift();

  const coords = getCanvasCoords(e);
  lastPaintX = coords.fx;
  lastPaintY = coords.fy;
  paintAtCoords(coords.fx, coords.fy, coords.cssWidth);
  redrawRetouchCanvas();
  e.preventDefault();
});

window.addEventListener('mousemove', function(e) {
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
  if (!isPainting || !isBrushActive) return;
  const cursor = document.getElementById('brush-cursor');
  if (cursor && cursor.style.display !== 'block') {
    cursor.style.display = 'block';
  }
  updateBrushCursor(e.clientX, e.clientY);
  const coords = getCanvasCoords(e);
  paintInterpolated(coords.fx, coords.fy, coords.cssWidth);
});

window.addEventListener('mouseup', function(e) {
  if (isPainting) {
    isPainting = false;
    lastPaintX = null;
    lastPaintY = null;
    syncMasterFromFrameMasks();
    redrawRetouchCanvas();
    const vp = document.getElementById('viewport');
    const rect = vp ? vp.getBoundingClientRect() : retouchCanvas.getBoundingClientRect();
    if (e.clientX < rect.left || e.clientX > rect.right || e.clientY < rect.top || e.clientY > rect.bottom) {
      const cursor = document.getElementById('brush-cursor');
      if (cursor) cursor.style.display = 'none';
    }
  }
});

function onBrushWheel(e) {
  if (!isBrushActive) return;
  const vp = document.getElementById('viewport');
  const rect = vp ? vp.getBoundingClientRect() : retouchCanvas.getBoundingClientRect();
  const isOverViewport = (
    e.clientX >= rect.left && e.clientX <= rect.right &&
    e.clientY >= rect.top && e.clientY <= rect.bottom
  );
  if (!isOverViewport) return;

  e.preventDefault();
  // Slower scroll with multiplier, and inverted direction: deltaY > 0 increases size
  brushWheelAccumulator += (e.deltaY * BRUSH_SCROLL_MULTIPLIER);
  if (Math.abs(brushWheelAccumulator) >= 1) {
    const delta = Math.trunc(brushWheelAccumulator);
    brushWheelAccumulator -= delta;
    changeBrushSize(delta);
    if (lastMouseX !== null && lastMouseY !== null) {
      updateBrushCursor(lastMouseX, lastMouseY);
    }
  }
}

retouchCanvas.addEventListener('wheel', onBrushWheel, { passive: false });

let zoomWheelAccumulator = 0;
function onViewportWheel(e) {
  if (isBrushActive) {
    onBrushWheel(e);
    return;
  }
  e.preventDefault();
  zoomWheelAccumulator += e.deltaY;
  if (zoomWheelAccumulator <= -60) {
    zoomWheelAccumulator = 0;
    zoomIn();
  } else if (zoomWheelAccumulator >= 60) {
    zoomWheelAccumulator = 0;
    zoomOut();
  }
}

const vpEl = document.getElementById('viewport');
if (vpEl) {
  vpEl.addEventListener('mouseenter', handleBrushMove);
  vpEl.addEventListener('mousemove', handleBrushMove);
  vpEl.addEventListener('mouseleave', function() {
    if (!isPainting) {
      const cursor = document.getElementById('brush-cursor');
      if (cursor) cursor.style.display = 'none';
    }
  });
  vpEl.addEventListener('wheel', onViewportWheel, { passive: false });
}

const sidebarEl = document.getElementById('sidebar');
if (sidebarEl) {
  sidebarEl.addEventListener('mouseenter', function() {
    if (!isPainting) {
      const cursor = document.getElementById('brush-cursor');
      if (cursor) cursor.style.display = 'none';
    }
  });
}

window.addEventListener('mouseup', function() {
  if (isSliderBlinking) {
    onSliderStopBlink();
  }
});

window.addEventListener('resize', function() {
  syncCanvasDimensions();
});

// Interactive Pan / Drag handlers on Viewport when brush is not active (or spacebar is held)
const previewImg = document.getElementById('preview-img');

function startPanDrag(e) {
  if (currentViewMode === 'fit' || (isBrushActive && !isSpacePressed) || e.button !== 0) return;
  isDraggingImage = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  const cCont = document.getElementById('canvas-container');
  const pImg = document.getElementById('preview-img');
  if (cCont) cCont.classList.add('grabbing');
  if (pImg) pImg.classList.add('grabbing');
  syncPanOverview();
  e.preventDefault();
}

previewImg.addEventListener('mousedown', startPanDrag);

window.addEventListener('mousemove', function(e) {
  if (!isDraggingImage) return;
  const dx = e.clientX - dragStartX;
  const dy = e.clientY - dragStartY;
  const pImg = document.getElementById('preview-img');
  const oImg = document.getElementById('pan-overview-img');
  const rCanvas = document.getElementById('retouch-canvas');
  if (pImg) pImg.style.transform = `translate(${dx}px, ${dy}px)`;
  if (oImg) oImg.style.transform = `translate(${dx}px, ${dy}px)`;
  if (rCanvas) rCanvas.style.transform = `translate(${dx}px, ${dy}px)`;
  updateMinimapBoxLive(dx, dy);
});

function updateMinimapBoxLive(dx, dy) {
  const pImg = document.getElementById('preview-img');
  if (!pImg) return;
  const dispW = pImg.offsetWidth || pImg.naturalWidth || 800;
  const dispH = pImg.offsetHeight || pImg.naturalHeight || 800;
  const [rx, ry, rw, rh] = currentCropRect;
  const fullW = dispW / (rw || 0.1);
  const fullH = dispH / (rh || 0.1);
  const deltaU = -dx / fullW;
  const deltaV = -dy / fullH;
  const simCx = Math.max(0, Math.min(1, currentCx + deltaU));
  const simCy = Math.max(0, Math.min(1, currentCy + deltaV));
  updateMinimapBox(simCx, simCy);
}

window.addEventListener('mouseup', function(e) {
  if (!isDraggingImage) return;
  isDraggingImage = false;
  const pImg = document.getElementById('preview-img');
  const oImg = document.getElementById('pan-overview-img');
  const cCont = document.getElementById('canvas-container');
  if (cCont) cCont.classList.remove('grabbing');
  if (pImg) pImg.classList.remove('grabbing');

  const dx = e.clientX - dragStartX;
  const dy = e.clientY - dragStartY;

  if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
    // Keep low-res overview proxy at the new dragged offset while crop recalculates
    // Subtle blur and slight fade clearly indicate a temporary low-res proxy
    if (oImg) {
      oImg.style.filter = 'brightness(0.50) contrast(0.95) blur(1.5px)';
      oImg.style.opacity = '0.70';
      // keep oImg.style.transform as `translate(${dx}px, ${dy}px)`!
    }
    if (pImg) {
      pImg.style.display = 'none'; // hide old high-res crop to prevent visual snap-back jitter
    }

    const dispW = pImg ? (pImg.offsetWidth || pImg.naturalWidth || 800) : 800;
    const dispH = pImg ? (pImg.offsetHeight || pImg.naturalHeight || 800) : 800;
    const [rx, ry, rw, rh] = currentCropRect;
    const fullW = dispW / (rw || 0.1);
    const fullH = dispH / (rh || 0.1);
    const deltaU = -dx / fullW;
    const deltaV = -dy / fullH;

    currentCx = Math.max(0, Math.min(1, currentCx + deltaU));
    currentCy = Math.max(0, Math.min(1, currentCy + deltaV));
    updateMinimapBox();
    fetchPreview();
  } else {
    if (pImg) pImg.style.transform = '';
    if (oImg) oImg.style.transform = '';
    const rCanvas = document.getElementById('retouch-canvas');
    if (rCanvas) rCanvas.style.transform = '';
  }
});

// Click-to-zoom on Fit View switches to 100% Loupe at clicked point
previewImg.addEventListener('click', function(e) {
  if (currentViewMode === 'fit' && !isDraggingImage && !isBrushActive) {
    const rect = previewImg.getBoundingClientRect();
    const u = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const v = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    currentCx = u;
    currentCy = v;
    setViewMode('100%');
  }
});

// Mini-map box dragging
const minimapBox = document.getElementById('minimap-box');
minimapBox.addEventListener('mousedown', function(e) {
  isDraggingMinimap = true;
  dragStartX = e.clientX;
  dragStartY = e.clientY;
  e.stopPropagation();
  e.preventDefault();
});

window.addEventListener('mousemove', function(e) {
  if (!isDraggingMinimap) return;
  const mImg = document.getElementById('minimap-img');
  if (!mImg || mImg.naturalWidth === 0) return;

  const rect = mImg.getBoundingClientRect();
  const dx = (e.clientX - dragStartX) / rect.width;
  const dy = (e.clientY - dragStartY) / rect.height;
  dragStartX = e.clientX;
  dragStartY = e.clientY;

  currentCx = Math.max(0, Math.min(1, currentCx + dx));
  currentCy = Math.max(0, Math.min(1, currentCy + dy));
  updateMinimapBox();

  clearTimeout(previewDebounceTimer);
  previewDebounceTimer = setTimeout(fetchPreview, 60);
});

window.addEventListener('mouseup', function(e) {
  if (isDraggingMinimap) {
    isDraggingMinimap = false;
    fetchPreview();
  }
});

// Help Modal Functions
function toggleHelpModal() {
  const modal = document.getElementById('help-modal');
  if (!modal) return;
  if (modal.style.display === 'flex') {
    modal.style.display = 'none';
  } else {
    modal.style.display = 'flex';
  }
}

function closeHelpModal() {
  const modal = document.getElementById('help-modal');
  if (modal) modal.style.display = 'none';
}

function onHelpModalOverlayClick(e) {
  closeHelpModal();
}

// Global Keyboard Shortcuts
window.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  if (e.key === '?' || (e.shiftKey && e.key === '/')) {
    e.preventDefault();
    toggleHelpModal();
  } else if (e.key === 'Escape') {
    closeHelpModal();
  } else if ((e.key === 'D' || e.key === 'd') && e.shiftKey) {
    e.preventDefault();
    toggleDevMode();
  } else if (e.key === 'd' || e.key === 'D') {
    if (!isDevMode) setDevMode(true);
    setVisualMode(currentVisualMode === 'depth' ? 'image' : 'depth');
  } else if (e.key === 's' || e.key === 'S') {
    if (!isDevMode) setDevMode(true);
    setVisualMode(currentVisualMode === 'saliency' ? 'image' : 'saliency');
  } else if (e.key === 'b' || e.key === 'B') {
    if (!isBrushActive || brushToolMode !== 'paint') {
      if (!isBrushActive) toggleBrush();
      setBrushToolMode('paint');
    } else {
      toggleBrush();
    }
  } else if (e.key === 'e' || e.key === 'E') {
    if (!isBrushActive || brushToolMode !== 'erase') {
      if (!isBrushActive) toggleBrush();
      setBrushToolMode('erase');
    } else {
      toggleBrush();
    }
  } else if (e.key === '\\' || e.key === '\\\\') {
    startBlink();
  } else if (e.key === 'p' || e.key === 'P') {
    startPeaking();
  } else if (e.key === '[' || e.key === '{') {
    if (e.shiftKey || e.key === '{') {
      changeBrushFeather(-10);
    } else {
      changeBrushSize(-10);
    }
  } else if (e.key === ']' || e.key === '}') {
    if (e.shiftKey || e.key === '}') {
      changeBrushFeather(10);
    } else {
      changeBrushSize(10);
    }
  } else if (e.key === 'ArrowLeft' || e.key === ',' || e.key === '<') {
    e.preventDefault();
    stepFrame(-1);
  } else if (e.key === 'ArrowRight' || e.key === '.' || e.key === '>') {
    e.preventDefault();
    stepFrame(1);
  } else if ((e.metaKey || e.ctrlKey) && (e.key === 'z' || e.key === 'Z')) {
    e.preventDefault();
    undoStroke();
  } else if (e.key === 'z' || e.key === 'Z') {
    if (isBrushActive) {
      e.preventDefault();
      undoStroke();
    } else {
      setViewMode(currentViewMode === 'fit' ? '100%' : 'fit');
    }
  } else if (e.key === '1') {
    setViewMode('fit');
  } else if (e.key === '2') {
    setViewMode('50%');
  } else if (e.key === '3') {
    setViewMode('100%');
  } else if (e.key === '4') {
    setViewMode('200%');
  } else if (e.code === 'Space' || e.key === ' ') {
    if (!isSpacePressed) {
      isSpacePressed = true;
      const cCont = document.getElementById('canvas-container');
      if (cCont) cCont.classList.add('space-pan');
      const rCanvas = document.getElementById('retouch-canvas');
      if (rCanvas) rCanvas.style.cursor = 'grab';
    }
    e.preventDefault();
  } else if (e.key === '+' || e.key === '=') {
    zoomIn();
  } else if (e.key === '-' || e.key === '_') {
    zoomOut();
  }
});

window.addEventListener('keyup', function(e) {
  if (e.code === 'Space' || e.key === ' ') {
    isSpacePressed = false;
    const cCont = document.getElementById('canvas-container');
    if (cCont) cCont.classList.remove('space-pan');
    const rCanvas = document.getElementById('retouch-canvas');
    if (rCanvas) rCanvas.style.cursor = '';
  } else if (e.key === '\\' || e.key === '\\\\') {
    stopBlink();
  } else if (e.key === 'p' || e.key === 'P') {
    stopPeaking();
  }
});

let renderTimerInterval = null;

function cancelStacking() {
  if (confirm("Cancel focus stacking? Temporary files will be cleaned up.")) {
    fetch('/api/cancel', { method: 'POST', keepalive: true })
      .then(r => r.json())
      .then(data => {
        if (data && data.can_return_to_dropzone) {
          const dropZone = document.getElementById('drop-zone');
          const loader = document.getElementById('init-loader');
          if (loader) loader.classList.add('hidden');
          if (dropZone) dropZone.classList.remove('hidden');
          if (initPollTimer) clearInterval(initPollTimer);
          initPollTimer = setInterval(checkInitStatus, 180);
          return;
        }
        showToast("Stacking cancelled.", 1500);
        setTimeout(() => {
          try { window.close(); } catch(e) {}
          document.body.innerHTML = `
            <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100vh;background:#121214;color:#eaeaea;font-family:-apple-system,BlinkMacSystemFont,sans-serif;text-align:center;padding:24px;">
              <div style="font-size:42px;margin-bottom:14px;color:#ef4444;">✕</div>
              <h2 style="margin-bottom:8px;font-size:18px;">Stacking Cancelled</h2>
              <p style="color:#888892;font-size:13px;margin-bottom:20px;">Temporary scratch files have been cleaned up.</p>
              <button onclick="window.close()" style="background:#27272a;color:#fff;border:1px solid #3f3f46;padding:9px 18px;border-radius:6px;cursor:pointer;font-size:12px;font-weight:600;">Close Window</button>
            </div>
          `;
        }, 400);
      })
      .catch(() => {
        try { window.close(); } catch(e) {}
      });
  }
}

async function triggerFinalRender() {
  const btn = document.getElementById('btn-save');
  const modal = document.getElementById('render-modal');

  let targetOutputPath = null;
  if (isStandaloneMode) {
    try {
      const dlgResp = await fetch('/api/save_dialog', { method: 'POST' });
      if (!dlgResp.ok) throw new Error("Save dialog request failed");
      const dlgData = await dlgResp.json();
      if (dlgData.cancelled || !dlgData.save_path) {
        return; // User cancelled the Save As dialog cleanly
      }
      targetOutputPath = dlgData.save_path;
    } catch (err) {
      alert("Save As dialog error: " + err);
      return;
    }
  }

  if (btn) btn.disabled = true;
  if (modal) modal.style.display = 'flex';

  const startTime = Date.now();
  const timerEl = document.getElementById('render-timer');
  const barEl = document.getElementById('render-bar');
  const titleEl = document.getElementById('render-title');
  const descEl = document.getElementById('render-desc');

  if (titleEl) titleEl.innerText = "⚡ Computing 16-Bit Focus Stack";
  if (descEl) descEl.innerText = "Initializing multi-scale streaming engine...";
  if (barEl) barEl.style.width = '0%';

  clearInterval(renderTimerInterval);
  renderTimerInterval = setInterval(() => {
    const elapsed = Math.round((Date.now() - startTime) / 1000);
    if (timerEl) timerEl.innerText = `Elapsed: ${elapsed}s`;
  }, 500);

  let renderPollTimer = setInterval(() => {
    fetch('/api/status')
      .then(r => r.json())
      .then(st => {
        if (descEl && st.stage) {
          descEl.innerText = st.stage + (st.detail ? ` — ${st.detail}` : "");
        }
        if (barEl && typeof st.progress === 'number') {
          const pct = Math.max(0, Math.min(100, Math.round(st.progress * 100)));
          barEl.style.width = pct + '%';
        }
      })
      .catch(() => {});
  }, 250);

  let overrides = null;
  syncMasterFromFrameMasks();
  if (masterMaskCanvas.width > 0 && (hasRetouchOverrides || strokeHistory.length > 0)) {
    overrides = masterMaskCanvas.toDataURL("image/png");
  }

  const payload = {
    method: 'egmsd',
    z_jump: parseFloat(document.getElementById('z_jump')?.value || 5.0),
    halo_gamma: parseFloat(document.getElementById('halo_gamma')?.value || 0.45),
    anisotropy: parseFloat(document.getElementById('anisotropy')?.value || 0.50),
    threshold: parseFloat(document.getElementById('threshold')?.value || 1.0),
    specular_thresh: parseFloat(document.getElementById('specular_thresh')?.value || 0.94),
    bg_mode: currentBgMode,
    stack_order: currentStackOrder,
    overrides: overrides,
    output_path: targetOutputPath
  };

  fetch('/api/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(async r => {
    if (!r.ok) {
      const errTxt = await r.text();
      throw new Error(errTxt || `HTTP ${r.status}`);
    }
    return r.json();
  })
  .then(data => {
    clearInterval(renderTimerInterval);
    clearInterval(renderPollTimer);
    if (barEl) barEl.style.width = '100%';
    const title = document.getElementById('render-title');
    const desc = document.getElementById('render-desc');
    if (title) title.innerText = "✅ 16-Bit Stack Complete!";
    if (desc) {
      desc.innerText = isStandaloneMode
        ? ("Photo saved successfully to: " + (data.output_path || "destination"))
        : "Photo saved successfully. Importing into Lightroom...";
    }

    showToast("Render complete! Saved: " + (data.output_path || ""), 4000);

    setTimeout(() => {
      try { window.close(); } catch(e) {}
    }, 1800);
  })
  .catch(err => {
    clearInterval(renderTimerInterval);
    clearInterval(renderPollTimer);
    if (modal) modal.style.display = 'none';
    if (btn) btn.disabled = false;
    alert("Render error: " + err);
  });
}

// URL Parameters Initializer (for headless verification & direct view deep linking)
let initialParamsApplied = false;
function applyUrlParamsOnce() {
  if (initialParamsApplied) return;
  initialParamsApplied = true;
  const urlParams = new URLSearchParams(window.location.search);
  const qView = urlParams.get('view');
  const qVisual = urlParams.get('visual');
  const qPeaking = urlParams.get('peaking');
  if (qView && ['fit', '50%', '100%', '200%'].includes(qView)) {
    setViewMode(qView);
  }
  if (qVisual && ['image', 'depth', 'saliency'].includes(qVisual)) {
    setVisualMode(qVisual);
  }
  if (qPeaking === '1' || qPeaking === 'true') {
    startPeaking();
  }
}

// Background Worker Real-Time Status Polling
let initPollTimer = null;

function checkInitStatus() {
  fetch('/api/status')
    .then(r => r.json())
    .then(data => {
      if (typeof data.standalone !== 'undefined') {
        isStandaloneMode = !!data.standalone;
        const btnSave = document.getElementById('btn-save');
        const renderSubnote = document.getElementById('render-subnote');
        if (btnSave) {
          btnSave.innerText = isStandaloneMode ? "💾 Render & Save As..." : "⚡ Render & Import to Lightroom";
        }
        if (renderSubnote) {
          renderSubnote.innerText = isStandaloneMode
            ? "Processing 16-bit stack with float32 precision. Output will be saved to your chosen file location."
            : "Processing 61MP TIFFs with float32 precision. This window will automatically close when the image imports into Lightroom.";
        }
      }

      if (data.dropzone) {
        const dropZone = document.getElementById('drop-zone');
        const loader = document.getElementById('init-loader');
        if (dropZone) dropZone.classList.remove('hidden');
        if (loader) loader.classList.add('hidden');
        const vLoader = document.getElementById('viewport-loader');
        if (vLoader) vLoader.style.display = 'none';
        const imgInfo = document.getElementById('img-info');
        if (imgInfo) imgInfo.innerText = "Waiting for photos...";
        return;
      }

      const progressBar = document.getElementById('init-progress-bar');
      const stageText = document.getElementById('init-stage');
      const pctText = document.getElementById('init-pct');
      const detailText = document.getElementById('init-detail');

      if (stageText) stageText.innerText = data.stage;
      if (detailText) detailText.innerText = data.detail;
      const pct = Math.round((data.progress || 0) * 100);
      if (progressBar) progressBar.style.width = pct + '%';
      if (pctText) pctText.innerText = pct + '%';

      if (data.error) {
        clearInterval(initPollTimer);
        alert("Processing error: " + data.error);
        return;
      }

      if (data.ready) {
        clearInterval(initPollTimer);
        const loader = document.getElementById('init-loader');
        if (loader) {
          loader.classList.add('hidden');
        }
        fetchPreview();
        applyUrlParamsOnce();
      }
    })
    .catch(() => {});
}

// Drag and Drop & Folder/File Selection Handlers
function onDropCardClick(e) {
  browseFolder(e);
}

function showDropError(msg) {
  const errEl = document.getElementById('drop-error');
  if (errEl) {
    errEl.innerText = msg;
    errEl.style.display = 'block';
  }
}

function clearDropError() {
  const errEl = document.getElementById('drop-error');
  if (errEl) {
    errEl.innerText = '';
    errEl.style.display = 'none';
  }
}

async function submitLoadedFiles(paths) {
  clearDropError();
  try {
    const res = await fetch('/api/load_files', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths: paths || [] })
    });
    const data = await res.json();
    if (data.success) {
      const dropZone = document.getElementById('drop-zone');
      const loader = document.getElementById('init-loader');
      if (dropZone) dropZone.classList.add('hidden');
      if (loader) loader.classList.remove('hidden');

      if (initPollTimer) clearInterval(initPollTimer);
      initPollTimer = setInterval(checkInitStatus, 180);
      checkInitStatus();
    } else {
      showDropError(data.error || "Failed to load photos. Please ensure at least 2 photos are selected.");
    }
  } catch (err) {
    showDropError("Failed to communicate with stacking engine: " + err);
  }
}

async function browseFolder(e) {
  if (e) e.stopPropagation();
  clearDropError();
  try {
    const res = await fetch('/api/browse_files?type=folder', { method: 'POST' });
    const data = await res.json();
    if (data.cancelled) return;
    if (data.success) {
      const dropZone = document.getElementById('drop-zone');
      const loader = document.getElementById('init-loader');
      if (dropZone) dropZone.classList.add('hidden');
      if (loader) loader.classList.remove('hidden');

      if (initPollTimer) clearInterval(initPollTimer);
      initPollTimer = setInterval(checkInitStatus, 180);
      checkInitStatus();
    } else if (data.error) {
      showDropError(data.error);
    }
  } catch (err) {
    showDropError("Browse folder error: " + err);
  }
}

async function browseFiles(e) {
  if (e) e.stopPropagation();
  clearDropError();
  try {
    const res = await fetch('/api/browse_files?type=files', { method: 'POST' });
    const data = await res.json();
    if (data.cancelled) return;
    if (data.success) {
      const dropZone = document.getElementById('drop-zone');
      const loader = document.getElementById('init-loader');
      if (dropZone) dropZone.classList.add('hidden');
      if (loader) loader.classList.remove('hidden');

      if (initPollTimer) clearInterval(initPollTimer);
      initPollTimer = setInterval(checkInitStatus, 180);
      checkInitStatus();
    } else if (data.error) {
      showDropError(data.error);
    }
  } catch (err) {
    showDropError("Browse photos error: " + err);
  }
}

// Window Drag & Drop Event Listeners
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  window.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false);
  document.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false);
  if (document.body) document.body.addEventListener(eventName, (e) => { e.preventDefault(); e.stopPropagation(); }, false);
});

window.addEventListener('dragenter', (e) => {
  const card = document.querySelector('.drop-card');
  if (card) card.classList.add('drag-active');
});

window.addEventListener('dragover', (e) => {
  if (e.dataTransfer) e.dataTransfer.dropEffect = 'copy';
  const card = document.querySelector('.drop-card');
  if (card) card.classList.add('drag-active');
});

window.addEventListener('dragleave', (e) => {
  if (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= window.innerWidth || e.clientY >= window.innerHeight) {
    const card = document.querySelector('.drop-card');
    if (card) card.classList.remove('drag-active');
  }
});

window.addEventListener('drop', (e) => {
  const card = document.querySelector('.drop-card');
  if (card) card.classList.remove('drag-active');

  const paths = [];
  if (e.dataTransfer && e.dataTransfer.files) {
    for (let i = 0; i < e.dataTransfer.files.length; i++) {
      const f = e.dataTransfer.files[i];
      if (f.path) paths.push(f.path);
    }
  }
  if (paths.length > 0) {
    submitLoadedFiles(paths);
  } else {
    // In WKWebView Cocoa, performDragOperation_ calls submitLoadedFiles directly,
    // but we also check /api/load_files as fallback after 150ms
    setTimeout(() => {
      submitLoadedFiles([]);
    }, 150);
  }
});

initPollTimer = setInterval(checkInitStatus, 180);
checkInitStatus();
</script>
</body>
</html>
"""

def choose_folder_dialog(window=None):
    if window is not None:
        try:
            import webview
            res = window.create_file_dialog(webview.FileDialog.FOLDER)
            if res:
                return list(res)
        except Exception as e:
            sys.stderr.write(f"[Browse] webview folder dialog error: {e}\n")
    if sys.platform == 'darwin':
        try:
            cmd = ['osascript', '-e', 'POSIX path of (choose folder with prompt "Select folder containing bracketed macro photos:")']
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.strip():
                return [p.stdout.strip()]
        except Exception:
            pass
    return []

def choose_files_dialog(window=None):
    if window is not None:
        try:
            import webview
            file_types = ('Image Files (*.tif;*.tiff;*.jpg;*.jpeg;*.png;*.dng;*.arw;*.cr2;*.cr3;*.nef;*.orf;*.rw2)', 'All Files (*.*)')
            res = window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True, file_types=file_types)
            if res:
                return list(res)
        except Exception as e:
            sys.stderr.write(f"[Browse] webview file dialog error: {e}\n")
    if sys.platform == 'darwin':
        try:
            cmd = ['osascript', '-e', 'set filePaths to {}\ntell application "System Events" to activate\nset chosenFiles to (choose file with prompt "Select bracketed macro photos:" multiple selections allowed true)\nrepeat with f in chosenFiles\nset end of filePaths to POSIX path of f\nend repeat\nset AppleScript\'s text item delimiters to "\\n"\nreturn filePaths as text']
            p = subprocess.run(cmd, capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.strip():
                return [x.strip() for x in p.stdout.strip().split('\n') if x.strip()]
        except Exception:
            pass
    return []

def choose_save_dialog(window=None, default_path=None):
    save_filename = os.path.basename(default_path) if default_path else "photo_stacked.tif"
    directory = os.path.dirname(os.path.abspath(default_path)) if default_path and os.path.isabs(default_path) else os.path.expanduser("~/Pictures")
    if not os.path.exists(directory):
        directory = os.path.expanduser("~")

    if window is not None:
        try:
            import webview
            file_types = ('TIFF Image (*.tif;*.tiff)', 'JPEG Image (*.jpg;*.jpeg)', 'All Files (*.*)')
            res = window.create_file_dialog(
                webview.FileDialog.SAVE,
                directory=directory,
                save_filename=save_filename,
                file_types=file_types
            )
            if res:
                if isinstance(res, (list, tuple)):
                    return res[0]
                return str(res)
        except Exception as e:
            sys.stderr.write(f"[Save] webview save dialog error: {e}\n")

    if sys.platform == 'darwin':
        try:
            escaped_dir = directory.replace('"', '\\"')
            escaped_name = save_filename.replace('"', '\\"')
            script = (
                'tell application "System Events" to activate\n'
                f'set defaultFolder to POSIX file "{escaped_dir}"\n'
                f'set defaultName to "{escaped_name}"\n'
                'set chosen to (choose file name with prompt "Save Stacked Photo As:" default name defaultName default location defaultFolder)\n'
                'return POSIX path of chosen'
            )
            p = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            if p.returncode == 0 and p.stdout.strip():
                chosen = p.stdout.strip()
                if not os.path.splitext(chosen)[1]:
                    chosen += ".tif"
                return chosen
        except Exception as e:
            sys.stderr.write(f"[Save] AppleScript save dialog error: {e}\n")

    return None

def _start_stacking_worker(app_state, files):
    on_load_fn = app_state.get('on_load_files_callback')
    if not on_load_fn:
        raise RuntimeError("No on_load_files_callback registered")

    worker_fn, out_path = on_load_fn(files)
    app_state['output_path'] = out_path
    app_state['dropzone'] = False
    app_state['ready'] = False
    app_state['stage'] = 'Starting Stacking Engine'
    app_state['detail'] = f'Discovered {len(files)} photos'
    app_state['progress'] = 0.02
    app_state['error'] = None

    def _run():
        def update_status(stage, detail, progress):
            app_state['stage'] = str(stage)
            app_state['detail'] = str(detail)
            app_state['progress'] = max(0.0, min(1.0, float(progress)))

        def is_cancelled():
            return app_state.get('cancelled', False)

        try:
            worker_res, render_fin = worker_fn(update_status, is_cancelled)
            if is_cancelled():
                return
            if isinstance(worker_res, tuple):
                get_prev, get_frm, num_frames, frame_names = worker_res
                app_state['get_preview_fn'] = get_prev
                app_state['get_frame_fn'] = get_frm
                app_state['num_frames'] = num_frames
                app_state['frame_names'] = frame_names
            else:
                app_state['get_preview_fn'] = worker_res
            app_state['render_final_fn'] = render_fin
            app_state['stage'] = 'Ready'
            app_state['detail'] = f'All {len(files)} frames aligned and pre-rendered'
            app_state['progress'] = 1.0
            app_state['ready'] = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            app_state['error'] = str(e)
            app_state['stage'] = 'Processing Error'
            app_state['detail'] = str(e)

    threading.Thread(target=_run, daemon=True).start()

class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

class FocusStackRequestHandler(BaseHTTPRequestHandler):
    app_state = {}

    def log_message(self, format, *args):
        import sys
        sys.stderr.write("[HTTP] " + (format % args) + "\n")

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/" or parsed.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                if self.app_state.get('dropzone'):
                    html = HTML_TEMPLATE.replace('id="init-loader"', 'id="init-loader" class="hidden"')
                    html = html.replace('id="viewport-loader"', 'id="viewport-loader" style="display:none;"')
                    html = html.replace('id="img-info">Aligning frames...', 'id="img-info">Waiting for photos...')
                else:
                    html = HTML_TEMPLATE.replace('id="drop-zone"', 'id="drop-zone" class="hidden"')

                if self.app_state.get('standalone', True):
                    html = html.replace('⚡ Render & Import to Lightroom', '💾 Render & Save As...')
                    html = html.replace('This window will automatically close when the image imports into Lightroom.', 'Processing 16-bit stack with float32 precision. Output will be saved to your chosen file location.')

                self.wfile.write(html.encode("utf-8"))
            elif parsed.path == "/api/status":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                res = json.dumps({
                    "ready": self.app_state.get('ready', False),
                    "dropzone": self.app_state.get('dropzone', False),
                    "standalone": self.app_state.get('standalone', True),
                    "stage": self.app_state.get('stage', 'Initializing'),
                    "detail": self.app_state.get('detail', ''),
                    "progress": self.app_state.get('progress', 0.0),
                    "error": self.app_state.get('error', None)
                })
                self.wfile.write(res.encode("utf-8"))
            elif parsed.path == "/api/preview":
                if not self.app_state.get('ready') or self.app_state.get('get_preview_fn') is None:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Engine still initializing"}).encode("utf-8"))
                    return

                query = urllib.parse.parse_qs(parsed.query)
                method = query.get("method", ["egmsd"])[0]
                z_jump = float(query.get("z_jump", [5.0])[0])
                halo_gamma = float(query.get("halo_gamma", [0.45])[0])
                anisotropy = float(query.get("anisotropy", [0.50])[0])
                specular_thresh = float(query.get("specular_thresh", [0.94])[0])
                threshold = float(query.get("threshold", [1.0])[0])
                bg_mode = query.get("bg_mode", ["auto"])[0]
                stack_order = query.get("stack_order", ["near_to_far"])[0]
                mode = query.get("mode", ["fit"])[0]
                cx = float(query.get("cx", [0.5])[0])
                cy = float(query.get("cy", [0.5])[0])
                vw = int(query.get("vw", [1600])[0])
                vh = int(query.get("vh", [1000])[0])

                res_tuple = self.app_state['get_preview_fn'](
                    method=method,
                    z_jump=z_jump,
                    halo_gamma=halo_gamma,
                    anisotropy=anisotropy,
                    specular_thresh=specular_thresh,
                    noise_threshold=threshold,
                    bg_mode=bg_mode,
                    stack_order=stack_order,
                    mode=mode,
                    cx=cx,
                    cy=cy,
                    vw=vw,
                    vh=vh
                )
                eff_bg = bg_mode
                if len(res_tuple) == 6:
                    img_b64, depth_b64, saliency_b64, info_str, crop_rect, eff_bg = res_tuple
                elif len(res_tuple) == 5:
                    img_b64, depth_b64, saliency_b64, info_str, crop_rect = res_tuple
                elif len(res_tuple) == 4:
                    img_b64, depth_b64, info_str, crop_rect = res_tuple
                    saliency_b64 = depth_b64
                else:
                    img_b64, info_str, crop_rect = res_tuple
                    depth_b64 = img_b64
                    saliency_b64 = img_b64

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                res = json.dumps({
                    "image": "data:image/jpeg;base64," + img_b64,
                    "depth": "data:image/jpeg;base64," + depth_b64,
                    "saliency": "data:image/jpeg;base64," + saliency_b64,
                    "info": info_str,
                    "crop_rect": crop_rect,
                    "cx": cx,
                    "cy": cy,
                    "mode": mode,
                    "num_frames": self.app_state.get('num_frames', 1),
                    "frame_names": self.app_state.get('frame_names', []),
                    "effective_bg_mode": eff_bg
                })
                self.wfile.write(res.encode("utf-8"))
            elif parsed.path == "/api/frame":
                if not self.app_state.get('ready') or self.app_state.get('get_frame_fn') is None:
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Engine still initializing"}).encode("utf-8"))
                    return

                query = urllib.parse.parse_qs(parsed.query)
                idx = int(query.get("index", [0])[0])
                mode = query.get("mode", ["fit"])[0]
                cx = float(query.get("cx", [0.5])[0])
                cy = float(query.get("cy", [0.5])[0])
                vw = int(query.get("vw", [1600])[0])
                vh = int(query.get("vh", [1000])[0])
                peaking = query.get("peaking", ["false"])[0] in ("true", "1")

                frame_b64 = self.app_state['get_frame_fn'](idx, mode, cx, cy, peaking=peaking, vw=vw, vh=vh)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.end_headers()
                res = json.dumps({
                    "frame": "data:image/jpeg;base64," + frame_b64,
                    "index": idx
                })
                self.wfile.write(res.encode("utf-8"))
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_POST(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/api/render":
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length)
                params = json.loads(body.decode('utf-8'))
                self.app_state['render_params'] = params

                method = params.get('method', 'egmsd')
                z_jump = float(params.get('z_jump', 5.0))
                halo_gamma = float(params.get('halo_gamma', 0.45))
                anisotropy = float(params.get('anisotropy', 0.50))
                specular_thresh = float(params.get('specular_thresh', 0.94))
                threshold = float(params.get('threshold', 1.0))
                bg_mode = params.get('bg_mode', 'auto')
                stack_order = params.get('stack_order', 'near_to_far')
                overrides = params.get('overrides', None)
                custom_output_path = params.get('output_path', None)

                def _progress_cb(*args, **kwargs):
                    stage = args[0] if len(args) > 0 else kwargs.get('stage', 'Rendering')
                    if len(args) >= 3:
                        if isinstance(args[1], (int, float)):
                            prog = float(args[1])
                            detail = str(args[2])
                        else:
                            detail = str(args[1])
                            prog = float(args[2])
                    elif len(args) == 2:
                        if isinstance(args[1], (int, float)):
                            prog = float(args[1])
                            detail = ""
                        else:
                            detail = str(args[1])
                            prog = 0.0
                    else:
                        detail = kwargs.get('detail', '')
                        prog = float(kwargs.get('progress', kwargs.get('prog', 0.0)))
                    self.app_state['stage'] = stage
                    self.app_state['detail'] = detail
                    self.app_state['progress'] = prog

                out_path = self.app_state['render_final_fn'](
                    method=method,
                    z_jump=z_jump,
                    halo_gamma=halo_gamma,
                    anisotropy=anisotropy,
                    specular_thresh=specular_thresh,
                    noise_threshold=threshold,
                    bg_mode=bg_mode,
                    stack_order=stack_order,
                    overrides=overrides,
                    output_path=custom_output_path,
                    update_progress=_progress_cb
                )

                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "output_path": out_path}).encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

                # If running pywebview native window, destroy window after 2.0s so user sees completion badge
                if self.app_state.get('window'):
                    def _close_webview():
                        time.sleep(2.0)
                        try:
                            self.app_state['window'].destroy()
                        except Exception:
                            pass
                    threading.Thread(target=_close_webview, daemon=True).start()

                # Delay server shutdown by 2.0s so client finishes receiving HTTP 200 response cleanly
                def _delayed_shutdown():
                    time.sleep(2.0)
                    try:
                        self.app_state['server'].shutdown()
                    except Exception:
                        pass
                threading.Thread(target=_delayed_shutdown, daemon=True).start()
            elif parsed.path == "/api/save_dialog":
                window = self.app_state.get('window')
                default_p = self.app_state.get('output_path', 'photo_stacked.tif')
                chosen_path = choose_save_dialog(window=window, default_path=default_p)
                if not chosen_path:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "cancelled": True}).encode("utf-8"))
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": True, "save_path": chosen_path}).encode("utf-8"))
            elif parsed.path == "/api/load_files":
                length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(length) if length > 0 else b'{}'
                params = json.loads(body.decode('utf-8'))
                raw_paths = params.get('paths', [])

                if not raw_paths:
                    try:
                        from webview.dom import _dnd_state
                        for _ in range(30):
                            if _dnd_state.get('paths'):
                                raw_paths = [item[1] for item in _dnd_state['paths']]
                                _dnd_state['paths'].clear()
                                break
                            time.sleep(0.04)
                    except Exception:
                        pass

                from cli import discover_input_files
                files = discover_input_files(raw_paths)
                if len(files) < 2:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": f"Found {len(files)} supported photo(s). Need at least 2 bracketed photos to stack."
                    }).encode("utf-8"))
                    return

                try:
                    _start_stacking_worker(self.app_state, files)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "count": len(files)}).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

            elif parsed.path == "/api/browse_files":
                query = urllib.parse.parse_qs(parsed.query)
                dlg_type = query.get("type", ["folder"])[0]

                window = self.app_state.get('window')
                chosen_paths = []
                if dlg_type == "folder":
                    chosen_paths = choose_folder_dialog(window)
                else:
                    chosen_paths = choose_files_dialog(window)

                if not chosen_paths:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "cancelled": True}).encode("utf-8"))
                    return

                from cli import discover_input_files
                files = discover_input_files(chosen_paths)
                if len(files) < 2:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({
                        "success": False,
                        "error": f"Found {len(files)} supported photo(s). Need at least 2 bracketed photos to stack."
                    }).encode("utf-8"))
                    return

                try:
                    _start_stacking_worker(self.app_state, files)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "count": len(files)}).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))

            elif parsed.path == "/api/cancel":
                can_return = self.app_state.get('was_dropzone', False) and not self.app_state.get('ready', False)
                if can_return:
                    self.app_state['cancelled'] = False
                    self.app_state['dropzone'] = True
                    self.app_state['stage'] = 'Waiting for Photos'
                    self.app_state['detail'] = 'Drop bracketed photos or select a folder'
                    self.app_state['progress'] = 0.0
                    self.app_state['error'] = None
                    try:
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "cancelled": True, "can_return_to_dropzone": True}).encode("utf-8"))
                    except Exception:
                        pass
                    return

                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "cancelled": True, "can_return_to_dropzone": False}).encode("utf-8"))
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                self.app_state['cancelled'] = True

                if self.app_state.get('window'):
                    def _close_webview_cancel():
                        time.sleep(0.3)
                        try:
                            self.app_state['window'].destroy()
                        except Exception:
                            pass
                    threading.Thread(target=_close_webview_cancel, daemon=True).start()

                threading.Thread(target=self.app_state['server'].shutdown).start()
            else:
                self.send_response(404)
                self.end_headers()
        except Exception as e:
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
            except Exception:
                pass

def run_interactive_server(worker_or_get_preview=None, render_final_fn=None, start_port=8765, is_dropzone=False, on_load_files_callback=None, is_lightroom=False, default_output_path=None):
    server_state = {
        'dropzone': is_dropzone,
        'was_dropzone': is_dropzone,
        'standalone': not is_lightroom,
        'on_load_files_callback': on_load_files_callback,
        'ready': False,
        'stage': 'Waiting for Photos' if is_dropzone else 'Initializing Stacking Engine',
        'detail': 'Drop bracketed photos or select a folder' if is_dropzone else 'Preparing alignment...',
        'progress': 0.0,
        'error': None,
        'get_preview_fn': None,
        'get_frame_fn': None,
        'num_frames': 1,
        'frame_names': [],
        'render_final_fn': None,
        'render_params': None,
        'output_path': default_output_path,
        'cancelled': False,
        'window': None,
        'server': None
    }

    if is_dropzone:
        try:
            from webview.dom import _dnd_state
            _dnd_state['num_listeners'] = 1
            _dnd_state['paths'].clear()
        except Exception:
            pass

    if render_final_fn is not None:
        server_state['get_preview_fn'] = worker_or_get_preview
        server_state['render_final_fn'] = render_final_fn
        server_state['ready'] = True
        server_state['progress'] = 1.0
    elif worker_or_get_preview is not None:
        worker_fn = worker_or_get_preview
        def _run_worker():
            def update_status(stage, detail, progress):
                server_state['stage'] = str(stage)
                server_state['detail'] = str(detail)
                server_state['progress'] = max(0.0, min(1.0, float(progress)))

            def is_cancelled():
                return server_state.get('cancelled', False)

            try:
                worker_res, render_fin = worker_fn(update_status, is_cancelled)
                if is_cancelled():
                    return
                if isinstance(worker_res, tuple):
                    get_prev, get_frm, num_frames, frame_names = worker_res
                    server_state['get_preview_fn'] = get_prev
                    server_state['get_frame_fn'] = get_frm
                    server_state['num_frames'] = num_frames
                    server_state['frame_names'] = frame_names
                else:
                    server_state['get_preview_fn'] = worker_res
                server_state['render_final_fn'] = render_fin
                server_state['stage'] = 'Ready'
                server_state['detail'] = 'All frames aligned and pre-rendered'
                server_state['progress'] = 1.0
                server_state['ready'] = True
            except Exception as e:
                import traceback
                traceback.print_exc()
                server_state['error'] = str(e)
                server_state['stage'] = 'Processing Error'
                server_state['detail'] = str(e)

        threading.Thread(target=_run_worker, daemon=True).start()

    # Signal any lingering orphaned FocusStack server from a previous run to shut down cleanly
    for p in range(start_port, start_port + 5):
        try:
            conn = http.client.HTTPConnection('127.0.0.1', p, timeout=0.25)
            conn.request("POST", "/api/cancel", body=b"{}", headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            conn.close()
            time.sleep(0.15)
        except Exception:
            pass

    httpd = None
    port = start_port
    for p in range(start_port, start_port + 10):
        try:
            httpd = ReusableHTTPServer(('127.0.0.1', p), FocusStackRequestHandler)
            port = p
            break
        except OSError:
            continue

    if httpd is None:
        raise RuntimeError(f"Could not bind to any port in range {start_port}-{start_port+9}")

    server_state['server'] = httpd
    FocusStackRequestHandler.app_state = server_state

    url = f"http://127.0.0.1:{port}"
    print(f"Opening interactive UI at: {url}")

    # Check if pywebview is available for native Cocoa desktop window
    has_webview = False
    try:
        import webview
        # webview requires main thread on macOS Cocoa
        has_webview = (threading.current_thread() is threading.main_thread())
    except ImportError:
        has_webview = False

    if has_webview:
        print("Launching native macOS desktop window via pywebview...")
        if sys.platform == "darwin":
            try:
                from webview.platforms.cocoa import BrowserView
                import AppKit

                def custom_draggingEntered_(self, sender):
                    return AppKit.NSDragOperationCopy

                def custom_draggingUpdated_(self, sender):
                    return AppKit.NSDragOperationCopy

                def custom_prepareForDragOperation_(self, sender):
                    return True

                def custom_performDragOperation_(self, sender):
                    try:
                        pboard = sender.draggingPasteboard()
                        paths = []
                        fn_plist = pboard.propertyListForType_(AppKit.NSFilenamesPboardType)
                        if fn_plist:
                            for p in fn_plist:
                                if os.path.exists(str(p)):
                                    paths.append(str(p))
                        if not paths:
                            classes = [AppKit.NSURL]
                            options = {AppKit.NSPasteboardURLReadingFileURLsOnlyKey: AppKit.NSNumber.numberWithBool_(True)}
                            urls = pboard.readObjectsForClasses_options_(classes, options) or []
                            for url_obj in urls:
                                fp = urllib.parse.unquote(url_obj.filePathURL().absoluteString().replace('file://', ''))
                                if os.path.exists(fp):
                                    paths.append(fp)

                        print(f"[DragDrop] Cocoa dropped paths received: {paths}")
                        if paths:
                            from cli import discover_input_files
                            files = discover_input_files(paths)
                            print(f"[DragDrop] Discovered {len(files)} files to stack")
                            if len(files) >= 2:
                                _start_stacking_worker(server_state, files)
                                js = "const dz=document.getElementById('drop-zone'); if(dz) dz.classList.add('hidden'); const ld=document.getElementById('init-loader'); if(ld) ld.classList.remove('hidden'); if (typeof initPollTimer === 'undefined' || !initPollTimer) initPollTimer = setInterval(checkInitStatus, 180); checkInitStatus();"
                                self.evaluateJavaScript_completionHandler_(js, None)
                                return True
                            else:
                                err_msg = f"Found {len(files)} supported photo(s). Need at least 2 bracketed photos to stack."
                                js = f"showDropError({json.dumps(err_msg)});"
                                self.evaluateJavaScript_completionHandler_(js, None)
                                return True
                    except Exception as err:
                        import traceback
                        traceback.print_exc()
                        print(f"Notice: Cocoa drag drop handling error: {err}")
                    return True

                BrowserView.WebKitHost.draggingEntered_ = custom_draggingEntered_
                BrowserView.WebKitHost.draggingUpdated_ = custom_draggingUpdated_
                BrowserView.WebKitHost.prepareForDragOperation_ = custom_prepareForDragOperation_
                BrowserView.WebKitHost.performDragOperation_ = custom_performDragOperation_
            except Exception as e:
                print(f"Notice: Cocoa drag-and-drop hook skipped ({e})")

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        server_thread.start()

        def on_closed():
            if not server_state.get('render_params'):
                server_state['cancelled'] = True
            try:
                httpd.shutdown()
            except Exception:
                pass

        try:
            window = webview.create_window(
                title="Open Focus Stack Pro",
                url=url,
                width=1340,
                height=880,
                min_size=(960, 640)
            )
            server_state['window'] = window
            window.events.closed += on_closed
            webview.start()
        except Exception as e:
            print(f"Notice: pywebview window failed ({e}), falling back...")
        finally:
            try:
                httpd.shutdown()
            except Exception:
                pass
            try:
                httpd.server_close()
            except Exception:
                pass
        return server_state.get('cancelled', False)

    def _launch_browser():
        if os.environ.get("FOCUS_STACK_HEADLESS") == "1":
            return
        use_app_mode = os.environ.get("FOCUS_STACK_APP_MODE", "0") == "1"
        if use_app_mode and sys.platform == "darwin":
            for browser in ["Google Chrome", "Microsoft Edge", "Brave Browser"]:
                app_path = f"/Applications/{browser}.app"
                if os.path.exists(app_path):
                    try:
                        ret = subprocess.call(
                            ["/usr/bin/open", "-na", browser, "--args", f"--app={url}", "--window-size=1340,880"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                        )
                        if ret == 0:
                            print(f"Launched native application window via {browser} App Mode.")
                            return
                    except Exception:
                        pass

        opened = False
        if sys.platform == "darwin":
            try:
                ret = subprocess.call(["/usr/bin/open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if ret == 0:
                    opened = True
            except Exception:
                pass
        if not opened:
            try:
                webbrowser.open(url)
            except Exception as e:
                print(f"Notice: Could not launch browser automatically ({e}). Please visit {url}")

    threading.Thread(target=_launch_browser, daemon=True).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        server_state['cancelled'] = True
    finally:
        httpd.server_close()
    return server_state.get('cancelled', False)
