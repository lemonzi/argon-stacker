#!/usr/bin/env python3
"""
Unit tests for EXIF timestamp ordering, discrepancy detection, and UI enhancements.
"""
import unittest
import os
import sys
import glob
import json
import tempfile
import http.client
import threading
import time
import numpy as np
import tifffile
from PIL import Image

_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_py_pkg = os.path.join(_base_dir, "FocusStack.lrplugin", "bin", "py_focus_stack")
if _py_pkg not in sys.path:
    sys.path.insert(0, _py_pkg)

from cli import (
    natural_sort_key,
    get_photo_capture_timestamp,
    check_sort_order_discrepancy
)
from server_ui import (
    HTML_TEMPLATE,
    ReusableHTTPServer,
    FocusStackRequestHandler
)


class TestOrderAndUI(unittest.TestCase):

    def test_exif_timestamp_extraction_real_images(self):
        """Test EXIF capture timestamp extraction from real JPEG bracketed stack."""
        test_images = sorted(glob.glob(os.path.join(_base_dir, "test_stack", "*.jpg")), key=natural_sort_key)
        self.assertGreaterEqual(len(test_images), 2, "Need test_stack JPEG images for test")

        ts_first = get_photo_capture_timestamp(test_images[0])
        self.assertIsNotNone(ts_first, f"Failed to extract timestamp from {test_images[0]}")
        self.assertTrue(":" in ts_first, f"Timestamp format invalid: {ts_first}")

        # Check subsecond precision
        self.assertTrue("." in ts_first, f"Expected subsecond dot in timestamp: {ts_first}")

    def test_check_sort_order_discrepancy_matching(self):
        """When filenames and EXIF timestamps are in agreement, disagree must be False."""
        test_images = sorted(glob.glob(os.path.join(_base_dir, "test_stack", "*.jpg")), key=natural_sort_key)
        self.assertGreaterEqual(len(test_images), 3)

        disagree, files_by_name, files_by_exif, details = check_sort_order_discrepancy(test_images)
        self.assertFalse(disagree, "Standard sequential test_stack should not report discrepancy")
        self.assertEqual(files_by_name, files_by_exif)

    def test_check_sort_order_discrepancy_mismatch(self):
        """When filenames and EXIF timestamps disagree, discrepancy must be detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two TIFF files: frame_01 with later timestamp, frame_02 with earlier timestamp
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            f1 = os.path.join(tmpdir, "frame_01.tif")
            f2 = os.path.join(tmpdir, "frame_02.tif")

            # Tag 36867 = DateTimeOriginal
            tifffile.imwrite(f1, arr, extratags=[(36867, 's', 1, '2026:10:11 12:00:10', False)])
            tifffile.imwrite(f2, arr, extratags=[(36867, 's', 1, '2026:10:11 12:00:02', False)])

            disagree, files_by_name, files_by_exif, details = check_sort_order_discrepancy([f1, f2])
            self.assertTrue(disagree, "Expected discrepancy when filename and EXIF disagree")
            self.assertEqual(files_by_name, [f1, f2])
            self.assertEqual(files_by_exif, [f2, f1])
            self.assertTrue(details.get("disagree"))
            self.assertEqual(len(details["files_by_name"]), 2)
            self.assertEqual(len(details["files_by_exif"]), 2)

    def test_ui_html_structure_and_scrollable_pane(self):
        """Verify HTML template contains required elements for order modal, scrollable algo pane, and responsive resizing."""
        # 1. Advanced algorithm settings pane must be closed by default per user specification
        self.assertIn('id="grp-algo-collapsible">', HTML_TEMPLATE, "Algorithm pane element missing")
        self.assertNotIn('id="grp-algo-collapsible" open', HTML_TEMPLATE, "Algorithm pane should default to closed")

        # 2. Advanced algorithm pane content must have overflow-y: auto and max-height
        self.assertIn('.algo-collapsible-content {', HTML_TEMPLATE)
        self.assertIn('max-height: 280px;', HTML_TEMPLATE)
        self.assertIn('overflow-y: auto;', HTML_TEMPLATE)

        # 3. Order discrepancy modal markup must be present
        self.assertIn('id="order-modal"', HTML_TEMPLATE, "Order modal markup missing")
        self.assertIn('id="order-list-exif"', HTML_TEMPLATE, "Order list EXIF missing")
        self.assertIn('id="order-list-name"', HTML_TEMPLATE, "Order list Name missing")
        self.assertIn('resolveOrderChoice', HTML_TEMPLATE, "resolveOrderChoice function missing")

        # 4. Wheel listener on algorithm pane must be present
        self.assertIn('algoPaneEl.addEventListener(\'wheel\'', HTML_TEMPLATE, "Mouse wheel listener on algo pane missing")

        # 5. Image resizing logic (syncFitDimensions) must be present
        self.assertIn('function syncFitDimensions()', HTML_TEMPLATE, "syncFitDimensions function missing")
        self.assertIn('window.addEventListener(\'resize\'', HTML_TEMPLATE, "resize listener missing")

    def test_api_status_and_resolve_order(self):
        """Test FocusStackRequestHandler status and resolve_order logic without requiring socket binding."""
        import io

        state = {
            'dropzone': False,
            'standalone': True,
            'ready': False,
            'stage': 'Order Confirmation',
            'detail': 'EXIF timestamp and filename order disagree',
            'progress': 0.0,
            'error': None,
            'has_order_conflict': True,
            'order_conflict': {
                'disagree': True,
                'count': 2,
                'files_by_name': [{'basename': 'b.tif', 'timestamp': '2026:10:11 12:00:10'}],
                'files_by_exif': [{'basename': 'a.tif', 'timestamp': '2026:10:11 12:00:01'}]
            },
            'on_resolve_order': None
        }

        resolved_choice = []
        def mock_resolve(choice):
            resolved_choice.append(choice)
            return lambda update_status, is_cancelled: ((None, None, 2, ['a.tif', 'b.tif']), None)

        state['on_resolve_order'] = mock_resolve

        class MockRequest:
            def __init__(self, raw_input):
                self.rfile = io.BytesIO(raw_input)
                self.wfile = io.BytesIO()

            def makefile(self, *args, **kwargs):
                if 'r' in args[0]:
                    return self.rfile
                return self.wfile

            def sendall(self, data):
                self.wfile.write(data)

        # 1. Test GET /api/status returns has_order_conflict
        req_get = MockRequest(b"GET /api/status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
        FocusStackRequestHandler.app_state = state
        handler = FocusStackRequestHandler(req_get, ('127.0.0.1', 8765), None)
        out_bytes = req_get.wfile.getvalue()
        body = out_bytes.split(b"\r\n\r\n", 1)[1]
        data = json.loads(body.decode('utf-8'))
        self.assertTrue(data.get("has_order_conflict"))
        self.assertIsNotNone(data.get("order_conflict"))

        # 2. Test POST /api/resolve_order with choice 'exif'
        post_body = json.dumps({"order": "exif"}).encode('utf-8')
        post_req = (
            b"POST /api/resolve_order HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(post_body)}\r\n\r\n".encode('ascii')
            + post_body
        )
        req_post = MockRequest(post_req)
        FocusStackRequestHandler(req_post, ('127.0.0.1', 8765), None)
        post_out = req_post.wfile.getvalue()
        post_res_body = post_out.split(b"\r\n\r\n", 1)[1]
        res_data = json.loads(post_res_body.decode('utf-8'))
        self.assertTrue(res_data.get("success"))
        self.assertEqual(res_data.get("choice"), "exif")
        self.assertEqual(resolved_choice, ["exif"])
        self.assertFalse(state.get("has_order_conflict"))

    def test_brush_cursor_css_and_paint_pipeline(self):
        """Verify brush cursor CSS rules override mode-fit zoom-in and paint pipeline initializes cleanly."""
        # 1. Verify CSS cursor override when brush is active
        self.assertIn("#viewport.brush-active", HTML_TEMPLATE)
        self.assertIn("#viewport.brush-active #preview-img", HTML_TEMPLATE)
        self.assertIn("#viewport.brush-active #canvas-container", HTML_TEMPLATE)
        self.assertIn("cursor: none !important;", HTML_TEMPLATE)
        self.assertIn("pointer-events: auto !important;", HTML_TEMPLATE)

        # 2. Verify ensureMasterMaskInitialized is present
        self.assertIn("function ensureMasterMaskInitialized()", HTML_TEMPLATE)

        # 3. Verify fit-mode frame cache parameter key stabilization
        self.assertIn("return { key: `${frameIdx}_fit`, vw: 2400, vh: 1600 };", HTML_TEMPLATE)

        # 4. If WebKit is available in the environment, test live WKWebView DOM execution
        try:
            import objc
            from Cocoa import NSObject, NSApplication, NSRunLoop, NSDate, NSDefaultRunLoopMode
            from WebKit import WKWebView, WKWebViewConfiguration

            class NavDelegate(NSObject):
                def webView_didFinishNavigation_(self, webview, navigation):
                    self.finished = True

            app = NSApplication.sharedApplication()
            config = WKWebViewConfiguration.alloc().init()
            webview = WKWebView.alloc().initWithFrame_configuration_(((0,0),(1340,880)), config)
            nav = NavDelegate.alloc().init()
            nav.finished = False
            webview.setNavigationDelegate_(nav)

            webview.loadHTMLString_baseURL_(HTML_TEMPLATE, None)

            timeout = time.time() + 5
            while not nav.finished and time.time() < timeout:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))

            self.assertTrue(nav.finished, "WebKit view failed to finish navigation")

            test_js = '''
            (function() {
                const dz = document.getElementById('drop-zone');
                if (dz) dz.classList.add('hidden');
                const il = document.getElementById('init-loader');
                if (il) il.classList.add('hidden');

                const img = document.getElementById('preview-img');
                Object.defineProperty(img, 'naturalWidth', { value: 2400 });
                Object.defineProperty(img, 'naturalHeight', { value: 1600 });
                img.className = 'loaded mode-fit';
                img.style.display = 'block';
                
                syncFitDimensions();
                syncCanvasDimensions();

                const pImgBefore = window.getComputedStyle(img).cursor;
                toggleBrush();

                const vp = document.getElementById('viewport');
                const cCont = document.getElementById('canvas-container');
                const pImg = document.getElementById('preview-img');
                const rCanvas = document.getElementById('retouch-canvas');

                const vpCursor = window.getComputedStyle(vp).cursor;
                const cContCursor = window.getComputedStyle(cCont).cursor;
                const pImgCursor = window.getComputedStyle(pImg).cursor;
                const rCanvasCursor = window.getComputedStyle(rCanvas).cursor;

                // Test paint dispatch
                const rect = rCanvas.getBoundingClientRect();
                const cx = rect.left + rect.width / 2;
                const cy = rect.top + rect.height / 2;

                const md = new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: cx, clientY: cy });
                rCanvas.dispatchEvent(md);

                const mm = new MouseEvent('mousemove', { bubbles: true, clientX: cx + 15, clientY: cy + 15 });
                window.dispatchEvent(mm);

                const mu = new MouseEvent('mouseup', { bubbles: true });
                window.dispatchEvent(mu);

                const frameMask = frameMaskCanvases[currentFrameIdx];
                let nonZero = 0;
                if (frameMask) {
                    const d = frameMask.getContext('2d').getImageData(0, 0, frameMask.width, frameMask.height).data;
                    for (let i = 3; i < d.length; i += 4) { if (d[i] > 0) nonZero++; }
                }

                return JSON.stringify({
                    pImgBefore: pImgBefore,
                    vpCursor: vpCursor,
                    cContCursor: cContCursor,
                    pImgCursor: pImgCursor,
                    rCanvasCursor: rCanvasCursor,
                    nonZero: nonZero
                });
            })();
            '''
            res = {'val': None, 'done': False}
            def handler(val, err):
                res['val'] = val
                res['done'] = True
            webview.evaluateJavaScript_completionHandler_(test_js, handler)

            timeout = time.time() + 3
            while not res['done'] and time.time() < timeout:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))

            self.assertTrue(res['done'])
            dom_res = json.loads(res['val'])
            self.assertEqual(dom_res['pImgBefore'], 'zoom-in')
            self.assertEqual(dom_res['vpCursor'], 'none')
            self.assertEqual(dom_res['cContCursor'], 'none')
            self.assertEqual(dom_res['pImgCursor'], 'none')
            self.assertEqual(dom_res['rCanvasCursor'], 'none')
            self.assertGreater(dom_res['nonZero'], 0)
        except ImportError:
            pass  # WebKit not installed on test host

    def test_pan_drag_release_proxy_retention(self):
        """Verify pan overview proxy retention, container dimension pinning, and loading badge visibility on drag release."""
        # Static template checks
        self.assertIn("image-rendering: pixelated;", HTML_TEMPLATE)
        self.assertIn("oImg.style.filter = 'none';", HTML_TEMPLATE)
        self.assertIn("cCont.style.width = dispW + 'px';", HTML_TEMPLATE)
        self.assertIn("cCont.style.height = dispH + 'px';", HTML_TEMPLATE)

        # Dynamic WebKit evaluation
        try:
            import objc
            from Cocoa import NSObject, NSApplication, NSRunLoop, NSDate, NSDefaultRunLoopMode
            from WebKit import WKWebView, WKWebViewConfiguration

            app = NSApplication.sharedApplication()
            config = WKWebViewConfiguration.alloc().init()
            webview = WKWebView.alloc().initWithFrame_configuration_(((0,0),(1400,900)), config)

            try:
                PanNavDelegate = objc.lookUpClass('PanNavDelegate')
            except objc.nosuchclass_error:
                class PanNavDelegate(NSObject):
                    def webView_didFinishNavigation_(self, wv, nav):
                        self.finished = True
                    def webView_didFailNavigation_withError_(self, wv, nav, err):
                        self.finished = True

            nav = PanNavDelegate.alloc().init()
            nav.finished = False
            webview.setNavigationDelegate_(nav)

            webview.loadHTMLString_baseURL_(HTML_TEMPLATE, None)

            timeout = time.time() + 5
            while not nav.finished and time.time() < timeout:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))

            self.assertTrue(nav.finished, "WebKit view failed to finish navigation")

            test_js = '''
            (function() {
                const dz = document.getElementById('drop-zone');
                if (dz) dz.classList.add('hidden');
                const il = document.getElementById('init-loader');
                if (il) il.classList.add('hidden');

                const img = document.getElementById('preview-img');
                Object.defineProperty(img, 'naturalWidth', { value: 1200 });
                Object.defineProperty(img, 'naturalHeight', { value: 800 });
                img.className = 'loaded mode-roi';
                img.style.display = 'block';

                overviewImageSrc = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
                const oImg = document.getElementById('pan-overview-img');
                oImg.src = overviewImageSrc;
                oImg.style.display = 'block';

                currentViewMode = '100%';
                currentCropRect = [0.2, 0.2, 0.4, 0.4];
                syncPanOverview();

                const cCont = document.getElementById('canvas-container');
                const pImg = document.getElementById('preview-img');

                // Simulate drag pan
                const md = new MouseEvent('mousedown', { bubbles: true, button: 0, clientX: 300, clientY: 300 });
                pImg.dispatchEvent(md);

                const mm = new MouseEvent('mousemove', { bubbles: true, clientX: 350, clientY: 360 });
                window.dispatchEvent(mm);

                const mu = new MouseEvent('mouseup', { bubbles: true, clientX: 350, clientY: 360 });
                window.dispatchEvent(mu);

                const badge = document.getElementById('loading-badge');

                return JSON.stringify({
                    cContWidth: cCont.offsetWidth,
                    cContHeight: cCont.offsetHeight,
                    oImgDisplay: window.getComputedStyle(oImg).display,
                    oImgOpacity: window.getComputedStyle(oImg).opacity,
                    oImgFilter: oImg.style.filter,
                    oImgTransform: oImg.style.transform,
                    badgeDisplay: window.getComputedStyle(badge).display,
                    pImgDisplay: pImg.style.display
                });
            })();
            '''
            res = {'val': None, 'done': False}
            def handler(val, err):
                res['val'] = val
                res['done'] = True
            webview.evaluateJavaScript_completionHandler_(test_js, handler)

            timeout = time.time() + 3
            while not res['done'] and time.time() < timeout:
                NSRunLoop.currentRunLoop().runMode_beforeDate_(NSDefaultRunLoopMode, NSDate.dateWithTimeIntervalSinceNow_(0.05))

            self.assertTrue(res['done'])
            dom_res = json.loads(res['val'])
            self.assertGreater(dom_res['cContWidth'], 0, "Canvas container must not collapse to 0 width")
            self.assertGreater(dom_res['cContHeight'], 0, "Canvas container must not collapse to 0 height")
            self.assertEqual(dom_res['oImgDisplay'], 'block')
            self.assertEqual(dom_res['oImgOpacity'], '1')
            self.assertEqual(dom_res['oImgFilter'], 'none')
            self.assertEqual(dom_res['oImgTransform'], 'translate(50px, 60px)')
            self.assertEqual(dom_res['badgeDisplay'], 'block')
            self.assertEqual(dom_res['pImgDisplay'], 'none')
        except ImportError:
            pass

    def test_window_close_confirmation_and_beforeunload(self):
        """Verify server_ui includes confirm_close and beforeunload protection on window close."""
        server_ui_path = os.path.join(_py_pkg, "server_ui.py")
        with open(server_ui_path, "r", encoding="utf-8") as f:
            code = f.read()

        self.assertIn("confirm_close=True", code)
        self.assertIn("global.quitConfirmation", code)
        self.assertIn("Cancel focus stacking? Temporary files will be cleaned up.", code)
        self.assertIn("beforeunload", code)
        self.assertIn("isClosingOrFinished", code)


if __name__ == "__main__":
    unittest.main()
