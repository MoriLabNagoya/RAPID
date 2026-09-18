# -*- coding: utf-8 -*-
import os
import json
import tkinter as tk

from tkinter import ttk
from PIL import Image,ImageTk,ImageDraw
from tkinter.filedialog import askopenfilename
from tkinter import messagebox, filedialog
from ARMP.gui_imageframe import ImageFrame
from ARMP.logic_config import Config
from ARMP.logic_logger import logging, handle_exception
import cv2
import numpy as np
import math
import time
import re
import traceback
from pathlib import Path
from skimage import morphology
#import glob
from ARMP.AnalyseRoot import AnalyseRoot as ar
import csv
from ARMP import resultWindow
import shutil
from ARMP.scrolledtreeview import ScrolledTreeview
from ultralytics import YOLO
import torch
from ARMP.gui_imageframe import ROIs,COLORS,EachRootData
from ARMP.mask_editor import CLASS_COLORS

import threading
#import datetime
import queue
import glob
import xml.etree.ElementTree as ET
Image.MAX_IMAGE_PIXELS = None
from ARMP.AnalyseRoot import ROOTTYPES, RootProp

# Main-view mask rendering.
# Transparency = 0 means the label itself is fully opaque, matching mask_editor.py.
# Pixels outside the mask are never overlaid, so the background remains unchanged.
MAIN_LABEL_TRANSPARENCY = 0.0
import os
import glob
import json

class MyScrolledTreeview(ScrolledTreeview):
    def __init__(self, master=None,head = None,height = None, data = None, **kw):
        ScrolledTreeview.__init__(
            self,
            master,
            columns=["DATA"],
            height=height,
            **kw)
        # 列定義
        self.column("DATA", width=100, minwidth=50)
        #self.column("COL_A", width=100, minwidth=100)
        #self.column("COL_B", width=100, minwidth=100)
        #self.column("COL_C", width=100, minwidth=100)
        # 見出し定義
        for i, txt in enumerate(head):
            self.heading(f"#{i}", text=txt)
        #for one in data:
        #    self.insert(parent='',index=0, values=os.path.basename(one))
        #self.heading("#1", text=head[1])
        if data is not None:
            for i,one in enumerate(data):
                #self.insert(parent='',index=0, values=os.path.basename(one))#self.file_name = os.path.splitext(os.path.basename(self.imgpath))[0]
                self.insert('',tk.END,text = "{:2d}".format(i+1), values=os.path.splitext(os.path.basename(one))[0])#values=os.path.basename(one)


class _RAPIDToolTip:
    def __init__(self, widget, text_getter, delay=450):
        self.widget, self.text_getter, self.delay = widget, text_getter, delay
        self.after_id = None; self.tip = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
    def _enter(self, event=None):
        self._hide()
        try: self.after_id=self.widget.after(self.delay, self._show)
        except tk.TclError: pass
    def _show(self):
        self.after_id=None
        try:
            s=str(self.text_getter() or "").strip()
            if not s: return
            x=self.widget.winfo_rootx()+14; y=self.widget.winfo_rooty()+self.widget.winfo_height()+6
            self.tip=tk.Toplevel(self.widget); self.tip.wm_overrideredirect(True)
            self.tip.wm_geometry(f"+{x}+{y}")
            tk.Label(self.tip,text=s,justify="left",relief="solid",borderwidth=1,
                     padx=7,pady=4,background="#ffffe0",wraplength=420).pack()
        except tk.TclError: self.tip=None
    def _hide(self, event=None):
        if self.after_id is not None:
            try: self.widget.after_cancel(self.after_id)
            except tk.TclError: pass
            self.after_id=None
        if self.tip is not None:
            try: self.tip.destroy()
            except tk.TclError: pass
            self.tip=None

_RAPID_JA={
"File":"ファイル","View":"表示","Tool":"ツール","Language":"言語","Help":"ヘルプ","About":"RAPIDについて",
"Create project":"プロジェクトを作成","Batch Processing":"バッチ処理","Open project":"プロジェクトを開く",
"Open recent":"最近使った項目","Close project":"プロジェクトを閉じる","Exit":"終了",
"Batch export dataset...":"バッチデータセットをエクスポート...","Fullscreen":"全画面表示","Default size":"既定サイズ",
"Scale calibration...":"スケール校正...","Detect ruler scale...":"定規スケールを検出...",
"AI Select":"AI 選択","AI Process":"AI 処理","Open root annotation editor":"根アノテーションエディターを開く",
"Standalone annotation editor...":"単独アノテーションエディター...",
"Project folder":"プロジェクトフォルダー","Image folder":"画像フォルダー",
"Project Folder":"プロジェクトフォルダーを選択","Image Folder":"画像フォルダーを選択",
"Select folder":"フォルダーを選択","Select image":"画像を選択","Image file":"画像ファイル",
"OK":"OK","Cancel":"キャンセル","Batch processing":"バッチ処理","Batch settings":"バッチ設定",
"Device:":"デバイス:","Segmentation:":"セグメンテーション:","Scale mode:":"スケールモード:",
"Ruler interval:":"定規間隔:","Min confidence:":"最小信頼度:","Skip already processed":"処理済み画像をスキップ",
"Continue if an image fails":"画像処理に失敗しても続行","Set shared scale":"共通スケールを設定",
"Preview ruler":"定規をプレビュー","Start":"開始","Pause":"一時停止","Resume":"再開","Export":"エクスポート",
"Progress":"進捗","Ready":"準備完了","Index":"番号","Image-names":"画像名","Auto":"自動",
"Same scale for all images":"全画像で同じスケール","Detect ruler for every image":"各画像で定規を検出","Pixel only":"ピクセルのみ"}
_RAPID_EN={v:k for k,v in _RAPID_JA.items()}

_RAPID_JA.update({
"Batch export":"バッチエクスポート","Project root":"プロジェクトルート","Browse":"参照",
"Output folder":"出力フォルダー","Export scope":"エクスポート範囲","Each ROI":"各 ROI","Full image":"画像全体",
"Outputs":"出力内容","Original image":"元画像","Colored label preview":"カラーラベルプレビュー",
"Binary root mask":"二値ルートマスク","Label mask (0=BG, 1=Leaf, 2=Primary, 3+=Lateral)":"ラベルマスク (0=背景, 1=葉, 2=主根, 3+=側根)",
"Project folder":"プロジェクトフォルダー","Image file":"画像ファイル","Scale calibration":"スケール校正",
"Known physical distance:":"既知の実距離:","Pixel distance:":"ピクセル距離:","Unit of length:":"長さの単位:",
"Verify detected ruler scale":"検出した定規スケールを確認","Detection candidate:":"検出候補:",
"Physical distance per minor tick:":"小目盛りあたりの実距離:","Unit:":"単位:",
"Apply verified scale":"確認したスケールを適用","Open image":"画像を開く","Open mask":"マスクを開く",
"Load GT mask":"GT マスクを読み込む","Save mask":"マスクを保存","Save color labels":"カラーラベルを保存",
"Clear mask":"マスクを消去","Fit":"ウィンドウに合わせる","Apply to project":"プロジェクトに適用",
"Current brush label":"現在のブラシラベル","New lateral  [N]":"新しい側根  [N]","Delete":"削除",
"Root / Leaf summary":"根 / 葉の概要","Drawing tool":"描画ツール","Brush [B]":"ブラシ [B]","Line [L]":"線 [L]",
"Fill [F]":"塗りつぶし [F]","Picker [I]":"スポイト [I]","◀ Previous ROI [A]":"◀ 前の ROI [A]",
"Next ROI [D] ▶":"次の ROI [D] ▶","Brush width (image px)":"ブラシ幅（画像 px）","Display":"表示",
"Image":"画像","Labels":"ラベル","Difference":"差分","Skeleton":"スケルトン","Label view:":"ラベル表示:",
"All":"すべて","One label":"1 ラベル","History":"履歴","Undo":"元に戻す","Redo":"やり直す",
"Persist history locally":"履歴をローカル保存","History is capped at 40 steps.":"履歴は最大 40 ステップです。",
"Save validation CSV":"検証 CSV を保存"
})
_RAPID_JA.update({
    "Language":"言語",
    "Batch Processing":"バッチ処理",
    "Current shared scale":"現在の共通スケール",
    "Done":"完了", "Failed":"失敗", "Review":"要確認", "Skipped":"スキップ",
})
_RAPID_EN={v:k for k,v in _RAPID_JA.items()}
_RAPID_EN["バッチ処理"]="Batch Processing"

_RAPID_TIPS_EN={
"Project Folder":"Choose the output folder for batch projects.","Image Folder":"Choose the folder containing images to process.",
"Select folder":"Choose a folder.","Select image":"Choose an image file.",
"Device:":"Choose Auto, GPU, or CPU for model inference.","Segmentation:":"Choose the organ-segmentation method.",
"Scale mode:":"Choose shared scale, per-image ruler detection, or pixel-only output.",
"Ruler interval:":"Physical distance represented by adjacent ruler marks.","Min confidence:":"Minimum confidence for automatic ruler-scale detection.",
"Set shared scale":"Set one physical scale for all batch images.","Preview ruler":"Preview automatic ruler detection.",
"Start":"Start batch processing.","Pause":"Temporarily pause batch processing.","Cancel":"Cancel or close the current operation.",
"Export":"Export batch results.","Skip already processed":"Skip images whose project results already exist.",
"Continue if an image fails":"Continue the batch when one image fails."}
_RAPID_TIPS_JA={
"Project Folder":"バッチ処理のプロジェクト保存先を選択します。","Image Folder":"処理する画像が入ったフォルダーを選択します。",
"Select folder":"フォルダーを選択します。","Select image":"画像ファイルを選択します。",
"Device:":"推論に使用する Auto / GPU / CPU を選択します。","Segmentation:":"器官セグメンテーション法を選択します。",
"Scale mode:":"共通スケール、画像ごとの定規検出、またはピクセルのみを選択します。",
"Ruler interval:":"定規の隣接目盛り間の実際の距離です。","Min confidence:":"自動スケール検出で採用する最小信頼度です。",
"Set shared scale":"バッチ全画像に共通するスケールを設定します。","Preview ruler":"定規の自動検出結果を確認します。",
"Start":"バッチ処理を開始します。","Pause":"バッチ処理を一時停止します。","Cancel":"操作をキャンセルまたは閉じます。",
"Export":"バッチ処理結果をエクスポートします。","Skip already processed":"処理済み画像をスキップします。",
"Continue if an image fails":"1 枚の処理に失敗してもバッチ処理を続行します。"}

class MainGUI(ttk.Frame):
    """ GUI of Image Viewer """
    
    def __init__(self, mainframe):
        """ Initialize the Frame """
        logging.info('Open GUI')
        ttk.Frame.__init__(self, master=mainframe)
        self.__create_instances()
        self.__create_main_window()
        self.__create_widgets()

    def __create_instances(self):
        """ Instances for GUI are created here """
        self.__config = Config()  # open config file of the main window
        self.__imframe = None  # empty instance of image frame (canvas)
        self.modelSeg = None
        self.yoloModel = None
        self._model_lock = threading.Lock()
        self.yolo_device = "auto"
        self.batch_thread = None
        self.batch_cancel_event = threading.Event()
        self.batch_pause_event = threading.Event()
        self.batch_queue = queue.Queue()
        self.ui_language='en'
        self._rapid_tooltips=[]
        self.InitialVariables()
    def __create_main_window(self):
        """ Create main window GUI"""
        self.__default_title = 'RAPID'
        self.master.title(self.__default_title)
        self.master.geometry(self.__config.get_win_geometry())  # get window size/position from config
        self._safe_apply_window_state(self.master, self.__config.get_win_state())
        self._set_window_icon(self.master)
        # self.destructor gets fired when the window is destroyed
        self.master.protocol('WM_DELETE_WINDOW', self.destroys)
        #
        self.__menubar = tk.Menu(self.master)  # create main menu bar
        self.master.configure(menu=self.__menubar)  # should be BEFORE iconbitmap, it's important
        # Add menubar to the main window BEFORE iconbitmap command. Otherwise it will shrink
        # in height by 20 pixels after each opening of the window.
        
        #
        self.__is_fullscreen = False  # enable / disable fullscreen mode
        self.__empty_menu = tk.Menu(self)  # empty menu to hide the real menubar in fullscreen mode
        self.__bugfix = False  # BUG when change: fullscreen --> zoomed --> normal
        self.__previous_state = 0  # previous state of the event
        # List of shortcuts in the following format: [name, keycode, function]
        # Use Tk keysyms instead of platform-specific numeric keycodes.
        # Numeric keycodes differ between Windows/X11/Wayland, while keysyms are portable.
        self.__shortcuts = [['Ctrl+N', 'n', self.__Create_Project],   # 1 open image
                            #['Ctrl+D', 'd', self.__open_folder],
                            ['Ctrl+W', 'w', self.__close_image],
                            ['Ctrl+S', 's', self.__save_Result],
                            ['Ctrl+O', 'o', self.__Open]]  # 2 close image
        
        
        # Bind events to the main window
        self.master.bind('<Motion>', self.__motion)  # track and handle mouse pointer position
        # Inspect the original-image pixel at the left mouse position.
        self.master.bind(
            "<ButtonPress-1>",
            self._show_pixel_at_pointer,
            add="+"
        )
        
        # Continue updating while the left mouse button is held and dragged.
        self.master.bind(
            "<B1-Motion>",
            self._show_pixel_at_pointer,
            add="+"
        )
        self.master.bind('<F11>', self.__fullscreen_toggle)  # toggle fullscreen mode
        self.master.bind('<Escape>', lambda e=None, s=False: self.__fullscreen_toggle(e, s))
        self.master.bind('<F4>', self.__SetSelectMode)  # 
        self.master.bind('<F5>', self.__default_geometry)  # reset default window geometry
        
        self.master.bind('<F6>', self.__SetResolution)  # reset default window geometry
        #self.master.bind('<F8>', self.__Threshold_toggle)  # reset default window geometry
        self.master.bind('<F9>', self.__ICML_NoAI_toggle)  # YOLO ROI + legacy ICML segmentation
        self.master.bind('<F10>', self.__ICML_Improve_toggle)  # YOLO ROI + conservative ICML improvements
        
        # Handle main window resizing in the idle mode, because consecutive keystrokes <F11> - <F5>
        # don't set default geometry from full screen if resizing is not postponed.
        self.master.bind('<Configure>', lambda event: self.master.after_idle(
            self.__resize_master, event))  # window is resized
        # Handle keystrokes in the idle mode, because program slows down on a weak computers,
        # when too many key stroke events in the same time.
        self.master.bind('<Key>', lambda event: self.master.after_idle(self.__keystroke, event))
        self.master.bind('<Double-Button-1>', self.__DoubleClick)  # zoom for Linux, wheel scroll up
    def __DoubleClick(self,event):
        """Open the annotation editor only when the double-click hits a processed ROI."""
        matched = -1
        if len(self.RootAll) > 0 and self.__imframe is not None:
            for i, eachroot in enumerate(self.RootAll):
                if eachroot.bProcessed is not True:
                    continue
                if len(eachroot.ROICoordinate) == 2:
                    lt = (eachroot.ROICoordinate[0][0], eachroot.ROICoordinate[0][1])
                    rt = (eachroot.ROICoordinate[1][0], eachroot.ROICoordinate[0][1])
                    rb = (eachroot.ROICoordinate[1][0], eachroot.ROICoordinate[1][1])
                    lb = (eachroot.ROICoordinate[0][0], eachroot.ROICoordinate[1][1])
                    polygon = np.array([lt, rt, rb, lb], np.int64)
                else:
                    polygon = np.array(eachroot.ROICoordinate, np.int64)
                x = self.__imframe.canvas.canvasx(event.x)
                y = self.__imframe.canvas.canvasy(event.y)
                bbox1 = self.__imframe.canvas.coords(self.__imframe.container)
                xx = (x - bbox1[0]) / self.__imframe.imscale
                yy = (y - bbox1[1]) / self.__imframe.imscale
                if cv2.pointPolygonTest(polygon, (xx, yy), False) >= 0:
                    matched = i
                    break
        if matched >= 0:
            self.curRoot = matched
            resultWindow.RootResult(self.master, self)

    def __OpenRootEditor(self):
        """Open the integrated editor for the current/only processed ROI."""
        processed = [i for i, root in enumerate(self.RootAll) if root.bProcessed]
        if self.curRoot not in processed:
            if len(processed) == 1:
                self.curRoot = processed[0]
            else:
                messagebox.showinfo(
                    self._loc("Root annotation editor", "根アノテーションエディター"),
                    self._loc(
                        "Double-click a processed root ROI first, then open the editor from the Tool menu.",
                        "処理済みの根 ROI をダブルクリックしてから、ツールメニューからエディターを開いてください。",
                    ),
                    parent=self.master,
                )
                return
        resultWindow.RootResult(self.master, self)

    def __OpenStandaloneEditor(self):
        """Open the drawing tool without requiring a project or AI model."""
        resultWindow.OpenStandaloneEditor(self.master, language=self.ui_language)

    def __fullscreen_toggle(self, event=None, state=None):
        """ Enable/disable the fullscreen mode """
        if state is not None:
            self.__is_fullscreen = state
        else:
            self.__is_fullscreen = not self.__is_fullscreen  # toggling the boolean
        # Hide menubar in fullscreen mode or show it otherwise
        if self.__is_fullscreen:
            self.__menubar_hide()
        else:  # show menubar
            self.__menubar_show()
        self.master.wm_attributes('-fullscreen', self.__is_fullscreen)  # fullscreen mode on/off

    def __menubar_show(self):
        """ Show menu bar """
        self.master.configure(menu=self.__menubar)

    def __menubar_hide(self):
        """ Hide menu bar """
        self.master.configure(menu=self.__empty_menu)
    def _show_pixel_at_pointer(self, event=None):
        """Show image coordinates and pixel value at the left mouse position."""
        if event is None:
            return
    
        frame = getattr(self, "_MainGUI__imframe", None)
        image = getattr(self, "curImage", None)
    
        if frame is None or image is None:
            return
    
        canvas = getattr(frame, "canvas", None)
        if canvas is None:
            return
    
        # Only respond to the image canvas, not buttons or other widgets.
        if event.widget is not canvas:
            return
    
        scale = float(getattr(frame, "imscale", 0))
        if scale <= 0:
            return
    
        bbox = canvas.coords(frame.container)
        if len(bbox) < 2:
            return
    
        # Convert canvas coordinates to original image coordinates.
        cx = canvas.canvasx(event.x)
        cy = canvas.canvasy(event.y)
    
        x = int(math.floor((cx - bbox[0]) / scale))
        y = int(math.floor((cy - bbox[1]) / scale))
    
        arr = np.asarray(image)
        h, w = arr.shape[:2]
    
        # Do not read outside the image.
        if not (0 <= x < w and 0 <= y < h):
            self.pixel_status_var.set("X: --  Y: --  |  Pixel: --")
            return
    
        pixel = arr[y, x]
    
        if arr.ndim == 2:
            # Grayscale image
            value_text = f"Gray: {int(pixel)}"
    
        elif arr.ndim == 3 and arr.shape[2] == 1:
            value_text = f"Gray: {int(pixel[0])}"
    
        elif arr.ndim == 3 and arr.shape[2] >= 3:
            # curImage is RGB in the current image-loading path.
            r, g, b = (int(v) for v in pixel[:3])
            value_text = f"RGB: ({r}, {g}, {b})"
    
        else:
            value_text = str(pixel)
    
        self.pixel_status_var.set(
            f"X: {x}  Y: {y}  |  {value_text}"
        )
    def __motion(self, event):
        """ Track mouse pointer and handle its position """
        if self.__is_fullscreen:
            y = self.master.winfo_pointery()
            if 0 <= y < 20:  # if close to the upper side of the main window
                self.__menubar_show()
            else:
                self.__menubar_hide()

    def __keystroke(self, event):
        # Tk's Control modifier bit is portable; numeric keycodes are not.
        # This works on Windows and Ubuntu/X11/Wayland with the same shortcut table.
        control_down = bool(int(getattr(event, 'state', 0)) & 0x0004)
        if control_down:
            keysym = str(getattr(event, 'keysym', '') or '').lower()
            for shortcut in self.__shortcuts:
                if keysym == shortcut[1]:
                    shortcut[2]()
                    break
        self.__previous_state = int(getattr(event, 'state', 0))

    def __SetSelectMode(self, event=None):
        
        # print(self.mSelectMode, (self.mSelectMode) % 2)  # debug
        self.__imframe.SetSelectMode((self.mSelectMode)%2 )
        self.mSelectMode = self.mSelectMode + 1
        
    def __default_geometry(self, event=None):
        """ Reset default geomentry for the main GUI window """
        self.__fullscreen_toggle(state=False)  # exit from fullscreen
        self._safe_apply_window_state(self.master, self.__config.default_state)  # cross-platform
        self.__config.set_win_geometry(self.__config.default_geometry)  # save default to config
        self.master.geometry(self.__config.default_geometry)  # set default geometry

    def __resize_master(self, event=None):
        """ Save main window size and position into config file.
            There is a bug when changing window from fullscreen to zoomed and then to normal mode.
            Main window somehow remembers zoomed mode as normal, so I have to explicitly set
            previous geometry from config INI file to the main window. """
        if self.master.wm_attributes('-fullscreen'):  # don't remember fullscreen
            self.__bugfix = True  # fixing bug
            return
        if self.master.state() == 'normal':
            if self.__bugfix is True:  # fixing bug for: fullscreen --> zoomed --> normal
                self.__bugfix = False
                # Explicitly set previous geometry to fix the bug
                self.master.geometry(self.__config.get_win_geometry())
                return
            self.__config.set_win_geometry(self.master.winfo_geometry())
        self.__config.set_win_state(self.master.wm_state())

    @staticmethod
    def _safe_apply_window_state(window, state):
        """Apply a saved Tk window state without assuming Windows semantics.

        ``zoomed`` is not uniformly supported by Tk window managers on Ubuntu/X11/
        Wayland.  Try the normal state API first, then the ``-zoomed`` attribute,
        and finally fall back to ``normal`` instead of failing during startup.
        """
        state = str(state or 'normal').strip().lower()
        if state not in ('normal', 'zoomed', 'iconic'):
            state = 'normal'
        try:
            window.wm_state(state)
            return
        except tk.TclError:
            pass
        if state == 'zoomed':
            try:
                window.wm_attributes('-zoomed', True)
                return
            except tk.TclError:
                pass
        try:
            window.wm_state('normal')
        except tk.TclError:
            pass

    def _set_window_icon(self, window):
        """Set a window icon on Windows and Ubuntu without requiring iconbitmap."""
        try:
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            candidates = [
                os.path.join(cur_dir, 'root.png'),
                os.path.join(cur_dir, 'root.ico'),
            ]
            icon_path = next((p for p in candidates if os.path.isfile(p)), None)
            if not icon_path:
                return

            # iconbitmap is most reliable on Windows.  On Linux/X11/Wayland,
            # iconphoto is considerably more portable, including when the source
            # asset itself is an .ico file that Pillow can decode.
            if os.name == 'nt' and icon_path.lower().endswith('.ico'):
                try:
                    window.iconbitmap(icon_path)
                except tk.TclError:
                    pass

            with Image.open(icon_path) as icon_img:
                icon_img = icon_img.convert('RGBA')
                icon = ImageTk.PhotoImage(icon_img)
            window.iconphoto(True, icon)
            refs = getattr(self, '_rapid_window_icons', None)
            if refs is None:
                refs = []
                self._rapid_window_icons = refs
            refs.append(icon)  # keep Tk image alive
        except Exception:
            # A missing/unsupported icon must never prevent the GUI from opening.
            pass

    @staticmethod
    def _portable_xml_relpath(path, project_dir):
        """Return a project-relative path using '/' as the XML separator."""
        if not path:
            return ''
        try:
            rel = os.path.relpath(os.path.abspath(path), os.path.abspath(project_dir))
        except (OSError, ValueError):
            rel = os.path.basename(str(path).replace('\\', '/'))
        # Project resources are expected to live inside ProjectDir.  If a legacy
        # external path slips through, store only its basename after it has been
        # localized by _ensure_project_local_source().
        if rel == os.pardir or rel.startswith(os.pardir + os.sep):
            rel = os.path.basename(str(path).replace('\\', '/'))
        return rel.replace('\\', '/')

    def _ensure_project_local_source(self):
        """Ensure the source image is stored inside the project before XML save."""
        path = str(getattr(self, 'imgpath', '') or '').strip()
        project_dir = str(getattr(self, 'ProjectDir', '') or '').strip()
        if not path or not project_dir or not os.path.isfile(path):
            return path
        try:
            abs_path = os.path.abspath(path)
            abs_project = os.path.abspath(project_dir)
            inside = os.path.commonpath([abs_path, abs_project]) == abs_project
        except (OSError, ValueError):
            inside = False
        if inside:
            return path

        dst = os.path.join(project_dir, os.path.basename(path))
        try:
            os.makedirs(project_dir, exist_ok=True)
            if os.path.abspath(path) != os.path.abspath(dst):
                shutil.copy2(path, dst)
            self.imgpath = dst
            return dst
        except Exception:
            # Do not destroy save compatibility if a legacy source cannot be copied.
            return path

    def __create_widgets(self):
        """ Widgets for GUI are created here """
        # Enable/disable these menu labels in the main window
        self.__label_recent = 'Open recent'
        self.__label_close = 'Close project'
        self.__label_save = 'Save project'
        self.__AI_Select_Label = 'AI Select'
        self.__label_openProject = 'Open project'
        # Create menu for the image.
        self.__image_menu = tk.Menu(self.__menubar, tearoff=False, postcommand=self.__list_recent)
        self.__image_menu.add_command(label='Create project', command=self.__shortcuts[0][2],
                                      accelerator=self.__shortcuts[0][0])

        self.__image_menu.add_command(label='Batch Processing', command=self.__OpenFolder)
        self.__image_menu.add_command(label=self.__label_openProject, command=self.__shortcuts[3][2],
                                      accelerator=self.__shortcuts[3][0])#, state='disabled'
        self.__recent_images = tk.Menu(self.__image_menu, tearoff=False)
        self.__image_menu.add_cascade(label=self.__label_recent, menu=self.__recent_images)
        # Store numeric menu indices instead of using label text as an index.
        # Labels are translated at runtime, so label-based entryconfigure() calls
        # break after switching language (e.g. "Open recent" -> Japanese).
        self.__index_recent = self.__image_menu.index('end')
        self.__image_menu.add_command(label=self.__label_close, command=self.__shortcuts[1][2],
                                      accelerator=self.__shortcuts[1][0])#, state='disabled'
        self.__index_close = self.__image_menu.index('end')
        self.__image_menu.add_separator()
        self.__image_menu.add_command(label='Batch export dataset...', command=self.__BatchExportDialog)
        #self.__image_menu.add_command(label=self.__label_save, command=self.__shortcuts[2][2],
        #                              accelerator=self.__shortcuts[2][0])#, state='disabled'
        
         
        self.__menubar.add_cascade(label='File', menu=self.__image_menu)
        self.__image_menu.add_separator()
        self.__image_menu.add_command(label='Exit', command=self.destroy, accelerator=u'Alt+F4')
        # Create menu for the view: fullscreen, default size, etc.
        self.__view_menu = tk.Menu(self.__menubar, tearoff=False)
        self.__view_menu.add_command(label='Fullscreen', command=self.__fullscreen_toggle,
                                     accelerator='F11')
        self.__view_menu.add_command(label='Default size', command=self.__default_geometry,
                                     accelerator='F5')
        self.__view_menu.add_command(label='Scale calibration...', command=self.__SetResolution,
                                     accelerator='F6')
        self.__view_menu.add_command(label='Detect ruler scale...', command=self.__AutoDetectScale)
        #self.__view_menu.add_command(label='Rotate 90', command=self.__SetRotation)
        self.__menubar.add_cascade(label='View', menu=self.__view_menu)
        
        self.__process_menu = tk.Menu(self.__menubar, tearoff=False)
        self.__process_menu.add_command(label=self.__AI_Select_Label, command=self.__AI_Select)
        self.__process_menu.add_command(label='ICWAPR', command=self.__ICML_NoAI_toggle,
                                     accelerator='F9')
        self.__process_menu.add_command(label='Journal', command=self.__ICML_Improve_toggle,
                                     accelerator='F10')
        self.__process_menu.add_separator()
        self.__process_menu.add_command(label='Annotation(Addon)', command=self.__OpenRootEditor)
        self.__process_menu.add_command(label='Annotation(Standalone)', command=self.__OpenStandaloneEditor)
        
        #self.__process_menu.add_command(label='UPDOWN', command=self.__UpDown_toggle)
        self.__menubar.add_cascade(label='Tool', menu=self.__process_menu)
                 
        # Language sits immediately to the left of Help.  Keep the menu label
        # translatable, but keep the language choices themselves stable.
        self.__language_menu=tk.Menu(self.__menubar,tearoff=False)
        self.__language_menu.add_command(label='English',command=lambda:self._set_language('en'))
        self.__language_menu.add_command(label='日本語',command=lambda:self._set_language('ja'))
        self.__menubar.add_cascade(label='Language',menu=self.__language_menu)

        self.__help_menu = tk.Menu(self.__menubar, tearoff=False)
        self.__help_menu.add_command(label='Help', command=self.__HelpDialog)
        self.__help_menu.add_command(label='About', command=self.__AboutDialog)
        self.__menubar.add_cascade(label='Help', menu=self.__help_menu)
        
        # Create placeholder frame for the image
        self.master.rowconfigure(0, weight=1)  # make grid cell expandable
        self.master.columnconfigure(0, weight=1)
        self.__placeholder = ttk.Frame(self.master)
        self.__placeholder.grid(row=0, column=0, sticky='nswe')
        self.__placeholder.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__placeholder.columnconfigure(0, weight=1)
        
        # Main status bar
        status_frame = ttk.Frame(self.master)
        status_frame.grid(row=1, column=0, sticky="ew")
        status_frame.columnconfigure(0, weight=1)
        
        # Left: existing application messages
        self.status_var = tk.StringVar()
        self.status_var.set(self._tr("Ready"))
        
        self.status_label = tk.Label(
            status_frame,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#f0f0f0"
        )
        self.status_label.grid(row=0, column=0, sticky="ew")
        
        # Right: mouse image coordinates and pixel value
        self.pixel_status_var = tk.StringVar(value="X: --  Y: --  |  Pixel: --")
        
        self.pixel_status_label = tk.Label(
            status_frame,
            textvariable=self.pixel_status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg="#f0f0f0",
            padx=8
        )
        self.pixel_status_label.grid(row=0, column=1, sticky="e")

        
        self.bSetResolution = False
        self.scale_source = 'default'
        self.scale_confidence = 0.0
        
        self.InitialVariables()  
    def _tr(self, s):
        if getattr(self,"ui_language","en")=="ja": return _RAPID_JA.get(s,s)
        return _RAPID_EN.get(s,s)
    def _loc(self, en, ja):
        return ja if getattr(self, "ui_language", "en") == "ja" else en
    def _runtime_message(self, text):
        """Translate short runtime/batch messages without changing stored log values."""
        if getattr(self, "ui_language", "en") != "ja":
            return str(text)
        text = str(text)
        exact = {
            "YOLO detection": "YOLO 検出",
            "Saving measurements and masks": "計測結果とマスクを保存中",
            "pixel-only": "ピクセルのみ",
            "no YOLO ROI detected": "YOLO ROI を検出できませんでした",
            "invalid ruler spacing": "定規の目盛り間隔が無効です",
            "no regular ruler ticks detected": "規則的な定規目盛りを検出できませんでした",
        }
        if text in exact:
            return exact[text]
        substitutions = (
            ("ICML segmentation", "ICML セグメンテーション"),
            ("shared scale", "共通スケール"),
            ("low ruler confidence", "定規検出の信頼度が低い"),
            ("ruler ", "定規 "),
            ("confidence", "信頼度"),
            ("Skipped: already processed with", "スキップ: 処理済み"),
            ("already processed with", "処理済み"),
            ("Done:", "完了:"),
            ("Review:", "要確認:"),
            ("Failed:", "失敗:"),
            ("Journal segmentation", "論文セグメンテーション"),
            ("ROI(s)", "ROI"),
        )
        for src, dst in substitutions:
            text = text.replace(src, dst)
        return text
    def _tooltip_text(self,key):
        return (_RAPID_TIPS_JA if getattr(self,"ui_language","en")=="ja" else _RAPID_TIPS_EN).get(key,"")
    def _add_tooltip(self,w,key):
        try: self._rapid_tooltips.append(_RAPIDToolTip(w,lambda k=key:self._tooltip_text(k)))
        except Exception: pass
        return w
    def _translate_widget_tree(self,root):
        target=_RAPID_JA if self.ui_language=="ja" else _RAPID_EN
        reverse=_RAPID_EN if self.ui_language=="ja" else _RAPID_JA
        def cv(s):
            b=reverse.get(s,s); return target.get(b,b)
        def walk(w):
            try:
                if "text" in w.keys(): w.configure(text=cv(w.cget("text")))
            except Exception: pass
            try:
                for c in w.winfo_children(): walk(c)
            except Exception: pass
        walk(root)
    def _translate_menu(self,menu):
        target=_RAPID_JA if self.ui_language=="ja" else _RAPID_EN
        reverse=_RAPID_EN if self.ui_language=="ja" else _RAPID_JA
        try: end=menu.index("end")
        except Exception: return
        if end is None: return
        for i in range(end+1):
            try:
                lab=menu.entrycget(i,"label")
                if lab:
                    b=reverse.get(lab,lab); menu.entryconfig(i,label=target.get(b,b))
                sub=menu.entrycget(i,"menu")
                if sub: self._translate_menu(menu.nametowidget(sub))
            except Exception: pass
    def _set_language(self,lang):
        self.ui_language="ja" if lang=="ja" else "en"
        self._translate_menu(self.__menubar); self._translate_widget_tree(self.master)
        try:
            self.treeview_1.heading("#0",text=self._tr("Index"))
            self.treeview_1.heading("DATA",text=self._tr("Image-names"))
        except Exception: pass
        # Variable-backed labels are not handled by widget-tree translation.
        try:
            if self.status_var.get() in ("Ready", "準備完了"):
                self.status_var.set(self._tr("Ready"))
        except Exception:
            pass

    def __OpenFolder(self):
        self.new_windowFolder=tk.Toplevel(self.master); self.new_windowFolder.grab_set()
        self.new_windowFolder.title(self._tr("Batch Processing"))
        self.text_Pro=tk.StringVar(self.new_windowFolder); self.text_Img=tk.StringVar(self.new_windowFolder)
        self.text_ImgFolder=tk.StringVar(self.new_windowFolder)
        rp=self._load_recent_folders("batch_project_folder"); ri=self._load_recent_folders("batch_image_folder")
        self.text_Pro.set(rp[0] if rp else os.path.join(os.getcwd(),"proj"))
        if ri: self.text_ImgFolder.set(ri[0])
        tk.Label(self.new_windowFolder,text=self._tr("Project folder")).grid(row=0,column=0,padx=4,pady=4,sticky="w")
        self.entry3=tk.Entry(self.new_windowFolder,textvariable=self.text_Pro,width=52); self.entry3.grid(row=0,column=1,padx=4,pady=4,sticky="ew")
        b1=tk.Button(self.new_windowFolder,text=self._tr("Project Folder"),command=lambda:self.update_varfolder(self.text_Pro,"batch_project_folder"))
        b1.grid(row=0,column=2,padx=4,pady=4); self._add_tooltip(b1,"Project Folder")
        c1=ttk.Combobox(self.new_windowFolder,values=rp,state="readonly",width=28); c1.grid(row=0,column=3,padx=(2,6),pady=4,sticky="ew")
        if rp: c1.set(rp[0])
        c1.bind("<<ComboboxSelected>>",lambda e:self._apply_recent_folder(self.text_Pro,c1.get()))
        tk.Label(self.new_windowFolder,text=self._tr("Image folder")).grid(row=1,column=0,padx=4,pady=4,sticky="w")
        self.entry4=tk.Entry(self.new_windowFolder,textvariable=self.text_ImgFolder,width=52); self.entry4.grid(row=1,column=1,padx=4,pady=4,sticky="ew")
        b2=tk.Button(self.new_windowFolder,text=self._tr("Image Folder"),command=lambda:self.update_varfolder(self.text_ImgFolder,"batch_image_folder"))
        b2.grid(row=1,column=2,padx=4,pady=4); self._add_tooltip(b2,"Image Folder")
        c2=ttk.Combobox(self.new_windowFolder,values=ri,state="readonly",width=28); c2.grid(row=1,column=3,padx=(2,6),pady=4,sticky="ew")
        if ri: c2.set(ri[0])
        c2.bind("<<ComboboxSelected>>",lambda e:self._apply_recent_folder(self.text_ImgFolder,c2.get()))
        tk.Button(self.new_windowFolder,text=self._tr("OK"),command=self.show_selected_imagefolders).grid(row=2,column=1,pady=7)
        bc=tk.Button(self.new_windowFolder,text=self._tr("Cancel"),command=self.new_windowFolder.destroy); bc.grid(row=2,column=2,pady=7); self._add_tooltip(bc,"Cancel")
        self.new_windowFolder.columnconfigure(1,weight=1); self.new_windowFolder.columnconfigure(3,weight=1); self.new_windowFolder.minsize(760,120)
    def show_selected_imagefolders(self):
        
        self.proPath, imgFolder = os.path.normpath(self.entry3.get()), os.path.normpath(self.entry4.get())
        self.batch_input_dir = imgFolder
        if os.path.isdir(self.proPath): self._remember_recent_folder('batch_project_folder',self.proPath)
        if os.path.isdir(imgFolder): self._remember_recent_folder('batch_image_folder',imgFolder)
        patterns = ('*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff')
        self.imgList = []
        for pattern in patterns:
            self.imgList.extend(glob.glob(os.path.join(imgFolder, '**', pattern), recursive=True))
        self.imgList = sorted(set(self.imgList))
        #self.new_windowFolder.grab_release()
        self.new_windowFolder.destroy()
        
        self.__ControlImgs()
    def __ControlImgs(self):
        """Batch-processing control window with progress, scale and device options."""
        if not self.imgList:
            messagebox.showwarning(self._tr("Batch Processing"), self._loc("No supported images were found in the selected folder.", "選択したフォルダーに対応画像が見つかりませんでした。"))
            return

        self.contWindow = tk.Toplevel(self.master)
        self.contWindow.title(self._tr("Batch Processing"))
        self.contWindow.geometry("900x650")
        self.contWindow.minsize(760, 520)

        controls = ttk.LabelFrame(self.contWindow, text=self._tr("Batch settings"), padding=10)
        controls.pack(fill='x', padx=10, pady=(10, 6))

        ttk.Label(controls, text=self._tr("Device:")).grid(row=0, column=0, sticky='w', padx=(0, 5), pady=3)
        self.batch_device_var = tk.StringVar(value="Auto")
        ttk.Combobox(controls, textvariable=self.batch_device_var,
                     values=("Auto", "GPU", "CPU"), state='readonly', width=10).grid(
                         row=0, column=1, sticky='w', pady=3)

        ttk.Label(controls, text=self._tr("Segmentation:")).grid(row=0, column=2, sticky='w', padx=(18, 5), pady=3)
        self.batch_segmentation_var = tk.StringVar(value="ICML-improve")
        ttk.Combobox(controls, textvariable=self.batch_segmentation_var, state='readonly', width=28,
                     values=("ICML-NoAI", "ICML-improve")).grid(
                         row=0, column=3, sticky='w', pady=3)

        ttk.Label(controls, text=self._tr("Scale mode:")).grid(row=1, column=0, sticky='w', padx=(0, 5), pady=3)
        self.batch_scale_var = tk.StringVar(value=self._tr("Same scale for all images"))
        scale_combo = ttk.Combobox(
            controls, textvariable=self.batch_scale_var, state='readonly', width=28,
            values=(self._tr("Same scale for all images"), self._tr("Detect ruler for every image"), self._tr("Pixel only")))
        scale_combo.grid(row=1, column=1, columnspan=3, sticky='w', pady=3)

        ttk.Label(controls, text=self._tr("Ruler interval:")).grid(row=2, column=0, sticky='w', padx=(0, 5), pady=3)
        self.batch_tick_physical_var = tk.StringVar(value="1")
        ruler_value_frame = ttk.Frame(controls)
        ruler_value_frame.grid(row=2, column=1, sticky='w', pady=3)
        ttk.Entry(ruler_value_frame, textvariable=self.batch_tick_physical_var, width=9).pack(side='left')
        self.batch_tick_unit_var = tk.StringVar(value=getattr(self, 'unit', 'mm') if getattr(self, 'unit', '') in ('mm', 'cm', 'um') else 'mm')
        ttk.Combobox(ruler_value_frame, textvariable=self.batch_tick_unit_var,
                     values=('mm', 'cm', 'um'), state='readonly', width=6).pack(side='left', padx=(4, 0))

        ttk.Label(controls, text=self._tr("Min confidence:")).grid(row=2, column=2, sticky='w', padx=(18, 5), pady=3)
        self.batch_confidence_var = tk.StringVar(value="0.55")
        ttk.Entry(controls, textvariable=self.batch_confidence_var, width=8).grid(
            row=2, column=3, sticky='w', pady=3)

        self.batch_skip_var = tk.BooleanVar(value=True)
        self.batch_continue_var = tk.BooleanVar(value=True)
        _skip_cb=ttk.Checkbutton(controls, text=self._tr("Skip already processed"), variable=self.batch_skip_var)
        _skip_cb.grid(row=3, column=0, columnspan=2, sticky='w', pady=(5, 2)); self._add_tooltip(_skip_cb,"Skip already processed")
        _continue_cb=ttk.Checkbutton(controls, text=self._tr("Continue if an image fails"), variable=self.batch_continue_var)
        _continue_cb.grid(row=3, column=2, columnspan=2, sticky='w', pady=(5, 2)); self._add_tooltip(_continue_cb,"Continue if an image fails")

        current_scale = self._loc(
            f"Current shared scale: {getattr(self, 'resolution', 1.0):.8g} {getattr(self, 'unit', 'px')}/pixel; source={getattr(self, 'scale_source', 'default')}",
            f"現在の共通スケール: {getattr(self, 'resolution', 1.0):.8g} {getattr(self, 'unit', 'px')}/pixel; ソース={getattr(self, 'scale_source', 'default')}"
        )
        self.batch_scale_info = ttk.Label(controls, text=current_scale, foreground='gray')
        self.batch_scale_info.grid(row=4, column=0, columnspan=4, sticky='w', pady=(4, 0))
        controls.columnconfigure(3, weight=1)

        actions = ttk.Frame(self.contWindow)
        actions.pack(fill='x', padx=10, pady=4)
        self.buttonRatio = ttk.Button(actions, text=self._tr("Set shared scale"), command=self.__SetResolution)
        self.buttonRatio.pack(side='left', padx=(0, 5))
        self._add_tooltip(self.buttonRatio, "Set shared scale")
        self.buttonAutoScale = ttk.Button(actions, text=self._tr("Preview ruler"), command=self.__AutoDetectScale)
        self.buttonAutoScale.pack(side='left', padx=5)
        self._add_tooltip(self.buttonAutoScale, "Preview ruler")
        self.button = ttk.Button(actions, text=self._tr("Start"), command=self.batchProcess)
        self.button.pack(side='left', padx=(20, 5))
        self._add_tooltip(self.button, "Start")
        self.buttonPauseBatch = ttk.Button(actions, text=self._tr("Pause"), command=self._toggle_batch_pause, state='disabled')
        self.buttonPauseBatch.pack(side='left', padx=5)
        self._add_tooltip(self.buttonPauseBatch, "Pause")
        self.buttonCancelBatch = ttk.Button(actions, text=self._tr("Cancel"), command=self._cancel_batch, state='disabled')
        self.buttonCancelBatch.pack(side='left', padx=5)
        self._add_tooltip(self.buttonCancelBatch, "Cancel")
        self.buttonExport = ttk.Button(actions, text=self._tr("Export"), command=self.__BatchExportDialog)
        self.buttonExport.pack(side='right', padx=5)
        self._add_tooltip(self.buttonExport, "Export")

        progress_frame = ttk.LabelFrame(self.contWindow, text=self._tr("Progress"), padding=8)
        progress_frame.pack(fill='x', padx=10, pady=6)
        self.batch_progress = ttk.Progressbar(progress_frame, mode='determinate', maximum=max(1, len(self.imgList)))
        self.batch_progress.pack(fill='x')
        self.batch_progress_text = tk.StringVar(value=f"0 / {len(self.imgList)}")
        ttk.Label(progress_frame, textvariable=self.batch_progress_text).pack(anchor='w', pady=(4, 0))
        self.batch_current_text = tk.StringVar(value=self._tr("Ready"))
        ttk.Label(progress_frame, textvariable=self.batch_current_text).pack(anchor='w')
        self.batch_counts_text = tk.StringVar(value=self._loc("Done: 0    Failed: 0    Review: 0    Skipped: 0", "完了: 0    失敗: 0    要確認: 0    スキップ: 0"))
        ttk.Label(progress_frame, textvariable=self.batch_counts_text).pack(anchor='w')

        tree_frame = ttk.Frame(self.contWindow)
        tree_frame.pack(fill='both', expand=True, padx=10, pady=(2, 10))
        self.treeview_1 = MyScrolledTreeview(tree_frame, head=[self._tr("Index"), self._tr("Image-names")], height=10, data=self.imgList)
        self.treeview_1.pack(fill='both', expand=True)
        self.treeview_1.column("#0", width=55, minwidth=45, stretch=False, anchor="center")
        self.treeview_1.column("DATA", width=720, minwidth=220, stretch=True, anchor="w")
        self.treeview_1.bind("<<TreeviewSelect>>", self.select_record)
        self._translate_widget_tree(self.contWindow)

    def select_record(self, event=None):
        if self.batch_thread is not None and self.batch_thread.is_alive():
            self.status_var.set(self._loc("Batch processing is running; project preview is temporarily disabled.", "バッチ処理中のため、プロジェクトのプレビューは一時的に無効です。"))
            return
        curid = self.treeview_1.focus()
        if not curid:
            return
        self.curIndex = self.treeview_1.index(curid)
        path = self.imgList[self.curIndex]
        project_dir, file_name = self._batch_project_location(path)
        project_path = os.path.join(project_dir, file_name + ".xml")
        if os.path.isfile(project_path):
            self.__Open_Project(project_path)
        else:
            self.status_var.set(self._loc(f"No processed project yet: {os.path.basename(path)}", f"まだ処理済みプロジェクトがありません: {os.path.basename(path)}"))

    @staticmethod
    def _safe_project_component(value):
        value = re.sub(r'[^\w.-]+', '_', value.strip(), flags=re.UNICODE)
        return value.strip('._') or 'image'

    def _batch_project_location(self, image_path):
        """Use the relative input path to avoid collisions between equal basenames."""
        root = getattr(self, 'batch_input_dir', os.path.dirname(image_path))
        try:
            rel = os.path.relpath(image_path, root)
        except Exception:
            rel = os.path.basename(image_path)
        rel_no_ext = os.path.splitext(rel)[0]
        parts = [self._safe_project_component(p) for p in Path(rel_no_ext).parts if p not in ('.', '..')]
        project_name = '__'.join(parts) if parts else self._safe_project_component(os.path.splitext(os.path.basename(image_path))[0])
        return os.path.join(self.proPath, project_name), project_name

    def _load_image_for_processing(self, path):
        """Load an input image consistently for interactive and batch analysis."""
        with Image.open(path) as pil:
            if pil.mode == 'I;16' or (pil.mode.startswith('I') and np.asarray(pil).dtype == np.uint16):
                arr16 = np.asarray(pil, dtype=np.uint16)
                arr = (arr16 >> 8).astype(np.uint8)
                return arr, cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB), 1
            if pil.mode in ('RGB', 'RGBA'):
                rgb = np.asarray(pil.convert('RGB'), dtype=np.uint8)
                return rgb, rgb.copy(), 2
            gray = np.asarray(pil.convert('L'), dtype=np.uint8)
            return gray, cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 1

    def _resolve_yolo_device(self, preference="Auto"):
        pref = str(preference or 'Auto').strip().lower()
        if pref == 'cpu':
            return 'cpu'
        if pref == 'gpu':
            if not torch.cuda.is_available():
                raise RuntimeError("GPU was requested, but PyTorch reports that CUDA is not available.")
            return '0'
        return '0' if torch.cuda.is_available() else 'cpu'

    def _ensure_yolo_model(self):
        with self._model_lock:
            if self.yoloModel is None:
                self.yoloModel = self._load_yolo_model()
        return self.yoloModel

    def _active_yolo_device(self):
        value = str(getattr(self, 'yolo_device', 'auto')).strip().lower()
        if value in ('0', 'cuda', 'cuda:0', 'gpu'):
            return '0' if torch.cuda.is_available() else 'cpu'
        if value == 'cpu':
            return 'cpu'
        return '0' if torch.cuda.is_available() else 'cpu'


    def _apply_batch_scale(self, mode, shared_scale, physical_tick, tick_unit, min_confidence):
        """Set scale for current batch image. Return (status, message)."""
        if mode == "Pixel only":
            self.resolution = 1.0
            self.unit = 'px'
            self.bSetResolution = False
            self.scale_source = 'pixel-only'
            self.scale_confidence = 1.0
            return 'ok', 'pixel-only'

        if mode == "Same scale for all images":
            self.resolution, self.unit, self.bSetResolution, self.scale_source, self.scale_confidence = shared_scale
            return 'ok', f"shared scale {self.resolution:.8g} {self.unit}/px"

        candidates = self._find_ruler_tick_candidates(self.curImage)
        if candidates:
            best = candidates[0]
            # Save the actual detection overlay for later audit, even in fully
            # automatic batch mode. Yellow=search band, green=ticks, red=spacing.
            try:
                preview = self._make_ruler_preview(best, max_size=(100000, 100000))
                preview.save(os.path.join(self.ProjectDir, 'ruler_detection.png'))
            except Exception:
                logging.warning("Could not save batch ruler preview for %s", getattr(self, 'imgpath', ''))
            confidence = float(best.get('confidence', 0.0))
            spacing = float(best.get('spacing', 0.0))
            if confidence >= min_confidence and spacing > 0:
                self.resolution = float(physical_tick) / spacing
                self.unit = tick_unit
                self.bSetResolution = True
                self.scale_source = 'ruler-auto-batch'
                self.scale_confidence = confidence
                self.scale_pixels_per_tick = spacing
                self.scale_physical_per_tick = float(physical_tick)
                return 'ok', f"ruler {spacing:.3f} px/tick, confidence {confidence:.0%}"
            reason = f"low ruler confidence {confidence:.0%}" if spacing > 0 else "invalid ruler spacing"
        else:
            confidence = 0.0
            reason = "no regular ruler ticks detected"

        # Never silently use a questionable physical scale. Continue measuring in pixels
        # and mark the image for review.
        self.resolution = 1.0
        self.unit = 'px'
        self.bSetResolution = False
        self.scale_source = 'ruler-review'
        self.scale_confidence = confidence
        return 'review', reason

    def _root_crop_source(self):
        """Return a 3-channel image for ROI cropping while preserving legacy RGB channel handling."""
        arr = np.asarray(self.curImage)
        if arr.ndim == 2:
            return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        return cv2.cvtColor(arr[:, :, :3].astype(np.uint8), cv2.COLOR_BGR2RGB)

    def _process_batch_image_sync(self, image_path, settings, stage_callback=None):
        """Run YOLO -> segmentation -> measurement synchronously for one image."""
        self.InitialVariables()
        self.imgpath = image_path
        self.ProjectDir, self.file_name = self._batch_project_location(image_path)
        os.makedirs(self.ProjectDir, exist_ok=True)

        self.curImage, self.mResultImg, self.imgType = self._load_image_for_processing(image_path)
        scale_status, scale_message = self._apply_batch_scale(
            settings['scale_mode'], settings['shared_scale'], settings['physical_tick'],
            settings['tick_unit'], settings['min_confidence'])

        if stage_callback:
            stage_callback("YOLO detection")
        model = self._ensure_yolo_model()
        if self.imgType == 1:
            detect_img = cv2.convertScaleAbs(self.curImage, alpha=1.75, beta=50)
            detect_img = cv2.cvtColor(detect_img, cv2.COLOR_GRAY2RGB)
        else:
            detect_img = cv2.cvtColor(self.curImage, cv2.COLOR_BGR2RGB)
        rois = self.detectYOLO(model, detect_img, allow_gui_rotation=False)
        rois = sorted(rois, key=lambda r: (r.ROICoord[1][0] + r.ROICoord[0][0]))
        self.RootAll = [EachRootData(r) for r in rois]
        if not self.RootAll:
            scale_status = 'review'
            scale_message = (scale_message + '; ' if scale_message else '') + 'no YOLO ROI detected'

        segmentation_method = settings.get('segmentation_method', 'ICML-improve')
        if segmentation_method == 'ICML-NoAI':
            segmentation_mode = 1
        else:
            segmentation_method = 'ICML-improve'
            segmentation_mode = 5
        self.processing_method = segmentation_method
        crop_source = self._root_crop_source()
        for roi_idx, eachroot in enumerate(self.RootAll, start=1):
            if stage_callback:
                if segmentation_method == 'ICML-improve':
                    stage_name = 'ICML-improve segmentation'
                else:
                    stage_name = 'ICML segmentation'
                stage_callback(f"{stage_name} ROI {roi_idx}/{len(self.RootAll)}")
            eachroot.Generatebb(crop_source, segmentation_mode)
            p = ar(eachroot.mImgCV, segmentation_mode, 0, 0)
            p.Processing()
            eachroot.prop = p.Get().copy()
            eachroot.roilabel = p.GetLabel().copy()
            eachroot.segmask = p.GetLabelLoad().copy()
            eachroot.bProcessed = True

        # Preserve the source image in the project folder only; do not pollute cwd.
        dst_image = os.path.join(self.ProjectDir, os.path.basename(image_path))
        if os.path.abspath(image_path) != os.path.abspath(dst_image) and not os.path.exists(dst_image):
            shutil.copy2(image_path, dst_image)
        self.imgpath = dst_image
        if stage_callback:
            stage_callback("Saving measurements and masks")
        self.SaveResult()
        return scale_status, scale_message, len(self.RootAll)

    @staticmethod
    def _saved_segmentation_method(xml_path):
        """Return the method recorded in a saved project, or None for legacy projects."""
        try:
            root = ET.parse(xml_path).getroot()
            node = root.find('segmentation_method')
            return node.text.strip() if node is not None and node.text else None
        except Exception:
            return None

    def batchProcess(self):
        if self.batch_thread is not None and self.batch_thread.is_alive():
            messagebox.showinfo(self._tr("Batch Processing"), self._loc("A batch is already running.", "バッチ処理はすでに実行中です。"), parent=self.contWindow)
            return
        if not self.imgList:
            return

        try:
            physical_tick = float(self.batch_tick_physical_var.get())
            min_conf = float(self.batch_confidence_var.get())
            if physical_tick <= 0 or not (0.0 <= min_conf <= 1.0):
                raise ValueError
        except ValueError:
            messagebox.showerror(self._tr("Batch Processing"), self._loc("Ruler interval must be > 0 and confidence must be between 0 and 1.", "定規間隔は 0 より大きく、信頼度は 0～1 の範囲で指定してください。"), parent=self.contWindow)
            return

        if _RAPID_EN.get(self.batch_scale_var.get(), self.batch_scale_var.get()) == "Same scale for all images" and not bool(getattr(self, 'bSetResolution', False)):
            messagebox.showwarning(
                self._tr("Batch Processing"),
                self._loc("No verified shared scale is currently set. Use 'Set shared scale', choose 'Detect ruler for every image', or choose 'Pixel only'.", "確認済みの共通スケールが設定されていません。［共通スケールを設定］を使用するか、［各画像で定規を検出］または［ピクセルのみ］を選択してください。"),
                parent=self.contWindow)
            return

        device_pref = self.batch_device_var.get()
        try:
            resolved_device = self._resolve_yolo_device(device_pref)
        except Exception as exc:
            messagebox.showerror(self._tr("Batch Processing"), str(exc), parent=self.contWindow)
            return

        settings = {
            'device_pref': device_pref,
            'device': resolved_device,
            'segmentation_method': _RAPID_EN.get(self.batch_segmentation_var.get(), self.batch_segmentation_var.get()),
            'scale_mode': _RAPID_EN.get(self.batch_scale_var.get(), self.batch_scale_var.get()),
            'physical_tick': physical_tick,
            'tick_unit': self.batch_tick_unit_var.get() or 'mm',
            'min_confidence': min_conf,
            'skip': bool(self.batch_skip_var.get()),
            'continue_on_error': bool(self.batch_continue_var.get()),
            'shared_scale': (
                float(getattr(self, 'resolution', 1.0)), str(getattr(self, 'unit', 'px')),
                bool(getattr(self, 'bSetResolution', False)), str(getattr(self, 'scale_source', 'default')),
                float(getattr(self, 'scale_confidence', 0.0))),
        }
        self.yolo_device = resolved_device
        self.batch_cancel_event.clear()
        self.batch_pause_event.clear()
        self.buttonPauseBatch.configure(text=self._tr('Pause'))
        while not self.batch_queue.empty():
            try:
                self.batch_queue.get_nowait()
            except queue.Empty:
                break
        self.batch_progress['maximum'] = max(1, len(self.imgList))
        self.batch_progress['value'] = 0
        self.batch_progress_text.set(f"0 / {len(self.imgList)}")
        seg_method = settings['segmentation_method']
        device_label = resolved_device.upper() if resolved_device != '0' else 'GPU 0'
        self.batch_current_text.set(self._loc(
            f"Loading YOLO on {device_label}; segmentation: {seg_method}...",
            f"{device_label} で YOLO を読み込み中; セグメンテーション: {seg_method}..."
        ))
        self.batch_counts_text.set(self._loc("Done: 0    Failed: 0    Review: 0    Skipped: 0", "完了: 0    失敗: 0    要確認: 0    スキップ: 0"))
        self.button.configure(state='disabled')
        self.buttonPauseBatch.configure(state='normal')
        self.buttonCancelBatch.configure(state='normal')
        self.buttonExport.configure(state='disabled')
        self.buttonRatio.configure(state='disabled')
        self.buttonAutoScale.configure(state='disabled')

        self.batch_thread = threading.Thread(target=self._batch_worker, args=(settings,), daemon=True)
        self.batch_thread.start()
        self.contWindow.after(100, self._poll_batch_queue)

    def _batch_worker(self, settings):
        counts = {'done': 0, 'failed': 0, 'review': 0, 'skipped': 0}
        os.makedirs(self.proPath, exist_ok=True)
        log_path = os.path.join(self.proPath, 'batch_log.csv')
        log_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0
        try:
            # YOLO localizes root ROIs; ICML performs within-ROI segmentation.
            self._ensure_yolo_model()
            with open(log_path, 'a', newline='', encoding='utf-8-sig') as log_f:
                writer = csv.writer(log_f)
                if not log_exists:
                    writer.writerow(['time', 'image', 'status', 'message', 'project_dir', 'roi_count',
                                     'segmentation_method', 'resolution_per_pixel', 'unit', 'scale_source', 'scale_confidence'])
                for index, fn in enumerate(self.imgList, start=1):
                    while self.batch_pause_event.is_set() and not self.batch_cancel_event.is_set():
                        time.sleep(0.10)
                    if self.batch_cancel_event.is_set():
                        break
                    project_dir, project_name = self._batch_project_location(fn)
                    xml_path = os.path.join(project_dir, project_name + '.xml')
                    result_path = os.path.join(project_dir, 'result.csv')
                    saved_method = self._saved_segmentation_method(xml_path) if os.path.isfile(xml_path) else None
                    if (settings['skip'] and os.path.isfile(xml_path) and os.path.isfile(result_path)
                            and saved_method == settings.get('segmentation_method')):
                        counts['skipped'] += 1
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), fn, 'skipped',
                                         f'already processed with {saved_method}', project_dir, '',
                                         settings.get('segmentation_method', ''), '', '', '', ''])
                        log_f.flush()
                        self.batch_queue.put({'kind': 'progress', 'index': index, 'file': fn, 'counts': counts.copy(),
                                              'message': f'Skipped: already processed with {saved_method}'})
                        continue
                    t0 = time.time()
                    try:
                        def stage(msg, _fn=fn, _index=index):
                            self.batch_queue.put({'kind': 'stage', 'index': _index, 'file': _fn, 'message': msg})
                        scale_status, scale_message, roi_count = self._process_batch_image_sync(fn, settings, stage)
                        if scale_status == 'review':
                            counts['review'] += 1
                            status = 'review'
                        else:
                            counts['done'] += 1
                            status = 'done'
                        message = f"{scale_message}; {roi_count} ROI(s); {time.time()-t0:.1f}s"
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), fn, status, message,
                                         self.ProjectDir, roi_count, settings.get('segmentation_method', ''),
                                         self.resolution, self.unit, self.scale_source, self.scale_confidence])
                        log_f.flush()
                    except Exception as exc:
                        counts['failed'] += 1
                        status = 'failed'
                        message = f"{type(exc).__name__}: {exc}"
                        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), fn, status, message,
                                         project_dir, '', settings.get('segmentation_method', ''), '', '', '', ''])
                        log_f.flush()
                        logging.error("Batch image failed: %s\n%s", fn, traceback.format_exc())
                        if not settings['continue_on_error']:
                            self.batch_queue.put({'kind': 'progress', 'index': index, 'file': fn, 'counts': counts.copy(), 'message': message})
                            break
                    self.batch_queue.put({'kind': 'progress', 'index': index, 'file': fn, 'counts': counts.copy(), 'message': f"{status.title()}: {message}"})
        except Exception as exc:
            self.batch_queue.put({'kind': 'fatal', 'message': f"{type(exc).__name__}: {exc}"})
            logging.error("Batch initialization failed\n%s", traceback.format_exc())
        finally:
            self.batch_queue.put({'kind': 'finished', 'counts': counts.copy(), 'cancelled': self.batch_cancel_event.is_set(), 'log_path': log_path})

    def _poll_batch_queue(self):
        finished = False
        while True:
            try:
                event = self.batch_queue.get_nowait()
            except queue.Empty:
                break
            kind = event.get('kind')
            if kind == 'stage':
                self.batch_current_text.set(f"{os.path.basename(event.get('file', ''))}: {self._runtime_message(event.get('message', ''))}")
            elif kind == 'progress':
                idx = int(event.get('index', 0))
                self.batch_progress['value'] = idx
                self.batch_progress_text.set(f"{idx} / {len(self.imgList)}")
                self.batch_current_text.set(f"{os.path.basename(event.get('file', ''))}: {self._runtime_message(event.get('message', ''))}")
                c = event.get('counts', {})
                self.batch_counts_text.set(self._loc(
                    f"Done: {c.get('done', 0)}    Failed: {c.get('failed', 0)}    Review: {c.get('review', 0)}    Skipped: {c.get('skipped', 0)}",
                    f"完了: {c.get('done', 0)}    失敗: {c.get('failed', 0)}    要確認: {c.get('review', 0)}    スキップ: {c.get('skipped', 0)}"
                ))
            elif kind == 'fatal':
                self.batch_current_text.set(self._loc("Batch initialization failed: ", "バッチ初期化に失敗しました: ") + event.get('message', ''))
            elif kind == 'finished':
                finished = True
                c = event.get('counts', {})
                self.batch_counts_text.set(self._loc(
                    f"Done: {c.get('done', 0)}    Failed: {c.get('failed', 0)}    Review: {c.get('review', 0)}    Skipped: {c.get('skipped', 0)}",
                    f"完了: {c.get('done', 0)}    失敗: {c.get('failed', 0)}    要確認: {c.get('review', 0)}    スキップ: {c.get('skipped', 0)}"
                ))
                if event.get('cancelled'):
                    self.batch_current_text.set(self._loc("Cancelled. Completed results have already been saved.", "キャンセルしました。完了済みの結果は保存されています。"))
                    self.status_var.set(self._loc("Batch processing cancelled", "バッチ処理をキャンセルしました"))
                else:
                    self.batch_current_text.set(self._loc(f"Finished. Log: {event.get('log_path', '')}", f"完了しました。ログ: {event.get('log_path', '')}"))
                    self.status_var.set(self._loc("Batch processing finished", "バッチ処理が完了しました"))
        if finished:
            self.button.configure(state='normal')
            self.buttonPauseBatch.configure(state='disabled', text=self._tr('Pause'))
            self.buttonCancelBatch.configure(state='disabled')
            self.buttonExport.configure(state='normal')
            self.buttonRatio.configure(state='normal')
            self.buttonAutoScale.configure(state='normal')
            self.batch_thread = None
        elif self.batch_thread is not None and self.batch_thread.is_alive():
            self.contWindow.after(100, self._poll_batch_queue)

    def _toggle_batch_pause(self):
        if self.batch_thread is None or not self.batch_thread.is_alive():
            return
        if self.batch_pause_event.is_set():
            self.batch_pause_event.clear()
            self.buttonPauseBatch.configure(text=self._tr('Pause'))
            self.batch_current_text.set(self._loc("Resuming batch processing...", "バッチ処理を再開しています..."))
        else:
            self.batch_pause_event.set()
            self.buttonPauseBatch.configure(text=self._tr('Resume'))
            self.batch_current_text.set(self._loc("Pause requested; processing will pause before the next image.", "一時停止を要求しました。次の画像を処理する前に停止します。"))

    def _cancel_batch(self):
        if self.batch_thread is not None and self.batch_thread.is_alive():
            self.batch_cancel_event.set()
            self.batch_pause_event.clear()
            self.buttonCancelBatch.configure(state='disabled')
            self.batch_current_text.set(self._loc("Cancelling after the current image...", "現在の画像処理後にキャンセルします..."))

    def __BatchExportDialog(self):
        """Dataset-style batch export of full images or per-root ROIs."""
        if hasattr(self, 'proPath') and self.proPath:
            default_root = self.proPath
        elif getattr(self, 'ProjectDir', None):
            default_root = os.path.dirname(self.ProjectDir)
        else:
            default_root = os.getcwd()

        win = tk.Toplevel(self.master)
        win.title(self._tr("Batch export"))
        win.transient(self.master)
        win.grab_set()
        win.resizable(False, False)

        source_var = tk.StringVar(value=default_root)
        output_var = tk.StringVar(value=os.path.join(default_root, "batch_export"))
        scope_var = tk.StringVar(value="roi")
        export_image = tk.BooleanVar(value=True)
        export_overlay = tk.BooleanVar(value=True)
        export_binary = tk.BooleanVar(value=True)
        export_semantic = tk.BooleanVar(value=True)

        ttk.Label(win, text=self._tr("Project root")).grid(row=0, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(win, textvariable=source_var, width=55).grid(row=0, column=1, padx=4)
        ttk.Button(win, text=self._tr("Browse"), command=lambda: source_var.set(
            filedialog.askdirectory(parent=win, initialdir=source_var.get()) or source_var.get()
        )).grid(row=0, column=2, padx=6)

        ttk.Label(win, text=self._tr("Output folder")).grid(row=1, column=0, sticky="w", padx=8, pady=6)
        ttk.Entry(win, textvariable=output_var, width=55).grid(row=1, column=1, padx=4)
        ttk.Button(win, text=self._tr("Browse"), command=lambda: output_var.set(
            filedialog.askdirectory(parent=win, initialdir=os.path.dirname(output_var.get())) or output_var.get()
        )).grid(row=1, column=2, padx=6)

        scope_box = ttk.LabelFrame(win, text=self._tr("Export scope"))
        scope_box.grid(row=2, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Radiobutton(scope_box, text=self._tr("Each ROI"), variable=scope_var, value="roi").pack(side="left", padx=10, pady=5)
        ttk.Radiobutton(scope_box, text=self._tr("Full image"), variable=scope_var, value="full").pack(side="left", padx=10, pady=5)

        out_box = ttk.LabelFrame(win, text=self._tr("Outputs"))
        out_box.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=6)
        ttk.Checkbutton(out_box, text=self._tr("Original image"), variable=export_image).grid(row=0, column=0, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(out_box, text=self._tr("Colored label preview"), variable=export_overlay).grid(row=0, column=1, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(out_box, text=self._tr("Binary root mask"), variable=export_binary).grid(row=1, column=0, sticky="w", padx=10, pady=3)
        ttk.Checkbutton(out_box, text=self._tr("Label mask (0=BG, 1=Leaf, 2=Primary, 3+=Lateral)"), variable=export_semantic).grid(row=1, column=1, sticky="w", padx=10, pady=3)

        ttk.Label(
            win,
            text=self._loc("A manifest.csv with source image and ROI coordinates is always written.", "元画像と ROI 座標を記録した manifest.csv は常に出力されます。"),
            foreground="#666",
        ).grid(row=4, column=0, columnspan=3, sticky="w", padx=8, pady=(2, 8))

        def run_export():
            try:
                count = self._batch_export_dataset(
                    source_var.get(), output_var.get(), scope_var.get(),
                    export_image.get(), export_overlay.get(), export_binary.get(), export_semantic.get()
                )
            except Exception as exc:
                messagebox.showerror(self._tr("Batch export"), str(exc), parent=win)
                return
            messagebox.showinfo(self._tr("Batch export"), self._loc(f"Exported {count} samples.", f"{count} サンプルをエクスポートしました。"), parent=win)
            win.destroy()

        btns = ttk.Frame(win)
        btns.grid(row=5, column=0, columnspan=3, sticky="e", padx=8, pady=8)
        ttk.Button(btns, text=self._tr("Export"), command=run_export).pack(side="left", padx=4)
        ttk.Button(btns, text=self._tr("Cancel"), command=win.destroy).pack(side="left", padx=4)

    @staticmethod
    def _read_cv_unicode(path, flags=cv2.IMREAD_UNCHANGED):
        if not path or not os.path.isfile(path):
            return None
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)

    @staticmethod
    def _write_png_unicode(path, image):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok, buf = cv2.imencode('.png', image)
        if not ok:
            raise IOError(f"Could not encode: {path}")
        buf.tofile(path)

    @staticmethod
    def _resolve_project_file(project_dir, xml_value, fallback_name=None):
        """Resolve a project resource saved on Windows or Linux.

        New projects store only relative XML paths.  For legacy XML files we also
        accept absolute paths.  If an old absolute path no longer exists (for
        example ``C:/old/project/image.jpg`` after copying to Ubuntu), the loader
        falls back to the file with the same basename inside the current project.
        """
        project_dir = os.path.abspath(os.path.normpath(project_dir))
        candidates = []
        raw = str(xml_value or '').strip()
        if raw:
            # Existing path on the current OS (legacy absolute path, for example).
            candidates.append(raw)

            # Relative XML uses '/' deliberately.  Also accept old Windows '\'.
            local_value = raw.replace('\\', os.sep).replace('/', os.sep)
            windows_abs = bool(re.match(r'^[A-Za-z]:[\\/]', raw) or raw.startswith('\\\\'))
            unix_abs = raw.startswith('/')
            if not windows_abs and not unix_abs and not os.path.isabs(local_value):
                candidates.append(os.path.join(project_dir, local_value))

            # Portable recovery for a copied legacy project whose XML contains an
            # obsolete absolute path from another machine/OS.
            basename = re.split(r'[\\/]+', raw)[-1]
            if basename:
                candidates.append(os.path.join(project_dir, basename))

        if fallback_name:
            candidates.append(os.path.join(project_dir, fallback_name))

        seen = set()
        for path in candidates:
            if not path:
                continue
            norm = os.path.normpath(os.path.expanduser(str(path)))
            key = os.path.normcase(norm)
            if key in seen:
                continue
            seen.add(key)
            if os.path.isfile(norm):
                return norm
        return None

    @staticmethod
    def _semantic_preview(image, semantic):
        """Create a coloured preview while preserving lateral-instance colours.

        ``semantic`` uses 1=leaf, 2=primary and 3+ for independent lateral-root
        instances.  The editor already keeps those 3+ ids separate, so do not
        collapse them to one generic lateral colour here.
        """
        if image.ndim == 2:
            base = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            base = image[:, :, :3].copy()
        else:
            base = image.copy()
        if base.dtype != np.uint8:
            lo, hi = float(np.min(base)), float(np.max(base))
            if hi > lo:
                base = ((base - lo) * 255.0 / (hi - lo)).astype(np.uint8)
            else:
                base = np.zeros_like(base, dtype=np.uint8)

        semantic = np.asarray(semantic)
        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        out = base.copy()
        for value in (int(v) for v in np.unique(semantic) if int(v) > 0):
            rtype = value if value <= 2 else 3
            # CLASS_COLORS is RGB (the main canvas is PIL/ImageTk), whereas this
            # batch-export preview is written by OpenCV and therefore needs BGR.
            rgb = MainGUI._color_from_label_value(value, rtype)
            bgr = np.asarray(rgb[::-1], dtype=np.uint8)
            out[semantic == value] = bgr
        return out

    def _discover_project_xmls(self, project_root):
        project_root = os.path.abspath(os.path.normpath(project_root))
        if not os.path.isdir(project_root):
            raise FileNotFoundError(f"Project root does not exist: {project_root}")
        xmls = []
        for path in glob.glob(os.path.join(project_root, '**', '*.xml'), recursive=True):
            try:
                root = ET.parse(path).getroot()
                if root.find('colorimage') is not None:
                    xmls.append(path)
            except Exception:
                continue
        return sorted(set(xmls))

    def _batch_export_dataset(self, project_root, output_root, scope,
                              export_image=True, export_overlay=True,
                              export_binary=True, export_semantic=True):
        xmls = self._discover_project_xmls(project_root)
        if not xmls:
            raise RuntimeError("No project XML files were found under the selected project root.")
        if not any((export_image, export_overlay, export_binary, export_semantic)):
            raise RuntimeError("Select at least one output type.")

        output_root = os.path.abspath(os.path.normpath(output_root))
        subdirs = {
            'image': os.path.join(output_root, 'images'),
            'overlay': os.path.join(output_root, 'color_labels'),
            'binary': os.path.join(output_root, 'masks_binary'),
            'semantic': os.path.join(output_root, 'masks_labels'),
        }
        for enabled, key in [
            (export_image, 'image'), (export_overlay, 'overlay'),
            (export_binary, 'binary'), (export_semantic, 'semantic')
        ]:
            if enabled:
                os.makedirs(subdirs[key], exist_ok=True)

        manifest_rows = []
        exported = 0

        for xml_path in xmls:
            try:
                xroot = ET.parse(xml_path).getroot()
            except Exception:
                continue
            project_dir = os.path.dirname(xml_path)
            base = os.path.splitext(os.path.basename(xml_path))[0]
            scale_node = xroot.find('resolution')
            unit_node = xroot.find('unit')
            try:
                project_resolution = float(scale_node.text) if scale_node is not None and scale_node.text else ''
            except ValueError:
                project_resolution = ''
            project_unit = unit_node.text if unit_node is not None and unit_node.text else ''
            color_node = xroot.find('colorimage')
            color_value = color_node.text if color_node is not None else None
            fallback_image = (re.split(r'[\\/]+', color_value)[-1]
                              if color_value else None)
            color_path = self._resolve_project_file(project_dir, color_value, fallback_image)
            image = self._read_cv_unicode(color_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            rotate_node = xroot.find('bRotate')
            if rotate_node is not None and rotate_node.text == '1':
                direction_node = xroot.find('rotation_direction')
                direction = (direction_node.text.strip().lower()
                             if direction_node is not None and direction_node.text else 'cw')
                rotate_code = (cv2.ROTATE_90_COUNTERCLOCKWISE
                               if direction == 'ccw' else cv2.ROTATE_90_CLOCKWISE)
                image = cv2.rotate(image, rotate_code)

            mask_node = xroot.find('resultmask')
            labels_node = xroot.find('resultlabels')
            binary_path = self._resolve_project_file(
                project_dir, mask_node.text if mask_node is not None else None, 'mask.png')
            semantic_path = self._resolve_project_file(
                project_dir, labels_node.text if labels_node is not None else None, 'mask_labels.png')
            binary = self._read_cv_unicode(binary_path, cv2.IMREAD_GRAYSCALE)
            semantic = self._read_cv_unicode(semantic_path, cv2.IMREAD_UNCHANGED)

            if semantic is None and binary is not None:
                semantic = np.where(binary > 0, 2, 0).astype(np.uint8)
            if binary is None and semantic is not None:
                binary = (semantic >= 2).astype(np.uint8) * 255
            if semantic is None or binary is None:
                continue

            h = min(image.shape[0], semantic.shape[0], binary.shape[0])
            w = min(image.shape[1], semantic.shape[1], binary.shape[1])
            image = image[:h, :w]
            semantic = semantic[:h, :w]
            binary = binary[:h, :w]

            samples = []
            if scope == 'full':
                samples.append((f"{base}", 0, 0, w, h, image, binary, semantic, ''))
            else:
                for roi_idx, roi in enumerate(xroot.findall('Roi'), start=1):
                    item = roi.find('item')
                    if item is None:
                        continue
                    try:
                        x0 = int(item.find('ltx').text)
                        y0 = int(item.find('lty').text)
                        x1 = int(item.find('rbx').text)
                        y1 = int(item.find('rby').text)
                    except Exception:
                        continue
                    x0, x1 = sorted((max(0, x0), min(w, x1)))
                    y0, y1 = sorted((max(0, y0), min(h, y1)))
                    if x1 <= x0 or y1 <= y0:
                        continue
                    name = f"{base}_roi_{roi_idx:03d}"
                    samples.append((
                        name, x0, y0, x1, y1,
                        image[y0:y1, x0:x1],
                        binary[y0:y1, x0:x1],
                        semantic[y0:y1, x0:x1],
                        str(roi_idx),
                    ))

            for name, x0, y0, x1, y1, img_crop, bin_crop, sem_crop, roi_id in samples:
                image_rel = overlay_rel = binary_rel = semantic_rel = ''
                if export_image:
                    image_rel = os.path.join('images', name + '.png')
                    self._write_png_unicode(os.path.join(output_root, image_rel), img_crop)
                if export_overlay:
                    overlay_rel = os.path.join('color_labels', name + '.png')
                    preview = self._semantic_preview(img_crop, sem_crop)
                    self._write_png_unicode(os.path.join(output_root, overlay_rel), preview)
                if export_binary:
                    binary_rel = os.path.join('masks_binary', name + '.png')
                    self._write_png_unicode(
                        os.path.join(output_root, binary_rel), np.where(bin_crop > 0, 255, 0).astype(np.uint8))
                if export_semantic:
                    semantic_rel = os.path.join('masks_labels', name + '.png')
                    self._write_png_unicode(
                        os.path.join(output_root, semantic_rel), np.asarray(sem_crop, dtype=np.uint8))

                manifest_rows.append([
                    base, color_path or '', scope, roi_id, x0, y0, x1, y1,
                    project_resolution, project_unit,
                    image_rel, overlay_rel, binary_rel, semantic_rel,
                ])
                exported += 1

        os.makedirs(output_root, exist_ok=True)
        manifest_path = os.path.join(output_root, 'manifest.csv')
        with open(manifest_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                'project', 'source_image', 'scope', 'roi_index',
                'x0', 'y0', 'x1', 'y1', 'resolution_per_pixel', 'unit',
                'image_file', 'color_label_file', 'binary_mask_file', 'semantic_mask_file'
            ])
            writer.writerows(manifest_rows)
        self.status_var.set(self._loc(f"Batch export: {exported} samples -> {output_root}", f"バッチエクスポート: {exported} サンプル -> {output_root}"))
        return exported

    def show_selected_folder(self):
        """Create a single-image project, warning before an existing project is overwritten."""
        pro_raw = self.entry1.get().strip()
        img_raw = self.entry2.get().strip()

        if not pro_raw:
            messagebox.showwarning(
                self._tr("Create project"), self._loc("Please select a project folder.", "プロジェクトフォルダーを選択してください。"), parent=self.new_window
            )
            return
        if not img_raw:
            messagebox.showwarning(
                self._tr("Create project"), self._loc("Please select an image file.", "画像ファイルを選択してください。"), parent=self.new_window
            )
            return

        proPath = os.path.normpath(pro_raw)
        imgpath = os.path.normpath(img_raw)

        # Validate the source image before touching any existing project folder.
        if not os.path.isfile(imgpath):
            messagebox.showinfo(
                self._loc("Please check this image", "画像を確認してください"),
                self._loc('Check the path : "{}" or file content please.'.format(imgpath), 'パスまたは画像ファイルの内容を確認してください: "{}"'.format(imgpath)),
                parent=self.new_window,
            )
            return
        try:
            with Image.open(imgpath) as img:
                img.verify()
        except Exception:
            messagebox.showinfo(
                self._loc("Please check this image", "画像を確認してください"),
                self._loc('Check the path : "{}" or file content please.'.format(imgpath), 'パスまたは画像ファイルの内容を確認してください: "{}"'.format(imgpath)),
                parent=self.new_window,
            )
            return

        file_name = os.path.splitext(os.path.basename(imgpath))[0]
        project_dir = os.path.join(proPath, file_name)

        # Each project is stored in a subfolder named after the source image.
        # If the target already exists, require an explicit overwrite decision.
        if os.path.isdir(project_dir):
            overwrite = messagebox.askyesno(
                self._loc("Project already exists", "プロジェクトはすでに存在します"),
                self._loc(
                    f"A project folder with the same name already exists:\n\n{project_dir}\n\nOverwrite it? Existing files in this project folder will be deleted.",
                    f"同名のプロジェクトフォルダーがすでに存在します:\n\n{project_dir}\n\n上書きしますか？ このプロジェクトフォルダー内の既存ファイルは削除されます。"
                ),
                parent=self.new_window,
            )
            if not overwrite:
                self.status_var.set(self._loc("Project creation cancelled", "プロジェクト作成をキャンセルしました"))
                return

            # Protect the unusual case where the selected source image itself
            # is located inside the project folder that is about to be replaced.
            source_inside_project = False
            try:
                source_inside_project = (
                    os.path.commonpath(
                        [os.path.abspath(imgpath), os.path.abspath(project_dir)]
                    )
                    == os.path.abspath(project_dir)
                )
            except ValueError:
                source_inside_project = False

            cached_source = None
            if source_inside_project:
                with open(imgpath, "rb") as f:
                    cached_source = f.read()

            try:
                shutil.rmtree(project_dir)
                os.makedirs(project_dir, exist_ok=False)
                dst_image = os.path.join(project_dir, os.path.basename(imgpath))
                if cached_source is not None:
                    with open(dst_image, "wb") as f:
                        f.write(cached_source)
                else:
                    shutil.copy2(imgpath, dst_image)
            except Exception as exc:
                messagebox.showerror(
                    self._tr("Create project"),
                    self._loc(f"Could not overwrite the existing project folder:\n{exc}", f"既存のプロジェクトフォルダーを上書きできませんでした:\n{exc}"),
                    parent=self.new_window,
                )
                return
        else:
            try:
                os.makedirs(project_dir, exist_ok=False)
                dst_image = os.path.join(project_dir, os.path.basename(imgpath))
                shutil.copy2(imgpath, dst_image)
            except Exception as exc:
                messagebox.showerror(
                    self._tr("Create project"),
                    self._loc(f"Could not create the project folder:\n{exc}", f"プロジェクトフォルダーを作成できませんでした:\n{exc}"),
                    parent=self.new_window,
                )
                return

        # From this point on, use the project-local copy of the source image.
        self.imgpath = dst_image
        self.file_name = file_name
        self.ProjectDir = project_dir

        try:
            pillimg = Image.open(self.imgpath)
        except Exception as exc:
            messagebox.showerror(
                self._tr("Create project"),
                self._loc(f"The project was created, but the copied image could not be opened:\n{exc}", f"プロジェクトは作成されましたが、コピーした画像を開けませんでした:\n{exc}"),
                parent=self.new_window,
            )
            return

        self.new_window.grab_release()
        self.new_window.destroy()

        self.master.title(self.__default_title + ': {}'.format(self.imgpath))
        self.__config.set_recent_path(
            os.path.join(os.path.normpath(self.ProjectDir), self.file_name + ".xml")
        )
        self.__set_image(pillimg, self.imgpath, 1, 1, None)
        self.__imframe.SetSelectMode(self.mSelectMode)
        self.__image_menu.entryconfigure(self.__index_close, state='normal')
        self.__config.set_opened_path(
            os.path.join(os.path.normpath(self.ProjectDir), self.file_name + ".xml")
        )
        self.__config.save()
        self._remember_recent_folder("project_create", proPath)
        self._remember_recent_folder("image_source", os.path.dirname(imgpath))
        self._remember_recent_folder("project_open", self.ProjectDir)
        self.status_var.set(self._loc("Successfully created project from image %s" % self.imgpath, "画像からプロジェクトを作成しました: %s" % self.imgpath))

    def _recent_folder_store_path(self):
        """Path used for persistent recent-folder history."""
        return os.path.join(os.path.expanduser("~"), ".rapid_recent_folders.json")

    def _load_recent_folders(self, key):
        try:
            with open(self._recent_folder_store_path(), "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get(str(key), []) if isinstance(data, dict) else []
        except Exception:
            items = []
        result = []
        for p in items:
            p = os.path.normpath(os.path.expanduser(str(p)))
            if os.path.isdir(p) and p not in result:
                result.append(p)
        return result[:10]

    def _remember_recent_folder(self, key, path):
        if not path:
            return
        path = os.path.abspath(os.path.normpath(os.path.expanduser(str(path))))
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if not os.path.isdir(path):
            return
        store = self._recent_folder_store_path()
        try:
            with open(store, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        items = [p for p in data.get(str(key), []) if os.path.normcase(p) != os.path.normcase(path)]
        data[str(key)] = [path] + items[:9]
        try:
            with open(store, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _initial_history_folder(self, key, current=""):
        """Return the best existing folder to use as a native dialog start path."""
        current = os.path.abspath(os.path.expanduser(current.strip())) if current else ""
        if current and os.path.isdir(current):
            return current
        recent = self._load_recent_folders(key)
        return recent[0] if recent else os.getcwd()

    def _askopenfilename_from_history(self, key, parent=None, title="Open file", filetypes=None):
        # Do not open an extra RAPID window.  The history is used only to choose
        # the starting directory of the normal operating-system file dialog.
        folder = self._initial_history_folder(key)
        path = filedialog.askopenfilename(
            parent=parent,
            title=title,
            initialdir=folder,
            filetypes=filetypes or [("All files", "*.*")],
        )
        if path:
            self._remember_recent_folder(key, os.path.dirname(path))
        return path

    def _active_dialog_parent(self):
        """Return a live Tk window for native file/folder dialogs.

        Several RAPID dialogs reuse update_var/update_varfolder.  A previously
        destroyed ``self.new_window`` may still exist as a Python attribute;
        passing that stale widget to filedialog causes TclError: bad window
        path name.  Prefer the currently open batch-folder window, then the
        new-project window, and finally the application root.
        """
        for name in ("new_windowFolder", "new_window"):
            win = getattr(self, name, None)
            if win is None:
                continue
            try:
                if int(win.winfo_exists()):
                    return win
            except (tk.TclError, AttributeError):
                pass
        return self.master

    def update_var(self, text_var, history_key="image_source"):
        current = text_var.get().strip()
        initial = os.path.dirname(current) if current and os.path.isfile(current) else ""
        path = filedialog.askopenfilename(
            parent=self._active_dialog_parent(),
            title="Select image",
            initialdir=self._initial_history_folder(history_key, initial),
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            text_var.set(path)
            self._remember_recent_folder(history_key, os.path.dirname(path))

    def update_varfolder(self, text_var, history_key="folder"):
        # Use a live parent window.  This is important for the batch-processing
        # dialog, which uses ``new_windowFolder`` rather than ``new_window``.
        path = filedialog.askdirectory(
            parent=self._active_dialog_parent(),
            title="Select folder",
            initialdir=self._initial_history_folder(history_key, text_var.get()),
        )
        if path:
            text_var.set(path)
            self._remember_recent_folder(history_key, path)

    def _apply_recent_folder(self, text_var, value):
        value = str(value).strip()
        if value and os.path.isdir(value):
            text_var.set(value)

    def Create_New_Project(self,root):
        self.new_window = tk.Toplevel(root)
        self.new_window.grab_set()
        self._set_window_icon(self.new_window)
        self.text_Pro = tk.StringVar(self.new_window)
        self.text_Img = tk.StringVar(self.new_window)

        # Project folder row.  Recent folders are inline: no extra window.
        tk.Label(self.new_window, text=self._tr("Project folder")).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.entry1 = tk.Entry(self.new_window, textvariable=self.text_Pro, width=55)
        recent_project_dirs = self._load_recent_folders("project_create")
        current_directory = recent_project_dirs[0] if recent_project_dirs else os.getcwd()
        self.text_Pro.set(current_directory)
        self.entry1.grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(
            self.new_window, text=self._tr("Select folder"),
            command=lambda: self.update_varfolder(self.text_Pro, "project_create")
        ).grid(row=0, column=2, padx=4, pady=4)

        self.recent_project_combo = ttk.Combobox(
            self.new_window, values=recent_project_dirs, state="readonly", width=30
        )
        if recent_project_dirs:
            self.recent_project_combo.set(recent_project_dirs[0])
        self.recent_project_combo.grid(row=0, column=3, padx=(2, 6), pady=4, sticky="ew")
        self.recent_project_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._apply_recent_folder(self.text_Pro, self.recent_project_combo.get())
        )

        # Image file row.  Keep the normal file chooser, but add an inline
        # recent-folder selector beside it so the user can change its start folder.
        tk.Label(self.new_window, text=self._tr("Image file")).grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.entry2 = tk.Entry(self.new_window, textvariable=self.text_Img, width=55)
        self.entry2.grid(row=1, column=1, padx=4, pady=4, sticky="ew")
        tk.Button(
            self.new_window, text=self._tr("Select image"),
            command=lambda: self.update_var(self.text_Img, "image_source")
        ).grid(row=1, column=2, padx=4, pady=4)

        recent_image_dirs = self._load_recent_folders("image_source")
        self.recent_image_combo = ttk.Combobox(
            self.new_window, values=recent_image_dirs, state="readonly", width=30
        )
        if recent_image_dirs:
            self.recent_image_combo.set(recent_image_dirs[0])
        self.recent_image_combo.grid(row=1, column=3, padx=(2, 6), pady=4, sticky="ew")

        def choose_image_from_recent(_event=None):
            folder = self.recent_image_combo.get().strip()
            if not os.path.isdir(folder):
                return
            path = filedialog.askopenfilename(
                parent=self.new_window, title="Select image", initialdir=folder,
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.tif *.tiff"), ("All files", "*.*")],
            )
            if path:
                self.text_Img.set(path)
                self._remember_recent_folder("image_source", os.path.dirname(path))

        self.recent_image_combo.bind("<<ComboboxSelected>>", choose_image_from_recent)

        tk.Button(self.new_window, text=self._tr("OK"), command=self.show_selected_folder).grid(row=2, column=1, pady=6)
        tk.Button(self.new_window, text=self._tr("Cancel"), command=self.new_window.destroy).grid(row=2, column=2, pady=6)
        self.new_window.columnconfigure(1, weight=1)
        self.new_window.columnconfigure(3, weight=1)

    def __SetRotation(self, direction=None):
        """Rotate the working image by 90 degrees using the selected orientation.

        ``direction`` is ``"cw"`` or ``"ccw"``.  Older projects did not store
        the direction and therefore fall back to clockwise rotation.
        """
        if direction is None:
            direction = getattr(self, 'rotation_direction', 'cw')
        direction = str(direction or 'cw').lower()
        if direction not in ('cw', 'ccw'):
            direction = 'cw'

        rotate_code = (cv2.ROTATE_90_COUNTERCLOCKWISE
                       if direction == 'ccw' else cv2.ROTATE_90_CLOCKWISE)
        self.curImage = cv2.rotate(self.curImage, rotate_code)
        self.rotation_direction = direction
        self.mResultImg = self.curImage.copy()

        tmp = Image.fromarray(self.curImage)

        self.__close_image()  # close previous image

        self.__imframe = ImageFrame(placeholder=self.__placeholder,
                                    roi_size=self.__config.get_roi_size(), curImg=tmp,
                                    bSelect=1, ProjectDir=self.ProjectDir, label=None,
                                    aroot=self.RootAll)
       
    def __HelpDialog(self):
        self.newWindowH = tk.Toplevel(self.master)
        self.newWindowH.title("ヘルプ - RAPID" if self.ui_language == "ja" else "Help - RAPID")
        self.newWindowH.geometry("500x650")
        self.newWindowH.resizable(False, False)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self._set_window_icon(self.newWindowH)
        
        frame = ttk.Frame(self.newWindowH, padding=20)
        frame.pack(fill="both", expand=True)

        if self.ui_language == "ja":
            help_text = """本ソフトウェアの主な機能
■ プロジェクトを作成する
「ファイル」→「プロジェクトを作成」から、プロジェクトフォルダーと処理する画像を選択します。JPEG、PNG、TIFF などの画像を利用できます。

■ 画像を処理する
「ツール」から AI 処理または他の処理方法を選択すると、根の検出、領域分割、形態パラメータの計測を実行します。

■ スケールを設定する
「表示」→「スケール校正」では手動で実距離とピクセル距離を設定できます。「定規スケールを検出」では画像内の定規から候補を確認できます。

■ 結果を修正する
処理済み ROI をダブルクリックするか、「ツール」→「根アノテーションエディターを開く」から、葉・主根・側根のラベルを編集できます。

■ 保存と出力
プロジェクト結果は保存され、バッチエクスポートから画像、ラベル、二値マスクなどを出力できます。

■ 注意
根や定規が鮮明に写った高解像度画像を推奨します。自動結果は必要に応じてアノテーションエディターで確認してください。
"""
        else:
            help_text = """Main functions
■ Create a project
Choose File → Create project, then select a project folder and an image to process. JPEG, PNG, and TIFF images are supported.

■ Process an image
Choose an AI or Other processing methods from Tool to detect roots, segment regions, and calculate morphological measurements.

■ Set the physical scale
Use View → Scale calibration for manual distance calibration, or Detect ruler scale to review scale candidates detected from a ruler in the image.

■ Correct results
Double-click a processed ROI, or choose Tool → Open root annotation editor, to edit leaf, primary-root, and lateral-root labels.

■ Save and export
Project results are saved with the project. Batch export can write images, label masks, and binary masks for downstream analysis.

■ Note
Use high-resolution images with clearly visible roots and rulers. Review automatic results in the annotation editor when needed.
"""

        help_label = tk.Label(frame, text=help_text, justify="left", wraplength=450, anchor="nw")
        help_label.pack(fill="both", expand=True)

        close_button = ttk.Button(frame, text=("閉じる" if self.ui_language == "ja" else "Close"), command=self.newWindowH.destroy)
        close_button.pack(pady=(10, 0))
    def __AboutDialog(self):
        
        #title = "About software"
        #self.newWindowA = tk.Toplevel(self.master)
        #self.newWindowA.title(title)
        #self.newWindowA.geometry("{}x{}".format(400,600))
        #labelPhy = tk.Label(self.newWindowA, text = "This verion is created at 2024.04.15.")
        #labelPhy.grid(row=0, column=0)
        self.aboutWindowH = tk.Toplevel(self.master)
        self.aboutWindowH.title("RAPIDについて" if self.ui_language == "ja" else "About RAPID")
        self.aboutWindowH.geometry("400x600")
        self.aboutWindowH.resizable(False, False)
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self._set_window_icon(self.aboutWindowH)
        
        frame = ttk.Frame(self.aboutWindowH, padding=20)
        frame.pack(fill="both", expand=True)

        logo_image = Image.open(os.path.join(cur_dir, "root.ico"))
        logo_image = logo_image.resize((64, 64))  # 你可以根据需要调整尺寸
        logo = ImageTk.PhotoImage(logo_image)
        logo_label = ttk.Label(frame, image=logo)
        logo_label.image = logo  # 保存引用
        logo_label.pack(pady=(0, 10))

        # 软件名称
        app_name = ttk.Label(frame, text="RAPID", font=("Helvetica", 18, "bold"))
        app_name.pack(pady=(0, 10))

        # 版本信息
        version = ttk.Label(frame, text=("バージョン: v1.0.0" if self.ui_language == "ja" else "Version: v1.0.0"))
        version.pack()

        # 作者信息
        author = ttk.Label(frame, text=("開発: 名古屋大学大学院情報学研究科 森研究室" if self.ui_language == "ja" else "Developed by Mori Laboratory, Graduate School of Informatics, Nagoya University"), wraplength=350, justify="center")
        author.pack()
        description_text = ("""本ソフトウェアは、スキャン画像から植物の根の形態パラメータを取得するためのツールです。
根の自動計測、スケール校正、結果確認、アノテーション編集、バッチ出力の機能を備えています。
詳しい操作方法については同梱のマニュアルをご確認ください。
本ソフトウェアは Moonshot Goal 3 の支援を受けています。""" if self.ui_language == "ja" else """This software is designed to measure morphological parameters of plant roots from scanned images.
It includes automatic root measurement, scale calibration, result review, annotation editing, and batch export tools.
Please refer to the included manual for detailed operating instructions.
This software is supported by Moonshot Goal 3.""")
        # 简短说明
        description = ttk.Label(frame, text=description_text, wraplength=350, justify="center")
        description.pack(pady=(10, 20))

        # 超链接按钮
        link_button = ttk.Button(frame, text=("ホームページ" if self.ui_language == "ja" else "Website"), command=lambda: self.open_link("http://newves.org/wiki/"))
        link_button.pack(pady=(0, 20))

        # 版权信息
        copyright_label = ttk.Label(
            frame,
            text="©Nagoya University. All rights reserved.",
            font=("Helvetica", 8),
            foreground="gray"
        )
        copyright_label.pack(side="bottom", pady=(20, 0))

        # 关闭按钮
        close_button = ttk.Button(frame, text=("閉じる" if self.ui_language == "ja" else "Close"), command=self.aboutWindowH.destroy)
        close_button.pack(side="bottom", pady=(10, 0))

    def open_link(self, url):
        import webbrowser
        webbrowser.open(url)
    def __UpDown_toggle(self):
        self.curImage
    def _refresh_batch_scale_info(self):
        if hasattr(self, 'batch_scale_info') and self.batch_scale_info.winfo_exists():
            self.batch_scale_info.configure(
                text=self._loc(
                    f"Current shared scale: {getattr(self, 'resolution', 1.0):.8g} {getattr(self, 'unit', 'px')}/pixel; source={getattr(self, 'scale_source', 'default')}",
                    f"現在の共通スケール: {getattr(self, 'resolution', 1.0):.8g} {getattr(self, 'unit', 'px')}/pixel; ソース={getattr(self, 'scale_source', 'default')}"
                ))

    def __ScaleSetted(self):
        """Apply a manual physical scale: known physical distance / pixel distance."""
        try:
            physical = float(self.txtPhy.get())
            pixels = float(self.txtPixel.get())
            if physical <= 0 or pixels <= 0:
                raise ValueError
        except (TypeError, ValueError):
            messagebox.showerror(self._tr("Scale calibration"), self._loc("Known distance and pixel distance must both be positive numbers.", "既知の実距離とピクセル距離は、どちらも 0 より大きい数値で指定してください。"))
            return

        self.resolution = physical / pixels
        self.unit = self.cb.get() or 'mm'
        self.bSetResolution = True
        self.scale_source = 'manual'
        self.scale_confidence = 1.0
        self.status_var.set(self._loc(
            f"Scale: {self.resolution:.8g} {self.unit}/pixel ({pixels:.2f} px = {physical:g} {self.unit})",
            f"スケール: {self.resolution:.8g} {self.unit}/pixel ({pixels:.2f} px = {physical:g} {self.unit})"
        ))
        self._refresh_batch_scale_info()
        self.scaleWindow.destroy()

    def __SetResolution(self):
        """Manual scale calibration dialog."""
        self.scaleWindow = tk.Toplevel(self.master)
        self.scaleWindow.title(self._tr("Scale calibration"))
        self._set_window_icon(self.scaleWindow)
        self.scaleWindow.resizable(False, False)

        frame = ttk.Frame(self.scaleWindow, padding=12)
        frame.grid(row=0, column=0, sticky='nsew')

        ttk.Label(frame, text=self._tr("Known physical distance:")).grid(row=0, column=0, sticky='w', pady=4)
        self.txtPhy = ttk.Entry(frame, width=18)
        self.txtPhy.insert(0, "10")
        self.txtPhy.grid(row=0, column=1, sticky='ew', pady=4)

        ttk.Label(frame, text=self._tr("Pixel distance:")).grid(row=1, column=0, sticky='w', pady=4)
        self.txtPixel = ttk.Entry(frame, width=18)
        self.txtPixel.insert(0, "2360")
        self.txtPixel.grid(row=1, column=1, sticky='ew', pady=4)

        ttk.Label(frame, text=self._tr("Unit of length:")).grid(row=2, column=0, sticky='w', pady=4)
        v = tk.StringVar(value=getattr(self, 'unit', 'mm'))
        self.cb = ttk.Combobox(frame, textvariable=v, values=['mm', 'cm', 'um'], width=15, state='readonly')
        if v.get() not in ('mm', 'cm', 'um'):
            v.set('mm')
        self.cb.grid(row=2, column=1, sticky='ew', pady=4)

        note = self._loc(
            "Resolution = known distance / pixel distance.\nExample: a 10 mm segment spanning 2360 pixels.",
            "解像度 = 既知の実距離 / ピクセル距離。\n例: 10 mm の区間が 2360 ピクセルの場合。"
        )
        ttk.Label(frame, text=note, foreground='gray').grid(row=3, column=0, columnspan=2, sticky='w', pady=(6, 10))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=2, sticky='e')
        ttk.Button(buttons, text=self._tr("OK"), command=self.__ScaleSetted).pack(side='left', padx=4)
        ttk.Button(buttons, text=self._tr("Cancel"), command=self.scaleWindow.destroy).pack(side='left', padx=4)

    @staticmethod
    def _dedupe_tick_positions(values, tolerance=3.0):
        values = sorted(float(v) for v in values)
        if not values:
            return []
        groups = [[values[0]]]
        for value in values[1:]:
            if value - groups[-1][-1] <= tolerance:
                groups[-1].append(value)
            else:
                groups.append([value])
        return [float(np.median(group)) for group in groups]

    @staticmethod
    def _dominant_tick_length_group(lengths, rel_tol=0.18, abs_tol=2.0):
        """Return the largest group of ruler-tick segments with similar lengths.

        Hough detection may return both real ruler ticks and unrelated edge segments.
        A real ruler edge normally contains many tick marks of a repeated length, so
        the size of the dominant length group is useful for ranking edge candidates.
        """
        vals = np.asarray([float(v) for v in lengths if np.isfinite(v) and float(v) > 0],
                          dtype=float)
        if vals.size == 0:
            return 0, 0.0

        best_count = 0
        best_center = 0.0
        best_spread = float('inf')
        for center in vals:
            tol = max(float(abs_tol), float(rel_tol) * float(center))
            members = vals[np.abs(vals - center) <= tol]
            if members.size == 0:
                continue
            med = float(np.median(members))
            spread = float(np.std(members) / med) if members.size > 1 and med > 0 else 0.0
            if (members.size > best_count or
                    (members.size == best_count and spread < best_spread)):
                best_count = int(members.size)
                best_center = med
                best_spread = spread
        return best_count, best_center

    @staticmethod
    def _estimate_tick_spacing(positions):
        """Estimate the fundamental spacing of a regularly spaced 1-D tick sequence."""
        pos = np.asarray(sorted(positions), dtype=float)
        if pos.size < 5:
            return None
        gaps = np.diff(pos)
        gaps = gaps[(gaps >= 3.0) & np.isfinite(gaps)]
        if gaps.size < 4:
            return None

        candidate_values = []
        for gap in gaps:
            for divisor in range(1, 5):
                spacing = gap / divisor
                if spacing >= 3.0:
                    candidate_values.append(spacing)
        if not candidate_values:
            return None

        best = None
        for spacing in candidate_values:
            multiples = np.rint(gaps / spacing)
            valid_mult = (multiples >= 1) & (multiples <= 6)
            if not np.any(valid_mult):
                continue
            rel_resid = np.abs(gaps / spacing - multiples)
            fit = valid_mult & (rel_resid <= 0.18)
            direct = np.abs(gaps - spacing) <= 0.22 * spacing
            direct_count = int(np.count_nonzero(direct))
            if direct_count < 2:
                continue
            fit_fraction = float(np.count_nonzero(fit)) / float(gaps.size)
            direct_fraction = float(direct_count) / float(gaps.size)
            direct_gaps = gaps[direct]
            cv = float(np.std(direct_gaps) / np.mean(direct_gaps)) if direct_gaps.size > 1 else 0.0
            score = 0.62 * fit_fraction + 0.38 * min(1.0, direct_fraction * 2.0)
            score *= max(0.0, 1.0 - min(cv, 0.5))
            if best is None or score > best[0]:
                best = (score, float(np.median(direct_gaps)), fit_fraction, direct_count)

        if best is None:
            return None
        score, spacing, fit_fraction, direct_count = best
        tick_factor = min(1.0, max(0.0, (pos.size - 4) / 8.0))
        confidence = max(0.0, min(1.0, score * (0.65 + 0.35 * tick_factor)))
        return {
            'spacing': spacing,
            'confidence': confidence,
            'fit_fraction': fit_fraction,
            'direct_gaps': direct_count,
            'tick_count': int(pos.size),
        }

    def _find_ruler_tick_candidates(self, image=None):
        """Find regular ruler tick marks near image edges using Hough line segments.

        Besides the spacing estimate, each candidate keeps the actual detected line
        segments (in original-image coordinates).  They are used by the visual
        verification dialog so the user can inspect what the algorithm believed
        were ruler ticks before applying a physical scale.
        """
        arr = self.curImage if image is None else image
        if arr is None:
            return []
        arr = np.asarray(arr)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            else:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            gray = arr.copy()
        if gray.dtype != np.uint8:
            gmin, gmax = float(np.nanmin(gray)), float(np.nanmax(gray))
            if gmax > gmin:
                gray = np.clip((gray - gmin) * 255.0 / (gmax - gmin), 0, 255).astype(np.uint8)
            else:
                gray = np.zeros(gray.shape, dtype=np.uint8)

        h0, w0 = gray.shape[:2]
        if min(h0, w0) < 80:
            return []
        resize_scale = min(1.0, 1800.0 / float(max(h0, w0)))
        if resize_scale < 1.0:
            gray = cv2.resize(gray, (max(1, int(round(w0 * resize_scale))),
                                     max(1, int(round(h0 * resize_scale)))), interpolation=cv2.INTER_AREA)
        h, w = gray.shape[:2]
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        med = float(np.median(gray))
        lower = int(max(10, 0.66 * med))
        upper = int(min(255, max(lower + 20, 1.33 * med)))
        edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)

        band_h = max(50, min(h // 3, int(round(h * 0.24))))
        band_w = max(50, min(w // 3, int(round(w * 0.24))))
        bands = [
            ('top', 'horizontal', edges[:band_h, :], (0, 0)),
            ('bottom', 'horizontal', edges[h-band_h:, :], (0, h-band_h)),
            ('left', 'vertical', edges[:, :band_w], (0, 0)),
            ('right', 'vertical', edges[:, w-band_w:], (w-band_w, 0)),
        ]
        results = []
        for location, orientation, band, (off_x, off_y) in bands:
            bh, bw = band.shape[:2]
            cross = bh if orientation == 'horizontal' else bw
            min_len = max(7, int(round(cross * 0.06)))
            threshold = max(10, int(round(min_len * 0.7)))
            lines = cv2.HoughLinesP(
                band, 1, np.pi / 180.0, threshold=threshold,
                minLineLength=min_len, maxLineGap=max(2, int(round(min_len * 0.20)))
            )
            if lines is None:
                continue

            line_records = []
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = [float(v) for v in line]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                if orientation == 'horizontal':
                    # Horizontal ruler -> approximately vertical tick lines.
                    if abs(dy) < 1.7 * max(abs(dx), 1.0):
                        continue
                    if length > cross * 0.98:
                        continue
                    pos = (x1 + x2) / 2.0
                else:
                    # Vertical ruler -> approximately horizontal tick lines.
                    if abs(dx) < 1.7 * max(abs(dy), 1.0):
                        continue
                    if length > cross * 0.98:
                        continue
                    pos = (y1 + y2) / 2.0
                line_records.append((pos, (x1, y1, x2, y2), length))

            # First identify the dominant Hough-line length group.  Only this group
            # is allowed to determine ruler spacing.  Previously the dominant-length
            # count was used only to rank edge bands, while spacing was still estimated
            # from *all* detected lines.  Mixed major/minor ticks or unrelated edges
            # could therefore produce a visually plausible green set but a wrong scale.
            raw_lengths = [r[2] for r in line_records]
            raw_same_count, dominant_tick_length_scaled = self._dominant_tick_length_group(
                raw_lengths
            )
            if raw_same_count < 5 or dominant_tick_length_scaled <= 0:
                continue

            length_tol = max(2.0, 0.18 * float(dominant_tick_length_scaled))
            dominant_records = [
                r for r in line_records
                if abs(float(r[2]) - float(dominant_tick_length_scaled)) <= length_tol
            ]

            # De-duplicate physical tick positions within the dominant same-length
            # group, then estimate spacing from these positions only.
            positions = [r[0] for r in dominant_records]
            deduped = self._dedupe_tick_positions(
                positions, tolerance=max(2.0, 3.0 * resize_scale)
            )
            estimate = self._estimate_tick_spacing(deduped)
            if estimate is None:
                continue

            # Pick one representative Hough segment for each de-duplicated dominant
            # tick.  The green segments in the preview now correspond exactly to the
            # tick family used for scale calculation.
            tol = max(3.0, 4.0 * resize_scale)
            selected_lines = []
            selected_lengths = []
            for dp in deduped:
                nearby = [r for r in dominant_records if abs(r[0] - dp) <= tol]
                if not nearby:
                    nearby = sorted(dominant_records, key=lambda r: abs(r[0] - dp))[:1]
                if nearby:
                    rec = max(nearby, key=lambda r: r[2])
                    x1, y1, x2, y2 = rec[1]
                    selected_lines.append((
                        (x1 + off_x) / resize_scale,
                        (y1 + off_y) / resize_scale,
                        (x2 + off_x) / resize_scale,
                        (y2 + off_y) / resize_scale,
                    ))
                    selected_lengths.append(float(rec[2]) / resize_scale)

            # Count distinct physical ticks in the dominant same-length family rather
            # than raw Hough segments, which may contain duplicates for one tick.
            same_length_count = len(deduped)
            dominant_tick_length = float(np.median(selected_lengths)) if selected_lengths else (
                float(dominant_tick_length_scaled) / resize_scale
            )

            clutter_penalty = 1.0
            axis_len = w if orientation == 'horizontal' else h
            if len(deduped) > max(80, axis_len // 4):
                clutter_penalty = 0.65
            estimate['confidence'] *= clutter_penalty
            estimate['spacing'] /= resize_scale

            if location == 'top':
                band_rect = (0.0, 0.0, float(w0 - 1), float(band_h / resize_scale))
            elif location == 'bottom':
                band_rect = (0.0, float((h - band_h) / resize_scale), float(w0 - 1), float(h0 - 1))
            elif location == 'left':
                band_rect = (0.0, 0.0, float(band_w / resize_scale), float(h0 - 1))
            else:
                band_rect = (float((w - band_w) / resize_scale), 0.0, float(w0 - 1), float(h0 - 1))

            estimate.update({
                'location': location,
                'orientation': orientation,
                'positions': [p / resize_scale for p in deduped],
                'tick_lines': selected_lines,
                'same_length_count': int(same_length_count),
                'dominant_tick_length': float(dominant_tick_length),
                'band_rect': band_rect,
            })
            results.append(estimate)

        # Candidate 1 is the default choice in both the interactive dialog and
        # automatic batch calibration.  Rank primarily by the number of detected
        # tick segments sharing a similar length, then by spacing confidence.
        results.sort(
            key=lambda item: (
                int(item.get('same_length_count', 0)),
                float(item.get('confidence', 0.0)),
                int(item.get('tick_count', 0)),
            ),
            reverse=True,
        )
        return results

    def _make_ruler_preview(self, candidate, max_size=(920, 560)):
        """Return a PhotoImage-ready PIL preview with detected ruler evidence drawn on it."""
        arr = np.asarray(self.curImage)
        if arr.ndim == 2:
            rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        elif arr.shape[2] == 4:
            rgb = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2RGB)
        else:
            rgb = arr[:, :, :3].astype(np.uint8).copy()

        preview = Image.fromarray(rgb)
        draw = ImageDraw.Draw(preview)
        w, h = preview.size
        stroke = max(2, int(round(max(w, h) / 900.0)))

        # Yellow: the edge search band used for this candidate.
        x0, y0, x1, y1 = candidate.get('band_rect', (0, 0, w - 1, h - 1))
        draw.rectangle((x0, y0, x1, y1), outline=(255, 215, 0), width=max(2, stroke))

        # Green: actual Hough segments selected as ruler ticks.
        tick_lines = candidate.get('tick_lines', [])
        for lx1, ly1, lx2, ly2 in tick_lines:
            draw.line((lx1, ly1, lx2, ly2), fill=(0, 255, 80), width=max(3, stroke + 1))
            r = max(2, stroke + 1)
            cx, cy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=(0, 255, 80))

        # Red: show the actual adjacent gaps that support the estimated spacing.
        # Drawing several supported intervals is much easier to verify than drawing
        # a single tiny interval, which can collapse visually into one red dot after
        # the preview is downscaled.
        positions = sorted(float(v) for v in candidate.get('positions', []))
        spacing = float(candidate.get('spacing', 0.0))
        supported_pairs = []
        if len(positions) >= 2 and spacing > 0:
            for i in range(len(positions) - 1):
                p1, p2 = positions[i], positions[i + 1]
                if abs((p2 - p1) - spacing) <= 0.22 * spacing:
                    supported_pairs.append((p1, p2))
            # Keep the overlay readable on rulers with many ticks.
            if len(supported_pairs) > 10:
                step = max(1, int(math.ceil(len(supported_pairs) / 10.0)))
                supported_pairs = supported_pairs[::step][:10]

        if supported_pairs:
            red_width = max(6, stroke * 2 + 2)
            marker_r = max(5, stroke * 2 + 1)
            if candidate.get('orientation') == 'horizontal':
                y = min(h - 16, max(16, (y0 + y1) / 2.0))
                for p1, p2 in supported_pairs:
                    draw.line((p1, y, p2, y), fill=(255, 0, 0), width=red_width)
                    draw.ellipse((p1-marker_r, y-marker_r, p1+marker_r, y+marker_r),
                                 fill=(255, 0, 0))
                    draw.ellipse((p2-marker_r, y-marker_r, p2+marker_r, y+marker_r),
                                 fill=(255, 0, 0))
            else:
                x = min(w - 16, max(16, (x0 + x1) / 2.0))
                for p1, p2 in supported_pairs:
                    draw.line((x, p1, x, p2), fill=(255, 0, 0), width=red_width)
                    draw.ellipse((x-marker_r, p1-marker_r, x+marker_r, p1+marker_r),
                                 fill=(255, 0, 0))
                    draw.ellipse((x-marker_r, p2-marker_r, x+marker_r, p2+marker_r),
                                 fill=(255, 0, 0))

        # Downscale only for display; all annotations were drawn in original coordinates.
        mw, mh = max_size
        scale = min(1.0, mw / float(max(1, w)), mh / float(max(1, h)))
        if scale < 1.0:
            preview = preview.resize((max(1, int(round(w * scale))),
                                      max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
        return preview

    def __AutoDetectScale(self):
        """Detect ruler candidates and require visual verification before applying a scale."""
        if self.curImage is None:
            messagebox.showinfo(self._loc("Ruler scale", "定規スケール"), self._loc("Open an image first.", "先に画像を開いてください。"))
            return

        candidates = self._find_ruler_tick_candidates()
        if not candidates:
            messagebox.showwarning(
                self._loc("Ruler scale", "定規スケール"),
                self._loc(
                    "No sufficiently regular ruler ticks were detected near the image edges.\n\nUse View > Scale calibration... for manual calibration.",
                    "画像端付近で十分に規則的な定規目盛りを検出できませんでした。\n\n［表示］→［スケール校正...］で手動校正してください。"
                )
            )
            return

        self._show_auto_scale_dialog(candidates)

    def _show_auto_scale_dialog(self, candidates):
        """Show annotated ruler detections; the user chooses a candidate before applying it."""
        if isinstance(candidates, dict):
            candidates = [candidates]
        candidates = list(candidates)
        if not candidates:
            return

        win = tk.Toplevel(self.master)
        win.title(self._tr("Verify detected ruler scale"))
        win.resizable(True, True)
        win.minsize(720, 560)
        frame = ttk.Frame(win, padding=10)
        frame.grid(row=0, column=0, sticky='nsew')
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # Candidate selector.  Normally there are only a few edge candidates.
        selector_frame = ttk.Frame(frame)
        selector_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=(0, 6))
        ttk.Label(selector_frame, text=self._tr("Detection candidate:")).pack(side='left')
        candidate_var = tk.StringVar()
        candidate_box = ttk.Combobox(selector_frame, textvariable=candidate_var, state='readonly', width=55)
        candidate_box.pack(side='left', padx=(8, 0), fill='x', expand=True)

        labels = []
        edge_ja = {"top": "上", "bottom": "下", "left": "左", "right": "右"}
        for i, c in enumerate(candidates):
            if self.ui_language == "ja":
                labels.append(
                    f"{i+1}: {edge_ja.get(c['location'], c['location'])}端 | {c['tick_count']} 目盛り | "
                    f"同一長 {int(c.get('same_length_count', 0))} | "
                    f"{float(c['spacing']):.2f} px/目盛り | 信頼度 {float(c['confidence']):.0%}"
                )
            else:
                labels.append(
                    f"{i+1}: {c['location']} edge | {c['tick_count']} ticks | "
                    f"same-length {int(c.get('same_length_count', 0))} | "
                    f"{float(c['spacing']):.2f} px/tick | confidence {float(c['confidence']):.0%}"
                )
        candidate_box['values'] = labels
        candidate_box.current(0)

        info_var = tk.StringVar()
        ttk.Label(frame, textvariable=info_var, justify='left').grid(
            row=1, column=0, columnspan=2, sticky='w', pady=(0, 6)
        )

        # The preview is the evidence: yellow search band, green detected ticks,
        # red representative spacing.  User must inspect this before Apply.
        preview_holder = ttk.Frame(frame, relief='sunken', borderwidth=1)
        preview_holder.grid(row=2, column=0, columnspan=2, sticky='nsew', pady=(0, 8))
        preview_holder.rowconfigure(0, weight=1)
        preview_holder.columnconfigure(0, weight=1)
        preview_label = ttk.Label(preview_holder, anchor='center')
        preview_label.grid(row=0, column=0, sticky='nsew')
        legend = ttk.Label(
            frame,
            text=self._loc("Yellow box = searched edge band    Green = detected tick marks    Red = tick intervals supporting the spacing estimate", "黄枠 = 検索した画像端領域    緑 = 検出した目盛り    赤 = 間隔推定に使用した目盛り間隔"),
            foreground='gray'
        )
        legend.grid(row=3, column=0, columnspan=2, sticky='w', pady=(0, 8))

        form = ttk.Frame(frame)
        form.grid(row=4, column=0, columnspan=2, sticky='ew')
        ttk.Label(form, text=self._tr("Physical distance per minor tick:")).grid(row=0, column=0, sticky='w', pady=4)
        physical_var = tk.StringVar(value='1')
        ttk.Entry(form, textvariable=physical_var, width=16).grid(row=0, column=1, sticky='w', padx=(8, 14), pady=4)
        ttk.Label(form, text=self._tr("Unit:")).grid(row=0, column=2, sticky='w', pady=4)
        unit_var = tk.StringVar(value='mm')
        unit_box = ttk.Combobox(form, textvariable=unit_var, values=['mm', 'cm', 'um'], width=10, state='readonly')
        unit_box.grid(row=0, column=3, sticky='w', padx=(8, 0), pady=4)

        warning_var = tk.StringVar()
        ttk.Label(frame, textvariable=warning_var, foreground='gray', justify='left').grid(
            row=5, column=0, columnspan=2, sticky='w', pady=(6, 8)
        )

        state = {'index': 0, 'photo': None}

        def current_candidate():
            return candidates[state['index']]

        def refresh_candidate(event=None):
            idx = candidate_box.current()
            if idx < 0:
                idx = 0
            state['index'] = idx
            c = current_candidate()
            spacing = float(c['spacing'])
            location_ja = {"top": "上", "bottom": "下", "left": "左", "right": "右"}.get(c['location'], c['location'])
            info_var.set(self._loc(
                f"Candidate location: {c['location']} edge\nDetected tick marks: {c['tick_count']}\nDominant same-length ticks: {int(c.get('same_length_count', 0))} (length {float(c.get('dominant_tick_length', 0.0)):.1f} px)\nEstimated minor-tick spacing: {spacing:.3f} pixels\nConfidence: {float(c['confidence']):.0%}",
                f"候補位置: {location_ja}端\n検出目盛り数: {c['tick_count']}\n同一長の主要目盛り: {int(c.get('same_length_count', 0))}（長さ {float(c.get('dominant_tick_length', 0.0)):.1f} px）\n推定小目盛り間隔: {spacing:.3f} pixels\n信頼度: {float(c['confidence']):.0%}"
            ))
            if float(c['confidence']) < 0.42:
                warning_var.set(self._loc(
                    "Low-confidence detection. Inspect the green tick marks carefully; Cancel and use manual calibration if they are not the ruler ticks.",
                    "検出の信頼度が低いです。緑の目盛りをよく確認し、定規の目盛りでない場合はキャンセルして手動校正を使用してください。"
                ))
            else:
                warning_var.set(self._loc(
                    "Verify that the green marks really correspond to consecutive ruler ticks, then enter the physical value of one minor division.",
                    "緑の印が連続する定規目盛りに対応していることを確認し、小目盛り 1 区間の実距離を入力してください。"
                ))
            pil = self._make_ruler_preview(c)
            state['photo'] = ImageTk.PhotoImage(pil)
            preview_label.configure(image=state['photo'])

        candidate_box.bind('<<ComboboxSelected>>', refresh_candidate)
        refresh_candidate()

        def apply_scale():
            c = current_candidate()
            spacing = float(c['spacing'])
            try:
                physical = float(physical_var.get())
                if physical <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror(self._loc("Ruler scale", "定規スケール"), self._loc("Physical distance per tick must be a positive number.", "1 目盛りあたりの実距離は 0 より大きい数値で指定してください。"))
                return
            if spacing <= 0:
                messagebox.showerror(self._loc("Ruler scale", "定規スケール"), self._loc("Detected pixel spacing is invalid.", "検出されたピクセル間隔が無効です。"))
                return
            self.resolution = physical / spacing
            self.unit = unit_var.get() or 'mm'
            self.bSetResolution = True
            self.scale_source = 'ruler-auto'
            self.scale_confidence = float(c['confidence'])
            self.scale_pixels_per_tick = spacing
            self.scale_physical_per_tick = physical
            self.status_var.set(self._loc(
                f"Ruler scale: {self.resolution:.8g} {self.unit}/pixel ({spacing:.3f} px/tick, confidence {float(c['confidence']):.0%})",
                f"定規スケール: {self.resolution:.8g} {self.unit}/pixel ({spacing:.3f} px/目盛り, 信頼度 {float(c['confidence']):.0%})"
            ))
            self._refresh_batch_scale_info()
            win.destroy()

        buttons = ttk.Frame(frame)
        buttons.grid(row=6, column=0, columnspan=2, sticky='e')
        ttk.Button(buttons, text=self._tr("Apply verified scale"), command=apply_scale).pack(side='left', padx=4)
        ttk.Button(buttons, text=self._tr("Cancel"), command=win.destroy).pack(side='left', padx=4)

    def __AI_Select(self, event=None):
        self.status_var.set(self._loc("Start detecting roots using AI", "AI による根の検出を開始します"))
        self.resultWindow = None
        
        if self.curImage is not None:
            self.yoloModel = self._ensure_yolo_model()
            if self.imgType ==1:
                alpha = 1.75 # Contrast control (1.0-3.0)
                beta = 50 # Brightness control (0-100)
                img = cv2.convertScaleAbs(self.curImage, alpha=alpha, beta=beta)
                rois_AI = self.detectYOLO(self.yoloModel, cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            elif self.imgType ==2:
                rois_AI = self.detectYOLO(self.yoloModel, cv2.cvtColor(self.curImage,cv2.COLOR_BGR2RGB))
            srois_AI = sorted(rois_AI, key=lambda rois_AI: (rois_AI.ROICoord[1][0]+rois_AI.ROICoord[0][0]))

            for one in srois_AI:
                self.RootAll.append(EachRootData(one))
            if self.__imframe is not None:
                self.__imframe.SetHisCoord(self.RootAll)
    
    def SavePtXml(self):
        """Save project geometry, root types and ROI semantic masks."""
        xroot = ET.Element("data")
        image_node = ET.SubElement(xroot, "colorimage")
        source_path = self._ensure_project_local_source()
        image_node.text = self._portable_xml_relpath(source_path, self.ProjectDir)

        roi_mask_dir = os.path.join(self.ProjectDir, "roi_masks")
        os.makedirs(roi_mask_dir, exist_ok=True)

        for i, eachroot in enumerate(self.RootAll):
            roi = ET.SubElement(xroot, "Roi")
            roi.set("Roi", str(i + 1))
            one = eachroot.bbox
            element = ET.SubElement(roi, 'item')
            ET.SubElement(element, 'ltx').text = str(int(one[0]))
            ET.SubElement(element, 'lty').text = str(int(one[1]))
            ET.SubElement(element, 'rbx').text = str(int(one[2]))
            ET.SubElement(element, 'rby').text = str(int(one[3]))

            # Store a full-width semantic ROI mask so manual edits survive reopen.
            try:
                roi_binary, roi_semantic = self._roi_export_masks(eachroot)
                sem_path = os.path.join(roi_mask_dir, f"roi_{i + 1:03d}_labels.png")
                bin_path = os.path.join(roi_mask_dir, f"roi_{i + 1:03d}_mask.png")
                self._write_png_unicode(sem_path, np.asarray(roi_semantic, dtype=np.uint8))
                self._write_png_unicode(bin_path, (roi_binary > 0).astype(np.uint8) * 255)
                ET.SubElement(roi, "semanticmask").text = self._portable_xml_relpath(sem_path, self.ProjectDir)
                ET.SubElement(roi, "binarymask").text = self._portable_xml_relpath(bin_path, self.ProjectDir)
            except Exception:
                pass

            for j, rootp in enumerate(eachroot.prop):
                labels = ET.SubElement(roi, "Pts")
                labels.set("value", str(j + 1))
                labels.set("type", str(self._root_type(rootp)))
                if self._root_type(rootp) >= 3:
                    labels.set(
                        "label_value",
                        str(int(getattr(rootp, "label_value", max(3, j + 1))))
                    )

                # Save every instance mask independently. Unlike a single uint8
                # semantic PNG, these files preserve primary/lateral overlap.
                try:
                    prop_mask = self._prop_mask(rootp)
                    if prop_mask is not None:
                        prop_mask_path = os.path.join(
                            roi_mask_dir,
                            f"roi_{i + 1:03d}_prop_{j + 1:03d}_mask.png"
                        )
                        self._write_png_unicode(
                            prop_mask_path, prop_mask.astype(np.uint8) * 255
                        )
                        labels.set("maskfile", self._portable_xml_relpath(prop_mask_path, self.ProjectDir))
                except Exception:
                    pass

                for point in rootp.path:
                    coord_elem = ET.SubElement(labels, "coordinate")
                    coord_elem.set("x", str(int(point[0])))
                    coord_elem.set("y", str(int(point[1])))

        ET.SubElement(xroot, "bRotate").text = '1' if self.bNeedRotate else '0'
        ET.SubElement(xroot, "rotation_direction").text = (getattr(self, 'rotation_direction', 'cw')
                                                           if self.bNeedRotate else 'none')
        ET.SubElement(xroot, "resolution").text = str(float(getattr(self, 'resolution', 1.0)))
        ET.SubElement(xroot, "unit").text = str(getattr(self, 'unit', 'cm'))
        ET.SubElement(xroot, "scale_set").text = '1' if getattr(self, 'bSetResolution', False) else '0'
        ET.SubElement(xroot, "scale_source").text = str(getattr(self, 'scale_source', 'default'))
        ET.SubElement(xroot, "scale_confidence").text = str(float(getattr(self, 'scale_confidence', 0.0)))
        ET.SubElement(xroot, "segmentation_method").text = str(getattr(self, 'processing_method', 'unknown'))
        ET.SubElement(xroot, "resultimg").text = "result.jpg"
        ET.SubElement(xroot, "resultfile").text = "result.csv"
        ET.SubElement(xroot, "resultmask").text = "mask.png"
        ET.SubElement(xroot, "resultlabels").text = "mask_labels.png"
        ET.SubElement(xroot, "resultskeleton").text = "skeleton.png"

        tree = ET.ElementTree(xroot)
        tree.write(os.path.join(self.ProjectDir, self.file_name + ".xml"),
                   encoding="utf-8", xml_declaration=True)

    def ProcessROI(self,idx):    
        if 1:
                   mode = int(self.RootAll[idx].prsMode)
                   # Supported segmentation modes: ICML-NoAI (1) and ICML-improve (5).
                   if mode not in (1, 5):
                       mode = 5
                       self.RootAll[idx].prsMode = mode
                   p = ar(self.RootAll[idx].mImgCV, mode, 0, 0)
                   p.Processing()
                   prop = p.Get()
                   rlabel = p.GetLabel()
                   self.RootAll[idx].prop = prop.copy()
                   self.RootAll[idx].roilabel = rlabel.copy()
                   # Keep the raw binary root segmentation as well as the
                   # branch-label image.  The former is exported as mask.png
                   # so segmentation can be validated independently later.
                   self.RootAll[idx].segmask = p.GetLabelLoad().copy()
                   #modify the flag 
                   self.RootAll[idx].bProcessed = True
                   self.gui_queue.put(idx)
                   with threading.Lock():
                       self.active_count -=1
            
    def _prepare_unprocessed_rois(self, mode):
        """Prepare only ROIs that have not already been processed.

        Previously processed ROIs keep their masks, measurements and labels.
        A manually added ROI normally has bProcessed == False and is therefore
        included automatically.
        """
        crop_source = self._root_crop_source()
        pending = []

        for idx, eachroot in enumerate(self.RootAll):
            if bool(getattr(eachroot, "bProcessed", False)):
                continue

            eachroot.Generatebb(crop_source, mode)
            eachroot.bProcessed = False
            eachroot.prop = []
            eachroot.roilabel = None
            eachroot.segmask = None
            pending.append(idx)

        return pending

    def GetModePixle(self,im):
        vals,counts = np.unique(im, return_counts=True)
        index = np.argmax(counts)
        return vals[index]
    def ShowResultImage(self,img_lock):
        try:
            while not self.gui_queue.empty():
                idx = self.gui_queue.get()
                rlabel = self.RootAll[idx].roilabel.astype(np.uint8)
                bb = self.RootAll[idx].bbox
                maxL = len(self.RootAll[idx].prop)#int(np.max(rlabel))
                if maxL >0:
                    # mImgCV is displayed through PIL/ImageTk, so keep it in RGB.
                    img = self.RootAll[idx].mImgCV.copy()
                    for one in range(1, maxL + 1):
                        rootp = self.RootAll[idx].prop[one - 1]
                        color = self._root_display_color(self.RootAll[idx], one - 1)
                        draw_mask = self._display_instance_mask(
                            self.RootAll[idx], one - 1, rlabel
                        )
                        self._paint_main_label(img, draw_mask, color)
                    self.mResultImg[bb[1]:bb[3],bb[0]:bb[2]] = img.copy()
                self.status_var.set(self._loc("Processed %d of %d roots" % (len(self.RootAll)-self.active_count, len(self.RootAll)), "%d / %d 本の根を処理しました" % (len(self.RootAll)-self.active_count, len(self.RootAll))))
                img = Image.fromarray(self.mResultImg)
                with img_lock:
                    if self.__imframe is not None:
                       self.__imframe.UpdateImg(img)            
        finally:
            if self.active_count >0:
               self.after(100,self.ShowResultImage,img_lock)
            else:
                self.status_var.set(self._loc("Processing finished", "処理が完了しました"))
                blend = self.SaveResult()
                img = Image.fromarray(blend)
                if self.__imframe is not None:
                   self.__imframe.UpdateImg(img)  
    @staticmethod
    def _path_length_px(path):
        """Geodesic length of an ordered root path in pixels."""
        if path is None or len(path) < 2:
            return 0.0
        pts = np.asarray(path, dtype=float)
        diff = np.diff(pts[:, :2], axis=0)
        return float(np.sqrt(np.sum(diff * diff, axis=1)).sum())

    @staticmethod
    def _root_type(rootp):
        """Map branch-specific labels >=3 to the common lateral-root class."""
        return 3 if rootp.type > 2 else int(rootp.type)

    @staticmethod
    def _color_from_label_value(label_value, root_type=None):
        """Return the editor-style RGB colour for one semantic/instance id.

        Leaf (1) and primary root (2) keep their fixed class colours.  Lateral
        roots use their actual semantic instance id (3, 4, 5, ...).  Some versions
        of mask_editor expose colours for every instance directly through
        CLASS_COLORS; when they do, use those exact colours.  Older versions only
        define the base classes, so a stable high-contrast palette is used as a
        backwards-compatible fallback.
        """
        try:
            label_value = int(label_value)
        except (TypeError, ValueError):
            label_value = 3
        try:
            root_type = int(root_type) if root_type is not None else (label_value if label_value <= 2 else 3)
        except (TypeError, ValueError):
            root_type = 3

        key = root_type if root_type <= 2 else max(3, label_value)

        # Prefer the palette exported by mask_editor.  It can be either a dict or
        # an indexable sequence depending on the editor version.
        try:
            c = np.asarray(CLASS_COLORS[key]).reshape(-1)
            if c.size >= 3:
                return tuple(int(np.clip(v, 0, 255)) for v in c[:3])
        except (KeyError, IndexError, TypeError, ValueError):
            pass

        # Class-colour fallback for leaf/primary if an older editor palette is used.
        if root_type <= 2:
            fallback_class = {
                1: (255, 215, 0),     # leaf
                2: (255, 64, 64),     # primary root
            }
            return fallback_class.get(root_type, (255, 255, 255))

        # Deterministic lateral-instance fallback.  The first ten laterals are all
        # visually distinct; larger ids cycle in a stable way.
        lateral_palette = (
            (255, 127, 14),   # orange
            (148, 103, 189),  # purple
            (23, 190, 207),   # cyan
            (227, 119, 194),  # pink
            (188, 189, 34),   # olive/yellow
            (31, 119, 180),   # blue
            (44, 160, 44),    # green
            (214, 39, 40),    # red
            (140, 86, 75),    # brown
            (127, 127, 127),  # grey
        )
        return lateral_palette[(max(3, label_value) - 3) % len(lateral_palette)]

    def _root_instance_label_value(self, eachroot, prop_index):
        """Resolve the semantic id used by the editor for one RootProp."""
        rootp = eachroot.prop[prop_index]
        rtype = self._root_type(rootp)
        if rtype <= 2:
            return rtype

        # Edited/reloaded projects store the exact lateral id explicitly.
        value = getattr(rootp, 'label_value', None)
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = None
        if value is not None and value >= 3:
            return min(255, value)

        # Historically RootProp.type itself was 3,4,5,... for lateral instances.
        try:
            raw_type = int(getattr(rootp, 'type', 3))
        except (TypeError, ValueError):
            raw_type = 3
        if raw_type > 3:
            return min(255, raw_type)

        # Last fallback for projects where every lateral was stored as type==3.
        # Count only previous lateral props so the first lateral is 3 regardless of
        # whether a leaf is present in the ROI.
        lateral_before = sum(
            1 for p in eachroot.prop[:prop_index] if self._root_type(p) >= 3
        )
        return min(255, 3 + lateral_before)

    def _root_display_color(self, eachroot, prop_index):
        rootp = eachroot.prop[prop_index]
        rtype = self._root_type(rootp)
        value = self._root_instance_label_value(eachroot, prop_index)
        return self._color_from_label_value(value, rtype)

    @staticmethod
    def _prop_mask(rootp, shape=None):
        """Return one RootProp mask as bool without resolving overlaps."""
        mask = getattr(rootp, "mask", None)
        if mask is None:
            return None
        mask = np.asarray(mask) > 0
        if shape is not None and mask.shape != tuple(shape):
            return None
        return mask

    def _display_instance_mask(self, eachroot, prop_index, roi_label=None):
        """Mask used only for drawing; prefer overlap-aware RootProp.mask."""
        if 0 <= prop_index < len(eachroot.prop):
            pmask = self._prop_mask(eachroot.prop[prop_index])
            if pmask is not None:
                return pmask
        if roi_label is not None:
            return np.asarray(roi_label) == (prop_index + 1)
        return None

    def _load_saved_overlap_props(self, roi_index, roi_shape, roi_image):
        """Restore per-root masks saved in XML.

        Returns (props, display_label, semantic_projection, binary_root) or None
        when the project predates overlap-aware per-instance mask files.
        """
        pts_nodes = roi_index.findall("Pts")
        if not pts_nodes or not any(pts.get("maskfile") for pts in pts_nodes):
            return None

        processor = ar(roi_image, 5 if roi_image is not None else 1, 1, 0)
        props = []
        rlabel = np.zeros(roi_shape, dtype=np.uint16)
        semantic = np.zeros(roi_shape, dtype=np.uint8)
        binary_root = np.zeros(roi_shape, dtype=np.uint8)
        next_lateral = 3

        loaded_any = False
        for pts in pts_nodes:
            mask_path = pts.get("maskfile")
            if not mask_path:
                continue
            mask_path = self._resolve_project_file(self.ProjectDir, mask_path)
            mask = self._read_cv_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            if mask.shape != tuple(roi_shape):
                mask = cv2.resize(
                    mask, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_NEAREST
                )
            mask = (np.asarray(mask) > 0).astype(np.uint8)
            if not np.any(mask):
                continue

            value = int(pts.get("value", len(props) + 1))
            type_attr = pts.get("type")
            if type_attr is not None:
                rtype = int(type_attr)
            elif self.imgType == 2 and value == 1:
                rtype = 1
            elif (self.imgType == 2 and value == 2) or (self.imgType == 1 and value == 1):
                rtype = 2
            else:
                rtype = 3

            try:
                prop = processor.ProcessLoad(mask, self.imgType, rtype)
                prop.type = rtype
                # Keep exactly the stored mask, including shared junction pixels.
                prop.mask = mask.copy()
            except Exception:
                coordinates = [
                    (int(c.get("x")), int(c.get("y")))
                    for c in pts.findall("coordinate")
                ]
                if not coordinates:
                    coordinates = list(map(tuple, np.argwhere(mask > 0)))
                st = coordinates[0] if coordinates else (0, 0)
                prop = RootProp(0.0, 0.0, coordinates, rtype, st, mask, 0)

            saved_path = [
                (int(c.get("x")), int(c.get("y")))
                for c in pts.findall("coordinate")
            ]
            if saved_path:
                prop.path = saved_path

            if rtype >= 3:
                label_value = int(pts.get("label_value", next_lateral))
                label_value = max(3, min(255, label_value))
                prop.label_value = label_value
                next_lateral = max(next_lateral + 1, label_value + 1)
                semantic[mask > 0] = label_value
                binary_root[mask > 0] = 1
            else:
                prop.label_value = rtype
                semantic[mask > 0] = rtype
                if rtype == 2:
                    binary_root[mask > 0] = 1

            props.append(prop)
            rlabel[mask > 0] = len(props)
            loaded_any = True

        if not loaded_any:
            return None
        if len(props) <= 255:
            rlabel = rlabel.astype(np.uint8)
        return props, rlabel, semantic, binary_root

    @staticmethod
    def _paint_main_label(base, mask, color):
        """Paint mask pixels; transparency 0 is fully opaque like mask_editor."""
        if base is None or mask is None or not np.any(mask):
            return
        transparency = float(np.clip(MAIN_LABEL_TRANSPARENCY, 0.0, 1.0))
        rgb = np.asarray(color, dtype=np.float32)
        if transparency <= 0.001:
            # Exact editor behaviour: replace only mask pixels with the label color.
            base[mask] = rgb.astype(np.uint8)
        elif transparency >= 0.999:
            return
        else:
            src = base[mask].astype(np.float32)
            opacity = 1.0 - transparency
            base[mask] = np.clip(
                opacity * rgb + transparency * src, 0, 255
            ).astype(np.uint8)

    @staticmethod
    def _distance_point_to_path(point, path):
        if path is None or len(path) == 0:
            return float("inf")
        pts = np.asarray(path, dtype=float)
        p = np.asarray(point, dtype=float)
        return float(np.sqrt(np.sum((pts[:, :2] - p[:2]) ** 2, axis=1)).min())

    def _orient_primary_path(self, path, leaf_point=None):
        """Return primary-root path ordered base/seed -> root tip.

        If a leaf/seed reference is available, the endpoint closest to it is
        treated as the base.  Otherwise the upper endpoint in the image is
        treated as the base, matching the acquisition geometry of this tool.
        """
        if path is None or len(path) < 2:
            return list(path or [])
        pts = list(path)
        p0 = np.asarray(pts[0], dtype=float)
        p1 = np.asarray(pts[-1], dtype=float)
        if leaf_point is not None:
            leaf = np.asarray(leaf_point, dtype=float)
            if np.linalg.norm(p1 - leaf) < np.linalg.norm(p0 - leaf):
                pts.reverse()
        elif p1[0] < p0[0]:  # row/y is smaller toward the top of the image
            pts.reverse()
        return pts

    def _orient_lateral_path(self, path, primary_path):
        """Return a lateral path ordered insertion point -> lateral tip."""
        if path is None or len(path) < 2:
            return list(path or [])
        pts = list(path)
        if primary_path is None or len(primary_path) == 0:
            return pts
        d0 = self._distance_point_to_path(pts[0], primary_path)
        d1 = self._distance_point_to_path(pts[-1], primary_path)
        if d1 < d0:
            pts.reverse()
        return pts

    @staticmethod
    def _direction_metrics(oriented_path):
        """Endpoint vector and angles for an ordered (base -> tip) path.

        Image-direction convention: 0 deg = right, 90 deg = down,
        180 deg = left, 270 deg = up.  The signed vertical angle is measured
        from downward vertical: 0 deg = down, +90 deg = right, -90 deg = left.
        """
        if oriented_path is None or len(oriented_path) < 2:
            return None
        start = np.asarray(oriented_path[0], dtype=float)  # row, col
        end = np.asarray(oriented_path[-1], dtype=float)
        dy = float(end[0] - start[0])
        dx = float(end[1] - start[1])
        euclidean_px = math.hypot(dx, dy)
        direction_deg = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        vertical_signed_deg = math.degrees(math.atan2(dx, dy))
        return {
            "start": start,
            "end": end,
            "dx": dx,
            "dy": dy,
            "euclidean_px": euclidean_px,
            "direction_deg": direction_deg,
            "vertical_signed_deg": vertical_signed_deg,
            "vertical_abs_deg": abs(vertical_signed_deg),
        }

    def _lateral_insertion_angle(self, lateral_path, primary_path, sample_len=5):
        """2-D angle between a lateral root and the local primary-root tangent."""
        if lateral_path is None or primary_path is None:
            return None
        if len(lateral_path) < 2 or len(primary_path) < 2:
            return None

        lat = np.asarray(self._orient_lateral_path(lateral_path, primary_path), dtype=float)
        pri = np.asarray(primary_path, dtype=float)
        base = lat[0]
        idx = int(np.argmin(np.sum((pri - base) ** 2, axis=1)))

        k_child = min(sample_len, len(lat) - 1)
        child_v = lat[k_child] - lat[0]

        i0 = max(0, idx - sample_len)
        i1 = min(len(pri) - 1, idx + sample_len)
        parent_v = pri[i1] - pri[i0]

        nc = float(np.linalg.norm(child_v))
        npv = float(np.linalg.norm(parent_v))
        if nc == 0 or npv == 0:
            return None
        cosang = float(np.dot(child_v, parent_v) / (nc * npv))
        cosang = max(-1.0, min(1.0, cosang))
        return math.degrees(math.acos(cosang))

    @staticmethod
    def _skeleton_length_px(skeleton):
        """8-connected skeleton length using 1 and sqrt(2) edge weights."""
        sk = np.asarray(skeleton, dtype=bool)
        if not np.any(sk):
            return 0.0
        horizontal = np.count_nonzero(sk[:, :-1] & sk[:, 1:])
        vertical = np.count_nonzero(sk[:-1, :] & sk[1:, :])

        # Count a diagonal edge only when there is no orthogonal intermediate
        # pixel connecting the same two pixels.  This prevents artificial
        # sqrt(2) shortcuts around T/L junctions while retaining true diagonal
        # segments.
        diag1 = sk[:-1, :-1] & sk[1:, 1:]
        diag1 &= ~(sk[:-1, 1:] | sk[1:, :-1])
        diag2 = sk[:-1, 1:] & sk[1:, :-1]
        diag2 &= ~(sk[:-1, :-1] | sk[1:, 1:])
        diagonal = np.count_nonzero(diag1) + np.count_nonzero(diag2)
        return float(horizontal + vertical + math.sqrt(2.0) * diagonal)

    def _mask_metrics(self, root_mask):
        """Measurements derived directly from a binary root segmentation mask."""
        binary = (np.asarray(root_mask) > 0).astype(np.uint8)
        empty = {
            "network_area": 0.0, "skeleton_length": 0.0,
            "tips": 0, "branch_points": 0, "branching_frequency": 0.0,
            "avg_diameter": 0.0, "median_diameter": 0.0, "max_diameter": 0.0,
            "depth": 0.0, "width": 0.0, "width_depth_ratio": 0.0,
            "convex_area": 0.0, "solidity": 0.0, "perimeter": 0.0,
            "skeleton": np.zeros_like(binary, dtype=np.uint8),
        }
        if not np.any(binary):
            return empty

        res = float(self.resolution)
        coords = np.argwhere(binary > 0)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        depth = (float(y1 - y0) + 1.0) * res
        width = (float(x1 - x0) + 1.0) * res
        network_area = float(binary.sum()) * res * res

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        perimeter_px = sum(cv2.arcLength(cnt, True) for cnt in contours)
        perimeter = float(perimeter_px) * res

        # Pixel-count convex hull keeps the area convention consistent with
        # network_area and guarantees solidity <= 1.
        convex_mask = morphology.convex_hull_image(binary.astype(bool))
        convex_area = float(np.count_nonzero(convex_mask)) * res * res
        solidity = network_area / convex_area if convex_area > 0 else 0.0

        skel = morphology.skeletonize(binary.astype(bool))
        skeleton_length_px = self._skeleton_length_px(skel)
        skeleton_length = skeleton_length_px * res

        kernel = np.ones((3, 3), dtype=np.uint8)
        neighbor_count = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16U, kernel,
                                      borderType=cv2.BORDER_CONSTANT)
        neighbor_count = neighbor_count - skel.astype(np.uint16)
        tip_pixels = skel & (neighbor_count == 1)
        branch_pixels = skel & (neighbor_count >= 3)
        tips = int(np.count_nonzero(tip_pixels))
        n_branch_labels, _ = cv2.connectedComponents(branch_pixels.astype(np.uint8), connectivity=8)
        branch_points = max(int(n_branch_labels) - 1, 0)
        branching_frequency = branch_points / skeleton_length if skeleton_length > 0 else 0.0

        distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        diam_px = 2.0 * distance[skel]
        if diam_px.size:
            diam = diam_px * res
            avg_diameter = float(np.mean(diam))
            median_diameter = float(np.median(diam))
            max_diameter = float(np.max(diam))
        else:
            avg_diameter = median_diameter = max_diameter = 0.0

        return {
            "network_area": network_area,
            "skeleton_length": skeleton_length,
            "tips": tips,
            "branch_points": branch_points,
            "branching_frequency": branching_frequency,
            "avg_diameter": avg_diameter,
            "median_diameter": median_diameter,
            "max_diameter": max_diameter,
            "depth": depth,
            "width": width,
            "width_depth_ratio": width / depth if depth > 0 else 0.0,
            "convex_area": convex_area,
            "solidity": solidity,
            "perimeter": perimeter,
            "skeleton": skel.astype(np.uint8),
        }

    def _semantic_to_props(self, semantic, roi_image=None):
        """Rebuild measurable branches from an instance-aware root label mask.

        Values 0/1/2 mean background/leaf/primary. Every value >=3 is treated
        as a lateral-root instance id. Disconnected pieces of the same value
        are kept as separate measurable components rather than silently joined.
        """
        semantic = np.asarray(semantic)
        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        semantic = semantic.astype(np.uint8)
        # ``roi_image`` is used by ICML-improve leaf-instance estimation.
        # Without it, connected leaf blades would collapse back to one leaf
        # when an edited semantic mask is reconstructed.
        processor = ar(roi_image, 5 if roi_image is not None else 1, 1, 0)
        props = []
        rlabel = np.zeros(semantic.shape, dtype=np.uint16)
        instance_id = 0

        def append_components(class_mask, root_type, source_value):
            nonlocal instance_id
            nlabels, labels = cv2.connectedComponents(class_mask.astype(np.uint8), connectivity=8)
            for component_id in range(1, nlabels):
                component = (labels == component_id).astype(np.uint8)
                if not np.any(component):
                    continue
                measure_mask = component if root_type == 1 else morphology.skeletonize(component > 0).astype(np.uint8)
                if not np.any(measure_mask):
                    continue
                try:
                    if root_type == 1 and roi_image is not None and hasattr(processor, "GetLeafFromMaskImproved"):
                        # Preserve the ICML-improve leaf-instance count even when
                        # several blades form one connected semantic leaf region.
                        prop = processor.GetLeafFromMaskImproved(component)
                    else:
                        prop = processor.ProcessLoad(measure_mask, self.imgType, root_type)
                    prop.type = root_type
                    prop.mask = measure_mask.copy()
                except Exception:
                    pts = list(map(tuple, np.argwhere(measure_mask > 0)))
                    st = pts[0] if pts else (0, 0)
                    prop = RootProp(0.0, 0.0, pts, root_type, st, measure_mask, 0)
                # Keep the editor label id available for future UI/exports.
                prop.label_value = int(source_value)
                props.append(prop)
                instance_id += 1
                rlabel[measure_mask > 0] = instance_id

        append_components(semantic == 1, 1, 1)
        append_components(semantic == 2, 2, 2)
        for lateral_value in sorted(int(v) for v in np.unique(semantic) if int(v) >= 3):
            append_components(semantic == lateral_value, 3, lateral_value)

        if instance_id <= 255:
            rlabel = rlabel.astype(np.uint8)
        return props, rlabel

    def _roi_export_masks(self, eachroot):
        """Return binary root mask and a single-channel DISPLAY projection.

        RootProp masks are overlap-aware and remain the source of truth.  The
        returned semantic image is only a PNG/editor-compatible projection, so a
        shared primary/lateral pixel can show only one visible id. Laterals are
        painted after the primary so junctions remain visually connected.
        """
        semantic_edit = getattr(eachroot, "semantic_mask", None)
        if semantic_edit is not None:
            semantic_edit = np.asarray(semantic_edit, dtype=np.uint8)
            if semantic_edit.ndim == 2:
                semantic = semantic_edit.astype(np.uint8)
                segmask = getattr(eachroot, "segmask", None)
                if segmask is not None and np.shape(segmask) == np.shape(semantic):
                    binary = (np.asarray(segmask) > 0).astype(np.uint8)
                else:
                    binary = (semantic >= 2).astype(np.uint8)
                return binary, semantic

        rlabel = np.asarray(eachroot.roilabel) if eachroot.roilabel is not None else None
        if rlabel is not None:
            shape = rlabel.shape
        elif eachroot.prop:
            pmask = self._prop_mask(eachroot.prop[0])
            shape = pmask.shape if pmask is not None else (int(eachroot.bbox[5]), int(eachroot.bbox[4]))
        else:
            shape = (int(eachroot.bbox[5]), int(eachroot.bbox[4]))

        semantic = np.zeros(shape, dtype=np.uint8)

        # Paint in explicit display priority: leaf -> primary -> laterals.
        next_lateral = 3
        ordered = []
        ordered.extend((j, p) for j, p in enumerate(eachroot.prop) if self._root_type(p) == 1)
        ordered.extend((j, p) for j, p in enumerate(eachroot.prop) if self._root_type(p) == 2)
        ordered.extend((j, p) for j, p in enumerate(eachroot.prop) if self._root_type(p) >= 3)

        for j, rootp in ordered:
            rtype = self._root_type(rootp)
            if rtype >= 3:
                value = int(getattr(rootp, "label_value", next_lateral))
                value = max(3, min(255, value))
                next_lateral = min(255, max(next_lateral + 1, value + 1))
            else:
                value = rtype

            pmask = self._prop_mask(rootp, shape)
            if pmask is None and rlabel is not None:
                pmask = (rlabel == (j + 1))
            if pmask is not None:
                semantic[pmask] = value

        # The binary segmentation represents union(root) and naturally supports overlap.
        segmask = getattr(eachroot, "segmask", None)
        if segmask is not None and np.shape(segmask) == tuple(shape):
            binary_root = (np.asarray(segmask) > 0).astype(np.uint8)
        else:
            binary_root = np.zeros(shape, dtype=np.uint8)
            for rootp in eachroot.prop:
                if self._root_type(rootp) >= 2:
                    pmask = self._prop_mask(rootp, shape)
                    if pmask is not None:
                        binary_root[pmask] = 1

        return binary_root, semantic

    def _build_full_export_masks(self):
        binary = np.zeros(self.curImage.shape[:2], dtype=np.uint8)
        semantic = np.zeros(self.curImage.shape[:2], dtype=np.uint8)
        for eachroot in self.RootAll:
            bb = eachroot.bbox
            roi_binary, roi_semantic = self._roi_export_masks(eachroot)
            y0, y1 = int(bb[1]), int(bb[3])
            x0, x1 = int(bb[0]), int(bb[2])
            h = min(y1 - y0, roi_binary.shape[0])
            w = min(x1 - x0, roi_binary.shape[1])
            if h <= 0 or w <= 0:
                continue
            binary[y0:y0+h, x0:x0+w] = np.maximum(
                binary[y0:y0+h, x0:x0+w], roi_binary[:h, :w])
            # Later ROIs may overlap; non-background semantic pixels take priority.
            sem = roi_semantic[:h, :w]
            region = semantic[y0:y0+h, x0:x0+w]
            region[sem > 0] = sem[sem > 0]
        return binary, semantic

    def RefreshAfterEdit(self):
        """Refresh main canvas and derived outputs after manual semantic-mask editing."""
        if self.curImage is None:
            return
        self.mResultImg = self.curImage.copy()
        blend = self.SaveResult()
        if self.__imframe is not None:
            self.__imframe.UpdateImg(Image.fromarray(blend))
        self.status_var.set(self._loc("Manual edits applied and measurements recalculated", "手動編集を適用し、計測値を再計算しました"))

    def SaveResult(self):
        # Generate overlay, validation masks and measurements.
        if self.curImage is None:
            return
        # Rebuild from the untouched source image. Only mask pixels receive color;
        # all background pixels remain exactly the original image.
        if np.asarray(self.curImage).ndim == 2:
            blend = cv2.cvtColor(self.curImage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            blend = self.curImage.copy()

        label = np.zeros(self.curImage.shape[0:2], np.uint8)
        for i, eachroot in enumerate(self.RootAll):
            bb = eachroot.bbox
            roi_label = None
            if eachroot.roilabel is not None:
                roi_label = np.asarray(eachroot.roilabel)
                label[bb[1]:bb[3], bb[0]:bb[2]] = roi_label.copy()

            # Use the same instance-aware colour rule as the annotation editor:
            # leaf/primary have fixed colours; each lateral id (3,4,5,...) has its
            # own colour.  Measurement classes are unchanged.
            roi_blend = blend[bb[1]:bb[3], bb[0]:bb[2]]
            for j, rootp in enumerate(eachroot.prop):
                color = self._root_display_color(eachroot, j)
                draw_mask = self._display_instance_mask(eachroot, j, roi_label)
                self._paint_main_label(roi_blend, draw_mask, color)
                if rootp.type == 1 and self.imgType == 2:
                    pass
                else:
                    if rootp.lr == -1:
                        pt = (rootp.pt[1] + bb[0] - 50, rootp.pt[0] + bb[1])
                    elif rootp.lr == 0:
                        pt = (rootp.pt[1] + bb[0], rootp.pt[0] + bb[1])
                    else:
                        pt = (rootp.pt[1] + bb[0], rootp.pt[0] + bb[1] + 10)
                    blend = cv2.putText(blend, str(i + 1) + "-(" + str(j) + ")",
                                        pt, 3, 0.7, color, 1, 1)
            if i == 0:
                sStr = "1st Root"
            elif i == 1:
                sStr = "2nd Root"
            elif i == 2:
                sStr = "3rd Root"
            else:
                sStr = str(i + 1) + "th Root"
            blend = cv2.putText(blend, sStr,
                                (int(bb[2] / 2 + bb[0] / 2) - 20, bb[3] + 10),
                                3, 1.5, (0, 255, 0), 1, 1)

        cv_image = cv2.cvtColor(blend, cv2.COLOR_RGB2BGR)
        cv2.imencode(".jpg", cv_image)[1].tofile(os.path.join(self.ProjectDir, "result.jpg"))

        # Export masks for future validation.
        root_mask, semantic_mask = self._build_full_export_masks()
        full_metrics = self._mask_metrics(root_mask)
        cv2.imencode(".png", root_mask * 255)[1].tofile(os.path.join(self.ProjectDir, "mask.png"))
        cv2.imencode(".png", semantic_mask)[1].tofile(os.path.join(self.ProjectDir, "mask_labels.png"))
        cv2.imencode(".png", full_metrics["skeleton"] * 255)[1].tofile(
            os.path.join(self.ProjectDir, "skeleton.png"))

        filename = os.path.join(self.ProjectDir, "result.csv")
        with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow([
                "Root id", "Type",
                f"Path length ({self.unit})", f"Euclidean distance ({self.unit})",
                "Straightness", "Tortuosity",
                "Start X (px)", "Start Y (px)", "End X (px)", "End Y (px)",
                f"Delta X ({self.unit})", f"Delta Y ({self.unit})",
                "Direction (deg; 0=right, 90=down)",
                "Angle from downward vertical (deg; +right/-left)",
                "Lateral insertion angle (deg)",
                f"Area ({self.unit}^2)", "Count"
            ])

            # Summary columns retained from the old CSV, with corrected formulas.
            Item = ["Item"]
            TotalLeafarea = [f"Total leaf area [{self.unit}^2]"]
            NumOfLeaf = ["Number of leaf"]
            AverageLeaf = [f"Average leaf area [{self.unit}^2]"]
            totalLengths = [f"Total root length [{self.unit}]"]
            primarLength = [f"Primary root length [{self.unit}]"]
            primarVector = [f"Primary root vector [{self.unit}]"]
            straightness = ["Primary root straightness"]
            tortuosity = ["Primary root tortuosity"]
            primaryDirection = ["Primary direction [deg; 0=right, 90=down]"]
            primaryVerticalAngle = ["Primary angle from downward vertical [deg; +right/-left]"]
            TotalLaterLength = [f"Total lateral root length [{self.unit}]"]
            TotalLaterRootNum = ["Total lateral root number"]
            MeanLaterLength = [f"Mean lateral root length [{self.unit}]"]
            MaxLaterLength = [f"Max lateral root length [{self.unit}]"]
            MeanInsertionAngle = ["Mean lateral insertion angle [deg]"]
            LateralDensity = [f"Lateral branching density [count/{self.unit}]"]

            NetworkArea = [f"Root mask network area [{self.unit}^2]"]
            SkeletonLength = [f"Root mask skeleton length [{self.unit}]"]
            RootTips = ["Root tip count"]
            BranchPoints = ["Branch point count"]
            BranchFreq = [f"Branching frequency [count/{self.unit}]"]
            AvgDiameter = [f"Average root diameter [{self.unit}]"]
            MedianDiameter = [f"Median root diameter [{self.unit}]"]
            MaxDiameter = [f"Maximum root diameter [{self.unit}]"]
            RootDepth = [f"Root system depth [{self.unit}]"]
            RootWidth = [f"Maximum root system width [{self.unit}]"]
            WidthDepth = ["Width-to-depth ratio"]
            ConvexArea = [f"Convex area [{self.unit}^2]"]
            Solidity = ["Solidity"]
            Perimeter = [f"Root mask perimeter [{self.unit}]"]

            LateralRoots = []

            for i, eachroot in enumerate(self.RootAll):
                Item.append(f"Root {i + 1}")
                bb = eachroot.bbox
                leaf_props = [p for p in eachroot.prop if self._root_type(p) == 1]
                primary_props = [p for p in eachroot.prop if self._root_type(p) == 2]
                lateral_props = [p for p in eachroot.prop if self._root_type(p) == 3]

                leaf_point = None
                if leaf_props:
                    leaf_point = leaf_props[0].pt

                primary_prop = primary_props[0] if primary_props else None
                primary_path = self._orient_primary_path(
                    primary_prop.path, leaf_point) if primary_prop is not None else []

                total_root_length = 0.0
                primary_length = 0.0
                primary_vector = 0.0
                primary_straight = 0.0
                primary_tort = 0.0
                primary_dir = ""
                primary_vertical = ""
                lateral_lengths = []
                insertion_angles = []

                total_leaf_area = 0.0
                leaf_count = 0

                for j, rootp in enumerate(eachroot.prop):
                    rtype = self._root_type(rootp)
                    rid = f"{i + 1}-({j + 1})"

                    if rtype == 1:
                        area = float(rootp.length) * self.resolution * self.resolution
                        mean_area = float(rootp.distance) * self.resolution * self.resolution
                        count = int(round(rootp.length / rootp.distance)) if rootp.distance > 0 else 0
                        total_leaf_area += area
                        leaf_count += count
                        w.writerow([rid, ROOTTYPES[rtype], "", "", "", "", "", "", "", "",
                                    "", "", "", "", "", f"{area:.6f}", count])
                        continue

                    if rtype == 2:
                        oriented = primary_path
                    else:
                        oriented = self._orient_lateral_path(rootp.path, primary_path)

                    geom = self._direction_metrics(oriented)
                    path_px = self._path_length_px(oriented)
                    path_len = path_px * self.resolution
                    euclid = geom["euclidean_px"] * self.resolution if geom else 0.0
                    straight = euclid / path_len if path_len > 0 else 0.0
                    tort = path_len / euclid if euclid > 0 else 0.0
                    insertion = self._lateral_insertion_angle(oriented, primary_path) if rtype == 3 else None

                    total_root_length += path_len
                    if rtype == 2:
                        primary_length = path_len
                        primary_vector = euclid
                        primary_straight = straight
                        primary_tort = tort
                        if geom:
                            primary_dir = geom["direction_deg"]
                            primary_vertical = geom["vertical_signed_deg"]
                    else:
                        lateral_lengths.append(path_len)
                        if insertion is not None:
                            insertion_angles.append(insertion)

                    if geom:
                        sx = geom["start"][1] + bb[0]
                        sy = geom["start"][0] + bb[1]
                        ex = geom["end"][1] + bb[0]
                        ey = geom["end"][0] + bb[1]
                        row = [
                            rid, ROOTTYPES[rtype], f"{path_len:.6f}", f"{euclid:.6f}",
                            f"{straight:.6f}", f"{tort:.6f}",
                            f"{sx:.2f}", f"{sy:.2f}", f"{ex:.2f}", f"{ey:.2f}",
                            f"{geom['dx'] * self.resolution:.6f}",
                            f"{geom['dy'] * self.resolution:.6f}",
                            f"{geom['direction_deg']:.3f}",
                            f"{geom['vertical_signed_deg']:.3f}",
                            "" if insertion is None else f"{insertion:.3f}", "", ""
                        ]
                    else:
                        row = [rid, ROOTTYPES[rtype], f"{path_len:.6f}", f"{euclid:.6f}",
                               f"{straight:.6f}", f"{tort:.6f}"] + [""] * 11
                    w.writerow(row)

                TotalLeafarea.append(total_leaf_area)
                NumOfLeaf.append(leaf_count)
                AverageLeaf.append(total_leaf_area / leaf_count if leaf_count > 0 else 0.0)
                totalLengths.append(total_root_length)
                primarLength.append(primary_length)
                primarVector.append(primary_vector)
                straightness.append(primary_straight)
                tortuosity.append(primary_tort)
                primaryDirection.append(primary_dir)
                primaryVerticalAngle.append(primary_vertical)

                total_lateral_length = sum(lateral_lengths)
                TotalLaterLength.append(total_lateral_length)
                TotalLaterRootNum.append(len(lateral_lengths))
                MeanLaterLength.append(total_lateral_length / len(lateral_lengths) if lateral_lengths else 0.0)
                MaxLaterLength.append(max(lateral_lengths) if lateral_lengths else 0.0)
                MeanInsertionAngle.append(float(np.mean(insertion_angles)) if insertion_angles else "")
                LateralDensity.append(len(lateral_lengths) / primary_length if primary_length > 0 else 0.0)
                LateralRoots.append(lateral_lengths)

                roi_root_mask, _ = self._roi_export_masks(eachroot)
                mm = self._mask_metrics(roi_root_mask)
                NetworkArea.append(mm["network_area"])
                SkeletonLength.append(mm["skeleton_length"])
                RootTips.append(mm["tips"])
                BranchPoints.append(mm["branch_points"])
                BranchFreq.append(mm["branching_frequency"])
                AvgDiameter.append(mm["avg_diameter"])
                MedianDiameter.append(mm["median_diameter"])
                MaxDiameter.append(mm["max_diameter"])
                RootDepth.append(mm["depth"])
                RootWidth.append(mm["width"])
                WidthDepth.append(mm["width_depth_ratio"])
                ConvexArea.append(mm["convex_area"])
                Solidity.append(mm["solidity"])
                Perimeter.append(mm["perimeter"])

            w.writerow([])
            w.writerow(Item)
            for summary_row in [
                TotalLeafarea, NumOfLeaf, AverageLeaf,
                totalLengths, primarLength, primarVector, straightness, tortuosity,
                primaryDirection, primaryVerticalAngle,
                TotalLaterLength, TotalLaterRootNum, MeanLaterLength, MaxLaterLength,
                MeanInsertionAngle, LateralDensity,
                NetworkArea, SkeletonLength, RootTips, BranchPoints, BranchFreq,
                AvgDiameter, MedianDiameter, MaxDiameter,
                RootDepth, RootWidth, WidthDepth, ConvexArea, Solidity, Perimeter,
            ]:
                w.writerow(summary_row)

            w.writerow([f"Lateral root length [{self.unit}]"])
            max_later = max((len(sublist) for sublist in LateralRoots), default=0)
            for idx in range(max_later):
                row = [""]
                row.extend(sublist[idx] if idx < len(sublist) else "" for sublist in LateralRoots)
                w.writerow(row)

            # Whole-image mask metrics are also stored explicitly for reproducibility.
            w.writerow([])
            w.writerow(["Whole image mask metrics", "Value"])
            for name, key, unit in [
                ("Network area", "network_area", f"{self.unit}^2"),
                ("Skeleton length", "skeleton_length", self.unit),
                ("Root tips", "tips", "count"),
                ("Branch points", "branch_points", "count"),
                ("Branching frequency", "branching_frequency", f"count/{self.unit}"),
                ("Average diameter", "avg_diameter", self.unit),
                ("Median diameter", "median_diameter", self.unit),
                ("Maximum diameter", "max_diameter", self.unit),
                ("Depth", "depth", self.unit),
                ("Maximum width", "width", self.unit),
                ("Width-to-depth ratio", "width_depth_ratio", "ratio"),
                ("Convex area", "convex_area", f"{self.unit}^2"),
                ("Solidity", "solidity", "ratio"),
                ("Perimeter", "perimeter", self.unit),
            ]:
                w.writerow([name, full_metrics[key], unit])

            w.writerow([])
            w.writerow(["Processing method", getattr(self, 'processing_method', 'unknown'), "YOLO ROI detection is used for both methods"])
            w.writerow([])
            w.writerow(["Scale calibration", "Value", "Unit/Notes"])
            w.writerow(["Resolution per pixel", float(self.resolution), f"{self.unit}/pixel"])
            w.writerow(["Scale source", getattr(self, 'scale_source', 'default'), ""])
            w.writerow(["Scale confidence", float(getattr(self, 'scale_confidence', 0.0)), "0-1"])

        self.SavePtXml()
        return blend

    def ProcessWithThreads(self, status, indices=None):
        """Process only selected/unprocessed ROI indices in worker threads."""
        self.status_var.set(self._runtime_message(status))

        if indices is None:
            indices = [
                i for i, eachroot in enumerate(self.RootAll)
                if not bool(getattr(eachroot, "bProcessed", False))
            ]
        else:
            indices = [int(i) for i in indices]

        self.active_count = len(indices)
        if self.active_count == 0:
            self.status_var.set(self._loc("No unprocessed ROI to process", "未処理の ROI はありません"))
            return

        img_lock = threading.Lock()
        self.threads = []
        for idx in indices:
            thread = threading.Thread(target=self.ProcessROI, args=(idx,))
            thread.start()
            self.threads.append(thread)

        self.after(100, self.ShowResultImage, img_lock)
        self.__config.set_recent_path(
            os.path.join(os.path.normpath(self.ProjectDir), self.file_name + ".xml")
        )

    def ProcessNOThreads(self, indices=None):
        """Process only selected/unprocessed ROIs and preserve completed ROIs."""
        if indices is None:
            indices = [
                i for i, eachroot in enumerate(self.RootAll)
                if not bool(getattr(eachroot, "bProcessed", False))
            ]
        else:
            indices = [int(i) for i in indices]

        self.active_count = len(indices)
        if self.active_count == 0:
            self.status_var.set(self._loc("No unprocessed ROI to process", "未処理の ROI はありません"))
            return

        img_lock = threading.Lock()
        for idx in indices:
            self.ProcessROI(idx)
            self.after(100, self.ShowResultImage, img_lock)

        self.__config.set_recent_path(
            os.path.join(os.path.normpath(self.ProjectDir), self.file_name + ".xml")
        )

    def __ICML_NoAI_toggle(self, event=None):
        """Process only unprocessed ROIs with the legacy ICML method."""
        if self.curImage is None:
            messagebox.showinfo("ICML-NoAI", self._loc("Open an image first.", "先に画像を開いてください。"))
            return

        if len(self.RootAll) == 0:
            self.__AI_Select(event)
        if len(self.RootAll) == 0:
            self.status_var.set(self._loc("ICML-NoAI: YOLO did not detect any ROI", "ICML-NoAI: YOLO は ROI を検出できませんでした"))
            return

        self.processing_method = "ICML-NoAI"
        pending = self._prepare_unprocessed_rois(1)
        if not pending:
            self.status_var.set(self._loc("ICML-NoAI: all ROIs are already processed", "ICML-NoAI: すべての ROI は処理済みです"))
            return

        preserved = len(self.RootAll) - len(pending)
        self.status_var.set(self._loc(
            f"ICML-NoAI: processing {len(pending)} unprocessed ROI(s); {preserved} processed ROI(s) preserved",
            f"ICML-NoAI: 未処理 ROI {len(pending)} 件を処理中; 処理済み ROI {preserved} 件を保持"
        ))
        self.ProcessNOThreads(pending)

    def __ICML_Improve_toggle(self, event=None):
        """Process only unprocessed ROIs with the ICML-improve method."""
        if self.curImage is None:
            messagebox.showinfo("ICML-improve", self._loc("Open an image first.", "先に画像を開いてください。"))
            return

        if len(self.RootAll) == 0:
            self.__AI_Select(event)
        if len(self.RootAll) == 0:
            self.status_var.set(self._loc("ICML-improve: YOLO did not detect any ROI", "ICML-improve: YOLO は ROI を検出できませんでした"))
            return

        self.processing_method = "ICML-improve"
        pending = self._prepare_unprocessed_rois(5)
        if not pending:
            self.status_var.set(self._loc("ICML-improve: all ROIs are already processed", "ICML-improve: すべての ROI は処理済みです"))
            return

        preserved = len(self.RootAll) - len(pending)
        self.status_var.set(self._loc(
            f"ICML-improve: processing {len(pending)} unprocessed ROI(s); {preserved} processed ROI(s) preserved",
            f"ICML-improve: 未処理 ROI {len(pending)} 件を処理中; 処理済み ROI {preserved} 件を保持"
        ))
        self.ProcessNOThreads(pending)

    def __list_recent(self):
        """ List of the recent images """
        self.__recent_images.delete(0, 'end')  # empty previous list
        l = self.__config.get_recent_list()  # get list of recently opened images
        
        for path in l:  # get list of recent image paths
            self.__recent_images.add_command(label=path, command=lambda x=path: self.__Open_Project(x))

        # Disable recent list menu if it is empty.
        if self.__recent_images.index('end') is None:
            self.__image_menu.entryconfigure(self.__index_recent, state='disabled')
        else:
            self.__image_menu.entryconfigure(self.__index_recent, state='normal')
        

    def __set_image(self, img,path,edit,newImg,label):
        """ Close previous image and set a new one """
        if newImg:# if is read from recent or new image, will reset variables
            self.InitialVariables()
        if img.mode =="I;16":
            tif = np.array(img) /256-1
            self.curImage = tif.astype(np.uint8)
            self.mResultImg  = cv2.cvtColor(self.curImage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            h, w = self.curImage.shape
            self.imgType = 1
        elif img.mode =="RGB":  
            self.curImage = np.array(img, dtype=np.uint8)
            self.mResultImg = self.curImage.copy()
            h, w, _ = self.curImage.shape
            self.imgType = 2
        else:
            self.curImage = np.array(img, dtype=np.uint8)
            self.mResultImg  = cv2.cvtColor(self.curImage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
            h, w = self.curImage.shape
            self.imgType = 1
        if path != "":
            if self.bSetResolution == False:
                # Fixed default calibration measured from the acquisition setup:
                # 10 mm = 245 pixels.
                self.resolution = 10.0 / 245.0
                self.unit = 'mm'
                self.scale_source = 'default-245px-per-cm'
                self.scale_confidence = 1.0
        
        self.__imframe = ImageFrame(placeholder=self.__placeholder,
                                    roi_size=self.__config.get_roi_size(), curImg = Image.fromarray(self.curImage),bSelect = edit,ProjectDir = path, label = label,aroot = self.RootAll)
       
        # Enable 'Close image' submenu of the 'File' menu
        self.__image_menu.entryconfigure(self.__index_close, state='normal')
    def InitialVariables(self):
        if not getattr(self, 'bSetResolution', False):
            self.resolution = 10.0 / 245.0
            self.unit = 'mm'
            self.scale_source = 'default-245px-per-cm'
            self.scale_confidence = 1.0
        self.curImage = None
        self.mResultImg = None
        self.mSelectMode = 1
        #self.resolution =None
        self.curRoot = -1
        self.bNeedRotate = False
        self.rotation_direction = 'none'
        self.RootAll = list() #add EachRootData
        self.processing_method = getattr(self, 'processing_method', 'ICML-improve')
        
        # Deep-learning models are intentionally not recreated here.  They are
        # loaded lazily once and reused by interactive and batch processing.
        self.gui_queue = queue.Queue()
    @handle_exception(0)
    def __Create_Project(self):
        
        self.Create_New_Project(self.master)

    
    def __Open(self):
        project_path = self._askopenfilename_from_history(
            "project_open", parent=self.master, title="Open RAPID project",
            filetypes=[("XML files", "*.xml")],
        )
        if project_path:
            self._remember_recent_folder("project_open", os.path.dirname(project_path))
            self.__Open_Project(project_path)
    
    
    def __Open_Project(self,project_path):
        
        
        if project_path and os.path.exists(project_path):
           self.file_name = os.path.splitext(os.path.basename(project_path))[0]
           self.ProjectDir = os.path.normpath(os.path.dirname(project_path))
           
           #print("Selected file:", file_path)
           tree = ET.parse(project_path)
           xroot = tree.getroot()
           # 通过标签名获取特定元素
           element = xroot.find("colorimage")
           stored_image_path = element.text if element is not None else None
           fallback_image = (re.split(r'[\\/]+', stored_image_path)[-1]
                             if stored_image_path else None)
           self.imgpath = self._resolve_project_file(
               self.ProjectDir, stored_image_path, fallback_image
           )
           if not self.imgpath:
               raise FileNotFoundError(
                   "Project source image was not found. Stored path: %r; project: %s" %
                   (stored_image_path, self.ProjectDir)
               )
           img = Image.open(self.imgpath)
           self.__set_image(img,project_path,1,1,None)

           # Restore physical scale for projects that saved calibration metadata.
           resolution_node = xroot.find("resolution")
           unit_node = xroot.find("unit")
           scale_set_node = xroot.find("scale_set")
           method_node = xroot.find("segmentation_method")
           if method_node is not None and method_node.text:
               self.processing_method = method_node.text

           if resolution_node is not None and resolution_node.text:
               try:
                   restored_resolution = float(resolution_node.text)
                   if restored_resolution > 0:
                       self.resolution = restored_resolution
                       if unit_node is not None and unit_node.text:
                           self.unit = unit_node.text
                       self.bSetResolution = (scale_set_node is None or scale_set_node.text == '1')
                       src = xroot.find("scale_source")
                       conf = xroot.find("scale_confidence")
                       self.scale_source = src.text if src is not None and src.text else 'project'
                       try:
                           self.scale_confidence = float(conf.text) if conf is not None and conf.text else 0.0
                       except ValueError:
                           self.scale_confidence = 0.0
                       self.status_var.set(self._loc(
                           f"Scale restored: {self.resolution:.8g} {self.unit}/pixel ({self.scale_source})",
                           f"スケールを復元しました: {self.resolution:.8g} {self.unit}/pixel ({self.scale_source})"
                       ))
               except ValueError:
                   pass

           if img.mode =="I;16":
               self.imgType = 1
               label = np.zeros_like(self.curImage)
           elif img.mode =="RGB":  
               self.imgType = 2
               label = np.zeros_like(self.curImage[:,:,0])
           w, h = img.size
           #print(w,h)
           bRotate = False
           element = xroot.find("bRotate")
           direction_node = xroot.find("rotation_direction")
           saved_direction = (direction_node.text.strip().lower()
                              if direction_node is not None and direction_node.text else 'cw')
           if saved_direction not in ('cw', 'ccw'):
               saved_direction = 'cw'
           if element is not None and element.text == "1":
               self.__SetRotation(saved_direction)
               bRotate = True
               h,w= img.size
           self.bNeedRotate = bRotate
           self.rotation_direction = saved_direction if bRotate else 'none'
           broi = xroot.findall('Roi')
           if broi:
               full_sem = self._read_cv_unicode(
                   os.path.join(self.ProjectDir, "mask_labels.png"), cv2.IMREAD_UNCHANGED)
               for roi_index in broi:
                    item = roi_index.find("item")
                    ltx = int(item.find('ltx').text)
                    lty = int(item.find('lty').text)
                    rbx = int(item.find('rbx').text)
                    rby = int(item.find('rby').text)
                    eachroot = EachRootData(ROIs([(ltx, lty), (rbx, rby)], True))
                    restored_mode = 1 if getattr(self, 'processing_method', '') == 'ICML-NoAI' else 5
                    eachroot.Generatebb(cv2.cvtColor(self.curImage, cv2.COLOR_BGR2RGB), restored_mode)
                    roi_shape = eachroot.mImgCV.shape[:2]

                    semantic = None
                    sem_node = roi_index.find('semanticmask')
                    if sem_node is not None and sem_node.text:
                        sem_path = self._resolve_project_file(self.ProjectDir, sem_node.text)
                        semantic = self._read_cv_unicode(sem_path, cv2.IMREAD_UNCHANGED)
                    if semantic is None and full_sem is not None:
                        semantic = full_sem[lty:rby, ltx:rbx]
                    if semantic is not None and semantic.shape != roi_shape:
                        semantic = cv2.resize(semantic, (roi_shape[1], roi_shape[0]), interpolation=cv2.INTER_NEAREST)

                    overlap_restore = self._load_saved_overlap_props(
                        roi_index, roi_shape, eachroot.mImgCV
                    )
                    if overlap_restore is not None:
                        props, rlabel, semantic_projection, binary_root = overlap_restore
                        eachroot.prop = props
                        eachroot.roilabel = rlabel
                        # Keep the projection only for display/editor compatibility.
                        eachroot.semantic_mask = semantic_projection.copy()

                        bin_node = roi_index.find("binarymask")
                        saved_binary = None
                        if bin_node is not None and bin_node.text:
                            bin_path = self._resolve_project_file(self.ProjectDir, bin_node.text)
                            saved_binary = self._read_cv_unicode(bin_path, cv2.IMREAD_GRAYSCALE)
                        if saved_binary is not None:
                            if saved_binary.shape != roi_shape:
                                saved_binary = cv2.resize(
                                    saved_binary,
                                    (roi_shape[1], roi_shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            eachroot.segmask = (saved_binary > 0).astype(np.uint8)
                        else:
                            eachroot.segmask = binary_root.astype(np.uint8)
                    elif semantic is not None:
                        if semantic.ndim == 3:
                            semantic = semantic[:, :, 0]
                        semantic = np.asarray(semantic, dtype=np.uint8)
                        eachroot.semantic_mask = semantic.copy()
                        eachroot.segmask = (semantic >= 2).astype(np.uint8)
                        props, rlabel = self._semantic_to_props(semantic, eachroot.mImgCV)
                        eachroot.prop = props
                        eachroot.roilabel = rlabel
                    else:
                        # Backward-compatible fallback for old XML files that only saved paths.
                        p = ar(None, 1, 1, 0)
                        rlabel = np.zeros(roi_shape, dtype=np.uint8)
                        props = []
                        sem = np.zeros(roi_shape, dtype=np.uint8)
                        next_lateral = 3
                        for pts in roi_index.findall("Pts"):
                            mask = np.zeros(roi_shape, dtype=np.uint8)
                            value = int(pts.get("value", len(props) + 1))
                            coordinates = [(int(c.get("x")), int(c.get("y"))) for c in pts.findall("coordinate")]
                            for pt in coordinates:
                                cv2.circle(mask, (pt[1], pt[0]), 2, 1, -1)
                            type_attr = pts.get("type")
                            if type_attr is not None:
                                rtype = int(type_attr)
                            elif self.imgType == 2 and value == 1:
                                rtype = 1
                            elif (self.imgType == 2 and value == 2) or (self.imgType == 1 and value == 1):
                                rtype = 2
                            else:
                                rtype = 3
                            if np.any(mask):
                                try:
                                    prop = p.ProcessLoad(mask, self.imgType, rtype)
                                    prop.type = rtype
                                except Exception:
                                    coords = list(map(tuple, np.argwhere(mask > 0)))
                                    st = coords[0] if coords else (0, 0)
                                    prop = RootProp(0.0, 0.0, coords, rtype, st, mask, 0)
                                props.append(prop)
                                iid = len(props)
                                rlabel[mask > 0] = iid
                                if rtype >= 3:
                                    prop.label_value = next_lateral
                                    sem[mask > 0] = next_lateral
                                    next_lateral = min(255, next_lateral + 1)
                                else:
                                    prop.label_value = rtype
                                    sem[mask > 0] = rtype
                        eachroot.prop = props
                        eachroot.roilabel = rlabel
                        eachroot.semantic_mask = sem
                        eachroot.segmask = (sem >= 2).astype(np.uint8)

                    eachroot.bProcessed = True
                    self.RootAll.append(eachroot)

               # Rebuild the visible result from the restored branch instances.
               self.mResultImg = self.curImage.copy()
               for eachroot in self.RootAll:
                    rlabel = np.asarray(eachroot.roilabel)
                    bb = eachroot.bbox
                    if rlabel.size and np.any(rlabel):
                        roi_img = eachroot.mImgCV.copy()
                        for one in range(1, len(eachroot.prop) + 1):
                            rootp = eachroot.prop[one - 1]
                            color = self._root_display_color(eachroot, one - 1)
                            draw_mask = self._display_instance_mask(eachroot, one - 1, rlabel)
                            self._paint_main_label(roi_img, draw_mask, color)
                        self.mResultImg[bb[1]:bb[3], bb[0]:bb[2]] = roi_img.copy()
               if self.__imframe is not None:
                    self.__imframe.UpdateImg(Image.fromarray(self.mResultImg))
               self.status_var.set(self._loc("Loaded %d roots" % len(self.RootAll), "%d 本の根を読み込みました" % len(self.RootAll)))
    
    def __save_Result(self):
        self.SaveResult()
    def __close_image(self):
        """ Close image """
        if self.__imframe:
            self.__imframe.destroy()
            self.__imframe = None
            self.master.title(self.__default_title)  # set default window title
            # Disable 'Close image' submenu of the 'File' menu
            self.__image_menu.entryconfigure(self.__index_close, state='disabled')
    def destroys(self):
        self.InitialVariables()
        self.SaveResult()
        """ Destroy the main frame object and release all resources """
        if self.__imframe and self.ProjectDir is not None:  # image is not closed  os.path.normpath(self.entry1.get()),os.path.normpath(self.entry2.get())
            self.__config.set_opened_path(os.path.join(os.path.normpath(self.ProjectDir),self.file_name+".xml"))  # remember opened image
        else:  # image is closed
            self.__config.set_opened_path()  # no path
        self.__close_image()
        self.__config.destroy()
        logging.info('Exit Application')
        #self.quit()
        self.master.destroy()
    def _get_yolo_weights_path(self):
        """Return the custom Ultralytics detection weight used by the GUI.

        ARMP/best.pt is the preferred location for the YOLO11 model.  The
        existing ARMP/viewer/best.pt is kept as a fallback so an older
        Ultralytics (for example YOLOv8) weight can still be tested while the
        YOLO11 model is being retrained.
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base_dir, "best.pt"),
            os.path.join(base_dir, "viewer", "best.pt"),
        ]
        for weights in candidates:
            if os.path.isfile(weights):
                return weights
        raise FileNotFoundError(
            "YOLO weights not found. Put the YOLO11-trained best.pt in ARMP/best.pt."
        )

    def _load_yolo_model(self):
        weights = self._get_yolo_weights_path()
        logging.info("Load Ultralytics YOLO model: %s", weights)
        return YOLO(weights)

    def rotatePt90(self, x1, y1, x2, y2, image_width, image_height):
        """Transform a rectangle from the original image to a 90-deg CW image."""
        new_x1 = image_height - y1 - 1
        new_y1 = x1
        new_x2 = image_height - y2 - 1
        new_y2 = x2
        return new_x1, new_y1, new_x2, new_y2

    def rotatePt90CCW(self, x1, y1, x2, y2, image_width, image_height):
        """Transform a rectangle from the original image to a 90-deg CCW image."""
        new_x1 = y1
        new_y1 = image_width - x1 - 1
        new_x2 = y2
        new_y2 = image_width - x2 - 1
        return new_x1, new_y1, new_x2, new_y2

    @staticmethod
    def _leaf_side_in_roi(img_rgb, x1, y1, x2, y2):
        """Estimate whether the leaf mass lies on the left or right of a horizontal ROI.

        This is only an orientation cue used before organ segmentation.  A broad HSV
        green candidate mask is cleaned with small morphology, and the median x
        position of sufficiently large green components is compared with the ROI
        centre.  ``None`` is returned when the cue is too weak or ambiguous.
        """
        if img_rgb is None:
            return None
        arr = np.asarray(img_rgb)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None

        h, w = arr.shape[:2]
        xa = max(0, min(w, int(np.floor(min(x1, x2)))))
        xb = max(0, min(w, int(np.ceil(max(x1, x2)))))
        ya = max(0, min(h, int(np.floor(min(y1, y2)))))
        yb = max(0, min(h, int(np.ceil(max(y1, y2)))))
        if xb - xa < 8 or yb - ya < 8:
            return None

        crop = arr[ya:yb, xa:xb, :3].astype(np.uint8, copy=False)
        try:
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        except cv2.error:
            return None

        # Keep the same hue range as ICML-improve but use a slightly stronger
        # saturation threshold here so pale background is less likely to vote.
        green = cv2.inRange(
            hsv,
            np.array([25, 35, 20], dtype=np.uint8),
            np.array([90, 255, 255], dtype=np.uint8),
        )
        kernel = np.ones((3, 3), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
        green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (green > 0).astype(np.uint8), connectivity=8)
        if n <= 1:
            return None

        min_area = max(20, int(round(crop.shape[0] * crop.shape[1] * 0.001)))
        kept = np.zeros(green.shape, dtype=np.uint8)
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
                kept[labels == i] = 1
        ys, xs = np.nonzero(kept)
        if xs.size < min_area:
            return None

        leaf_x = float(np.median(xs))
        center_x = (crop.shape[1] - 1) / 2.0
        # A small dead band avoids making a 90-deg direction decision from a
        # nearly centred/ambiguous green region.
        dead_band = max(2.0, 0.05 * crop.shape[1])
        if abs(leaf_x - center_x) <= dead_band:
            return None
        return 'left' if leaf_x < center_x else 'right'

    def detectYOLO(self, model, imgCV, allow_gui_rotation=True):
        """Detect root ROIs and orient horizontal plants with leaves at the top.

        YOLO first decides whether the acquisition image is predominantly
        horizontal, as before.  If a 90-deg rotation is required, green leaf cues
        inside the horizontal YOLO ROIs vote for the direction: leaves on the left
        imply clockwise rotation, while leaves on the right imply counter-clockwise
        rotation.  If no reliable leaf cue is available, clockwise rotation is kept
        as a backward-compatible fallback.
        """
        results = model.predict(
            source=imgCV,
            imgsz=1024,
            conf=0.15,
            iou=0.45,
            max_det=100,
            device=self._active_yolo_device(),
            verbose=False,
        )

        HisCoord = []
        HisCoordCW = []
        HisCoordCCW = []
        rotate_votes = 0
        leaf_left_votes = 0
        leaf_right_votes = 0
        h, w = imgCV.shape[:2]

        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                for x1, y1, x2, y2 in reversed(xyxy_boxes):
                    HisCoord.append(
                        ROIs([(int(x1), int(y1)), (int(x2), int(y2))], False)
                    )

                    cwx1, cwy1, cwx2, cwy2 = self.rotatePt90(
                        x1, y1, x2, y2, w, h)
                    HisCoordCW.append(
                        ROIs([(int(cwx1), int(cwy1)), (int(cwx2), int(cwy2))], False)
                    )

                    ccwx1, ccwy1, ccwx2, ccwy2 = self.rotatePt90CCW(
                        x1, y1, x2, y2, w, h)
                    HisCoordCCW.append(
                        ROIs([(int(ccwx1), int(ccwy1)), (int(ccwx2), int(ccwy2))], False)
                    )

                    width = x2 - x1
                    height = y2 - y1
                    if height > 0 and (width / height) > 0.9:
                        rotate_votes += 1
                        side = self._leaf_side_in_roi(imgCV, x1, y1, x2, y2)
                        if side == 'left':
                            leaf_left_votes += 1
                        elif side == 'right':
                            leaf_right_votes += 1

        self.bNeedRotate = bool(HisCoord) and rotate_votes > (len(HisCoord) / 2)
        if self.bNeedRotate:
            # For a horizontal plant: left-side leaves move to the top under CW;
            # right-side leaves move to the top under CCW.
            direction = 'ccw' if leaf_right_votes > leaf_left_votes else 'cw'
            self.rotation_direction = direction
            rotate_code = (cv2.ROTATE_90_COUNTERCLOCKWISE
                           if direction == 'ccw' else cv2.ROTATE_90_CLOCKWISE)

            if allow_gui_rotation:
                self.__SetRotation(direction)
            else:
                self.curImage = cv2.rotate(self.curImage, rotate_code)
                if np.asarray(self.curImage).ndim == 2:
                    self.mResultImg = cv2.cvtColor(
                        self.curImage.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    self.mResultImg = self.curImage.copy()

            return HisCoordCCW if direction == 'ccw' else HisCoordCW

        self.rotation_direction = 'none'
        return HisCoord
    '''
    era_starts = {
    '令和': datetime(2019, 5, 1),
    '平成': datetime(1989, 1, 8),
    '昭和': datetime(1926, 12, 25),
    # 根据需要可以继续添加更多年号
}

# 获取当前日期
current_date = datetime.now()

# 转换为日本纪年方式
def to_japanese_era(date):
    for era, start_date in reversed(era_starts.items()):
        if date >= start_date:
            year_of_era = date.year - start_date.year + 1
            # 日本在年号的第一年使用“元年”表示
            year_of_era_str = "元年" if year_of_era == 1 else f"{year_of_era}年"
            return f"{era}{year_of_era_str}{date.month}月{date.day}日"

# 显示结果
japanese_era_date = to_japanese_era(current_date)
    '''
  
    def GenerateReport(self):
        from fpdf import FPDF
        import os
        from datetime import datetime
        import random

        # 创建PDF文档
        pdf = FPDF()
        pdf.add_page()
        
        font_path="ARMP/MSGOTHIC.TTF"
        pdf.add_font('MSGOTHIC', fname=font_path, uni=True)
        # 设置字体
        pdf.set_font("Arial", size=12)
        
        # 添加标题
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, "Report on the measurement of root", 0, 1, 'C')
        
        # 添加基础信息
        # pdf.set_font("Arial", size=12)
        pdf.set_font('MSGOTHIC','',12)
        pdf.cell(200, 10, f"filename:{self.file_name}", 0, 1)
        pdf.cell(200, 10, f"Operator:{os.getlogin()}", 0, 1)
        pdf.cell(200, 10, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
        #pdf.cell(200, 10, "説明: この画像は、スキャナー (DS-20000、エプソン、日本) によってキャプチャされた画像です。栽培の条件については以下の通りです：省略する", 0, 1)
        # 添加图像（这里使用预设的图像路径，实际使用时需要调整）
        pdf.set_font('MSGOTHIC','',12)
        text = "説明: この画像は、スキャナー (DS-20000、エプソン、日本) によってキャプチャされた画像です。栽培の条件については以下の通りです：省略する"
        pdf.multi_cell(0, 10, text,border=0, align='l', fill=False)
        pdf.image(os.path.join(self.file_name,"result.jpg"), x=10, y=150, w=180)
        # 添加表格
        data1 = [['Item', 'Root type', 'Path', "Distance"]]
        for i,eachroot in enumerate(self.RootAll):

            for j, rootp in enumerate(eachroot.prop):
                t = rootp.type
                if  t>2:
                    t=3
                data1.append([f"{i+1}-({j+1})",f"{ROOTTYPES[t]}","{:.2f}".format(rootp.distance*self.resolution),"{:.2f}".format(rootp.length*self.resolution )])
        
        pdf.set_font("Arial", size=12)
        
        line_height = pdf.font_size * 2.5
        col_width = 30#pdf.w / len(data1[0])  # 均分列宽
        for row_index, row in enumerate(data1):
            if row[1] == "Leaf" and self.imgType==2:  # 指定为第三行设置背景色
                pdf.set_fill_color(220, 220, 220)  # 设置淡灰色背景
                fill = True
            elif row[1] == "Primary root" and self.imgType==1:  # 指定为第三行设置背景色:
                pdf.set_fill_color(220, 220, 220)  # 设置淡灰色背景
                fill = True
            else:
                fill = False
    
            for i, item in enumerate(row):
                pdf.cell(col_width, line_height, item, border=1, ln=0, align='C', fill=fill)
            pdf.ln(line_height)
        pdf.set_font('MSGOTHIC','',12)
        pdf.cell(200,10,"図１　測定結果付き根部画像",0,1)
        
        
        pdf_path = os.path.join(self.file_name,"report.pdf")
        
        pdf.output(pdf_path)
        # print("Saved to pdf")  # debug