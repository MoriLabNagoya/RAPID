# -*- coding: utf-8 -*-
import os
import tkinter as tk

from ARMP.logic_logger import LogInitialize, logging
from ARMP.gui_main import MainGUI
#Previous method is tool->Process 
#import asyncio
if __name__ == '__main__':
    LogInitialize()
    logging.info('RAPID start')
    cur_dir = os.path.dirname(os.path.realpath(__file__))  #get current dir
    os.chdir(cur_dir)  
    root = tk.Tk()
    
    root.iconbitmap(os.path.join(cur_dir, r"ARMP\root.ico"))
    app = MainGUI(root)
    root.mainloop()
    logging.info('RAPID exit')