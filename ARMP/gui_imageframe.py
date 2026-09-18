# -*- coding: utf-8 -*-
import tkinter as tk

from tkinter import ttk
from PIL import Image, ImageTk
from ARMP.logic_logger import logging
from ARMP.gui_autoscrollbar import AutoScrollbar
import numpy as np
#1-20 is root, 255 is rectangle
import cv2
import keyboard
#from ARMP.gui_main import COLORS
COLORS = [(255,0,0), (0,255,0),(0,0,255), (255,255,0),(0,255,255), (255,0,255),(128,0,0),(255,0,255)]
class EachRootData:
    #def __init__(self,ROI,masks,values):
    def __init__(self,ROI):
        self.prop = list()
        self.ROICoordinate = ROI.ROICoord
        self.mResultImg = None
        self.mImgCV = None
        self.roilabel = None
        self.bProcessed = ROI.bProcessed
        self.prsMode = 1
        #self.topy,self.topx,self.roih,self.roiw = 0,0,0,0
        self.bbox = np.array([0,0,0,0,0,0])#topx,topy,bottomx,bottomy,roiw,roih
        if len(ROI.ROICoord) ==2:
           self.sMode = 0# for rectangle
        else:
           self.sMode = 1 #for polygon
           
    def Generatebb(self,imgCV,prsMode):
        roi = self.ROICoordinate
        if len(roi) == 2:
            (topy, topx) = (np.min([int(roi[0][1]),int(roi[1][1])]), np.min([int(roi[0][0]),int(roi[1][0])]))
            (bottomy, bottomx) = (np.max([int(roi[0][1]),int(roi[1][1])]), np.max([int(roi[0][0]),int(roi[1][0])]))
        else:
            ctr = np.array(roi).astype(np.int32)
            (topy, topx) = np.min(ctr, axis=0)
            (bottomy, bottomx) = np.max(ctr, axis=0)
        roih,roiw = bottomy-topy,bottomx-topx
        self.bbox = np.array([topx,topy,bottomx,bottomy,roiw,roih])
        self.mImgCV = imgCV[topy:bottomy, topx:bottomx].copy()
        self.prsMode = prsMode
    
class ROIs():
    def __init__(self,coord,bPred):#stype: 1 is hand, 2 is AI
        self.ROICoord =coord
        self.bProcessed = bPred
class ImageFrame():
    """ Display an image and necessary functional for rectangle, zoom, shift, etc. """
    
    #1 is edit,0 is result
    def __init__(self, placeholder, roi_size,curImg,bSelect,ProjectDir,label,aroot):#PIL image
        """ Initialize the ImageFrame """
        #self.__isEdit = Edit
        
        self.__mode = bSelect
        self.brushType = -1
        self.ProjectFolder = ProjectDir
        self.__roi_size = roi_size  # obtain size of the roi
        #self.__roi_tag = 'roi'
        self.__text_size = 14  # size of the text
        self.__text_tag = 'text'
        self.__font = 'Helvetica {size} normal'
        self.__font_max_size = 20  # max font size
        self.__font_min_size = 7  # min font size
       
        self.__delta = 1.3  # zoom magnitude
        self.__previous_state = 0  # previous state of the keyboard
        # Create ImageFrame in placeholder widget and make it expandable
        self.__imframe = ttk.Frame(placeholder)  # placeholder of the ImageFrame object
        self.__imframe.grid(row=0, column=0, sticky='nswe')
        self.__imframe.rowconfigure(0, weight=1)  # make grid cell expandable
        self.__imframe.columnconfigure(0, weight=1)
        # Vertical and horizontal scrollbars for canvas
        hbar = AutoScrollbar(self.__imframe, orient='horizontal')
        vbar = AutoScrollbar(self.__imframe, orient='vertical')
        hbar.grid(row=1, column=0, sticky='we')
        vbar.grid(row=0, column=1, sticky='ns')
        # Create canvas and bind it with scrollbars
        self.canvas = tk.Canvas(self.__imframe, highlightthickness=0,
                                  xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.configure(height = curImg.size[1])
        self.canvas.update()  # wait till canvas is created
        hbar.configure(command=self.__scroll_x)  # bind scrollbars to the canvas
        vbar.configure(command=self.__scroll_y)
        #logging.info('Open image: {}'.format(self.path))
        self.__state = 'hidden'
        #self.__image = Image.open(self.path)  # open image
        self.OrigIM = np.array(curImg)
        self.__image = curImg
        
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle((0, 0, self.__image.size), width=0)
        
        
        # Set region of interest (roi) rectangle on the canvas and make it invisible
        #self.__roi_rect = self.canvas.create_rectangle((0, 0, 0, 0), width=2, outline='red',
        #                                                 state='hidden')
        # Set text-warning on the canvas and make it invisible
        self.__text_warning = self.canvas.create_text(
            (0, 0), anchor='sw', fill='red', text='Too close to the edge', state='hidden',
            font=self.__font.format(size=self.__text_size), tag=self.__text_tag)
        #
        self.__w, self.__h = self.__image.size  # image width and height
        self.__min_side = min(self.__w, self.__h)  # get the smaller image side
        # Pillow >= 10 removed Image.ANTIALIAS; LANCZOS is its direct replacement.
        # getattr keeps this compatible with older Pillow versions as well.
        self.__filter = getattr(Image, 'Resampling', Image).LANCZOS
        self.ROICoord = list()
        
        self.RootAll = aroot
        self.DrawList = list()
        self.scale = 1
        self.__SelectMode = 0# select is rectangle or polygon
        #self.ROIImages = list()
        self.selectedRootPoint = list()
        self.drawRootPoint = list()
        
        self.curROI = 0
        
        self.maskIm = np.zeros([curImg.size[1],curImg.size[0]],dtype=np.uint8)
        if label is not None:
           #self.maskIm[:,:,2] = label*255#cv2.merge([label,label,label])
           self.maskIm = label#.astype(np.uint8)#cv2.merge([label,label,label])
        self.__max_side = max(self.__w, self.__h)  # get the smaller image side
        w,h  = self.canvas.winfo_width(),self.canvas.winfo_height()
        if self.__w > self.__h:
            self.imscale =  w / self.__w  # scale for the canvas image zoom
        else:
            self.imscale =  h / self.__h  # scale for the canvas image zoom
        self.canvas.scale('all', 0, 0, self.imscale,self.imscale)  # rescale all objects # will not display all

        # Bind canvas events only after every attribute used by the callbacks
        # (especially self.imscale) has been initialized.  Binding <Motion>
        # earlier allowed the mouse callback to run while __init__ was still
        # executing, which caused: AttributeError: ImageFrame has no imscale.
        self.canvas.bind('<Configure>', self.show_image)
        self.canvas.bind('<ButtonPress-1>', self.__move_from)
        self.canvas.bind('<ButtonPress-3>', self.__Clear)
        self.canvas.bind('<Motion>', self.__motion)
        self.canvas.bind('<MouseWheel>', self.__wheel)
        self.canvas.bind('<Button-5>', self.__wheel)
        self.canvas.bind('<Button-4>', self.__wheel)
        self.canvas.bind('<Leave>', self.__rid_focus)
        self.canvas.bind('<Enter>', self.__set_focus)
        self.canvas.bind('<Key>', lambda event: self.canvas.after_idle(self.__keystroke, event))

        self.canvas.event_generate('<Enter>')  # set focus on the canvas
        self.show_image()  # show image on the canvas
    def isProcessed(self):
        if len(self.RootAll)>0:
            return self.RootAll[0].bProcessed
        return False
    def UpdateImg(self,curImg):
        self.__image = curImg
        self.show_image()  # show image on the canvas
    def SetbrushType(self,cur,mask):
        if cur>=0:
           self.brushType = cur+1
        else:
            self.brushType = 0
        self.maskIm = mask.copy()
        #print(cur)
    def FindCurrentROI(self,event):
        curROI =-1
        if len(self.RootAll)>0:
             ROICoords = [ROIs(root.ROICoordinate,root.bProcessed) for root in self.RootAll]#ROIs([(int(x1),int(y1)),(int(x2),int(y2))],False)
             for i,one in enumerate(ROICoords):
                 #print(one)
                 if len(one.ROICoord) ==2:
                     lt = (one.ROICoord[0][0],one.ROICoord[0][1])
                     rt = (one.ROICoord[1][0],one.ROICoord[0][1])
                     rb = (one.ROICoord[1][0],one.ROICoord[1][1])
                     lb = (one.ROICoord[0][0],one.ROICoord[1][1])
                     polygon = np.array([lt,rt,rb,lb],np.int64)
                 else:
                     polygon = np.array(one.ROICoord,np.int64)
                 x = self.canvas.canvasx(event.x)
                 y = self.canvas.canvasy(event.y)
                 
                 bbox1 = self.canvas.coords(self.container) 
                 xx = (x-bbox1[0])/self.imscale
                 yy = (y-bbox1[1])/self.imscale
                 #print(xx,yy)
                 if cv2.pointPolygonTest(polygon, (xx,yy), False)>0:
                    curROI = i
                    
                    break
        return curROI
    def __Clear(self,event):

        if len(self.ROICoord)>0:
            self.ROICoord.pop()
        else:
            if len(self.RootAll)>0:
                cur = self.FindCurrentROI(event)
                if cur !=-1:
                   self.ROICoord = self.RootAll[cur].ROICoordinate
                   #self.RootAll.remove(cur)
                   del self.RootAll[cur]
        self.show_image()
        
        
    def __LButtonMove(self,event):
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        bbox1 = self.canvas.coords(self.container) 
        xx = int((x-bbox1[0])/self.imscale)
        yy = int((y-bbox1[1])/self.imscale)
        if self.__mode ==2:#1 is select rectangle or polygon,0 is result，2 is edit
            self.canvas.configure(cursor="cross")
            self.maskIm = cv2.circle(self.maskIm,(xx,yy),1,(255,255,0),-1)
        self.show_image()
    def SetSelectMode(self,mode):
        self.__SelectMode = mode
    def __move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        #self.canvas.scan_mark(event.x, event.y)
        #self.statusText.set("(%d,%d) is selected"%(event.x,event.y))
        
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        #print(self.canvas1.find_closest(x, y))
        bbox1 = self.canvas.coords(self.container) 
        xx = (x-bbox1[0])/self.imscale
        yy = (y-bbox1[1])/self.imscale
        #print(self.__mode)
        if self.__mode == 1 and self.FindCurrentROI(event)==-1:
            self.ROICoord.append((xx,yy))
            #print(x,y,xx,yy)
            if len(self.ROICoord)>1:
                if self.__SelectMode ==1:
                    lt = (min(int(self.ROICoord[0][0]),int(self.ROICoord[1][0])),min(int(self.ROICoord[0][1]),int(self.ROICoord[1][1])))
                    rb =(max(int(self.ROICoord[0][0]),int(self.ROICoord[1][0])),max(int(self.ROICoord[0][1]),int(self.ROICoord[1][1])))
                    tmp =ROIs([lt,rb],False)
                    
                    self.RootAll.append(EachRootData(tmp))
                    self.ROICoord.clear()
                elif self.__SelectMode ==0:
                    if (xx- self.ROICoord[0][0])*(xx- self.ROICoord[0][0]) + (yy- self.ROICoord[0][1])*(yy- self.ROICoord[0][1]) <100:
                       self.ROICoord[-1] = self.ROICoord[0]
                       self.RootAll.append(EachRootData(ROIs(self.ROICoord.copy(),False)))
                       self.ROICoord.clear()
        self.show_image()
    
        
    def __move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.show_image()  # zoom tile and show it on the canvas

    def __motion(self, event):
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        bbox1 = self.canvas.coords(self.container) 
        xx = int((x-bbox1[0])/self.imscale)
        yy = int((y-bbox1[1])/self.imscale)
        if self.__mode ==2:#1 is select rectangle or polygon,0 is result，2 is edit
            if keyboard.is_pressed('ctrl'):
                self.canvas.configure(cursor="cross")
                self.maskIm = cv2.circle(self.maskIm,(xx,yy),2,1,-1)                
            elif keyboard.is_pressed('shift'):
                self.canvas.configure(cursor="dotbox")
                self.maskIm = cv2.circle(self.maskIm,(xx,yy),3,0,-1)
        self.show_image(event)
    def SetHisCoord(self,root):
        self.RootAll = root
        self.show_image()
    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get coordinates of the event on the canvas
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.coords(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]: pass  # Ok! Inside the image
        else: return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            if int(self.__min_side * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.__delta
            scale          /= self.__delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.__delta
            scale          *= self.__delta
        self.canvas.scale('all', 0, 0, scale, scale)  # rescale all objects #self.canvas.scale('all', x, y, scale, scale) , set x,y, to 0,0 
        # Configure font size
        size = min(self.__font_max_size, max(self.__font_min_size,
                                             int(self.__text_size * self.imscale)))
        self.canvas.itemconfigure(self.__text_tag, font=self.__font.format(size=size))
        self.scale =  self.imscale
        self.show_image()  # zoom image and show it on the canvas

    def __set_focus(self, event):
        """ Set focus on the canvas and handle <Alt>+<Tab> switches between windows """
        self.canvas.focus_set()
        #self.__motion(event)

    def __rid_focus(self, event=None):
        """ Hide region of interest and remove focus from the canvas """
        self.canvas.itemconfigure(self.__text_warning, state='hidden')  # hide warning
        #self.canvas.itemconfigure(self.__roi_rect, state='hidden')  # hide roi
        self.__imframe.focus_set()  # remove focus from the canvas by setting it elsewhere

    def __keystroke(self, event):
        """ Scrolling with the keyboard.
            Independent from the language of the keyboard, CapsLock, <Ctrl>+<key>, etc. """
        if event.state - self.__previous_state == 4:  # means that the Control key is pressed
            pass  # do nothing if Control key is pressed
        else:
            self.__previous_state = event.state  # remember the last keystroke state
            # Up, Down, Left, Right keystrokes
            if event.keycode in [68, 39, 102]:  # scroll right, keys 'd' or 'Right'
                self.__scroll_x('scroll',  1, 'unit', event=event)
            elif event.keycode in [65, 37, 100]:  # scroll left, keys 'a' or 'Left'
                self.__scroll_x('scroll', -1, 'unit', event=event)
            elif event.keycode in [87, 38, 104]:  # scroll up, keys 'w' or 'Up'
                self.__scroll_y('scroll', -1, 'unit', event=event)
            elif event.keycode in [83, 40, 98]:  # scroll down, keys 's' or 'Down'
                self.__scroll_y('scroll',  1, 'unit', event=event)

    def __scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.show_image()  # redraw the image
        if kwargs and kwargs['event']:
            self.__motion(kwargs['event'])

    def __scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.show_image()  # redraw the image
        if kwargs and kwargs['event']:
            self.__motion(kwargs['event'])

    def show_image(self, event=None):
        """ Show image on the Canvas. Implements correct image zoom almost like in Google Maps """
        
        for oneD in self.DrawList:
           self.canvas.delete(oneD)
        if self.RootAll is not None:
            for eachroot in self.RootAll:
               if eachroot.bProcessed ==True:
                   drawC = 'green'
               else:
                   drawC = 'red'
               bbox1 = self.canvas.coords(self.container) 
               #oneroi  = eachroot.ROICoordinate
               for oneroi in eachroot.ROICoordinate:
                   x = oneroi[0]*self.imscale+bbox1[0]
                   y = oneroi[1]*self.imscale+bbox1[1]
                   c = self.canvas.create_oval(x-3, y-3, x+3,y +3, width = 1, outline ='black', fill ='blue' )   
                   self.DrawList.append(c)
               if len(eachroot.ROICoordinate)==2:
                   c = self.canvas.create_rectangle(eachroot.ROICoordinate[0][0]*self.imscale+bbox1[0],eachroot.ROICoordinate[0][1]*self.imscale+bbox1[1],eachroot.ROICoordinate[1][0]*self.imscale+bbox1[0],eachroot.ROICoordinate[1][1]*self.imscale+bbox1[1], outline=drawC)
                   self.DrawList.append(c)
                   #print("test")
               else:
                   for i in range(1,len(eachroot.ROICoordinate)):
                       ltx = eachroot.ROICoordinate[i-1][0]*self.imscale+bbox1[0]
                       lty = eachroot.ROICoordinate[i-1][1]*self.imscale+bbox1[1]
                       rbx = eachroot.ROICoordinate[i][0]*self.imscale+bbox1[0]
                       rby = eachroot.ROICoordinate[i][1]*self.imscale+bbox1[1]
                       c = self.canvas.create_line(ltx,lty,rbx,rby,fill=drawC)
                       self.DrawList.append(c)
        for one in self.ROICoord:
            bbox1 = self.canvas.coords(self.container) 
            x = one[0]*self.imscale + bbox1[0]
            y = one[1]*self.imscale + bbox1[1]
            
            c = self.canvas.create_oval(x-3, y-3, x+3,y +3, width = 1, outline ='black', fill ='yellow' )    
            self.DrawList.append(c)
        if 1:#self.__SelectMode ==1:
            for i in range(1,len(self.ROICoord)):
                ltx =self.ROICoord[i-1][0]*self.imscale+bbox1[0]
                lty =self.ROICoord[i-1][1]*self.imscale+bbox1[1]
                rbx =self.ROICoord[i][0]*self.imscale+bbox1[0]
                rby =self.ROICoord[i][1]*self.imscale+bbox1[1]
                c = self.canvas.create_line(ltx,lty,rbx,rby,fill='red')
                self.DrawList.append(c)        
        if 1:#self.__SelectMode ==1:
            for i in range(1,len(self.ROICoord)):
                ltx =int(self.ROICoord[i-1][0]*self.imscale+bbox1[0])
                lty =int(self.ROICoord[i-1][1]*self.imscale+bbox1[1])
                rbx =int(self.ROICoord[i][0]*self.imscale+bbox1[0])
                rby =int(self.ROICoord[i][1]*self.imscale+bbox1[1])
                c = self.canvas.create_line(ltx,lty,rbx,rby,fill='red')
                self.DrawList.append(c)   
        box_image = self.canvas.coords(self.container)  # get image area
        box_canvas = (self.canvas.canvasx(0),  # get visible area of the canvas
                      self.canvas.canvasy(0),
                      self.canvas.canvasx(self.canvas.winfo_width()),
                      self.canvas.canvasy(self.canvas.winfo_height()))
        #print(box_canvas,self.imscale)
        box_img_int = tuple(map(int, box_image))  # convert to integer or it will not work properly
        # Get scroll region box
        box_scroll = [min(box_img_int[0], box_canvas[0]), min(box_img_int[1], box_canvas[1]),
                      max(box_img_int[2], box_canvas[2]), max(box_img_int[3], box_canvas[3])]
        # Horizontal part of the image is in the visible area
        if  box_scroll[0] == box_canvas[0] and box_scroll[2] == box_canvas[2]:
            box_scroll[0]  = box_img_int[0]
            box_scroll[2]  = box_img_int[2]
        # Vertical part of the image is in the visible area
        if  box_scroll[1] == box_canvas[1] and box_scroll[3] == box_canvas[3]:
            box_scroll[1]  = box_img_int[1]
            box_scroll[3]  = box_img_int[3]
        # Convert scroll region to tuple and to integer
        self.canvas.configure(scrollregion=tuple(map(int, box_scroll)))  # set scroll region
        x1 = max(box_canvas[0] - box_image[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(box_canvas[1] - box_image[1], 0)
        x2 = min(box_canvas[2], box_image[2]) - box_image[0]
        y2 = min(box_canvas[3], box_image[3]) - box_image[1]
        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            #self.__image = Image.blend(, Image.fromarray(self.maskIm), 0.1)
            
            maxL = np.max(self.maskIm)
            if maxL >0:
                if len(self.OrigIM.shape)==3:
                   blend = np.copy(self.OrigIM)
                elif len(self.OrigIM.shape)==2:
                   blend = cv2.cvtColor(self.OrigIM, cv2.COLOR_GRAY2RGB)
                if self.brushType>0:
                    #only show current root
                    #blend[(self.maskIm==self.brushType)] = COLORS[self.brushType%len(COLORS)]
                    blend[(self.maskIm==1)] = COLORS[self.brushType%len(COLORS)]
                else:
                    for one in range(1,maxL+1):
                        blend[(self.maskIm==one)] = COLORS[one%len(COLORS)]
                self.__image =Image.fromarray(blend) 
            elif maxL==0 and self.brushType >0:
                self.__image =Image.fromarray(self.OrigIM) 
            image = self.__image.crop((int(x1 / self.imscale), int(y1 / self.imscale),
                                       int(x2 / self.imscale), int(y2 / self.imscale)))
            #
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1)), self.__filter))
            imageid = self.canvas.create_image(max(box_canvas[0], box_img_int[0]),
                                                 max(box_canvas[1], box_img_int[1]),
                                                 anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection
    def ReSetMask(self,mask):
        self.maskIm = mask
        self.show_image()
    def GetCurrentScale(self):
        return self.imscale
    def SetScale(self,scale):
        self.imscale = scale
        self.canvas.scale('all', 0, 0, self.imscale,self.imscale)  # rescale all objects # will not display all
    def destroy(self):
        """ ImageFrame destructor """
        
        logging.info('Close image: {}'.format(self.ProjectFolder))
        #self.   =self.maskIm
        self.__image.close()
        self.canvas.destroy()
        self.__imframe.destroy()
