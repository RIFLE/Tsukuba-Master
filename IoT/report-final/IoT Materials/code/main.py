from kivymd.app import MDApp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivymd.toast import toast
from kivy.lang.builder import Builder
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.graphics import Color, Ellipse
from kivy.clock import Clock

import socket
import time


layout = '''
MDBoxLayout:
    orientation: "vertical"
    Button:
        text: "Change cursor color"
        color: "white" 
        pos_hint: {'center_x':0.5}
        font_size: "12sp"
        size: (self.parent.width/2,self.parent.height/4)
        size_hint: None, None
        on_release: app.change_cursor()

    Button:
        text: "Change background color"
        color: "white" 
        pos_hint: {'center_x':0.5,'center_y':0.875}
        font_size: "12sp"
        size: (self.parent.width/2,self.parent.height/4)
        size_hint: None, None
        on_release: app.change_background()

    Button:
        text: "Button"
        color: "white" 
        pos_hint: {'center_x':0.5,'center_y':0.875}
        font_size: "12sp"
        size: (self.parent.width/2,self.parent.height/4)
        size_hint: None, None
        on_release: app.toast()

    MDRectangleFlatButton:
        text: "Calibrate"
        pos_hint: {'center_x':0.5,'center_y':0.875}
        font_size: "30sp"
        on_release: app.request_calibration()

    MDRectangleFlatButton:
        text: "Mouse mode"
        pos_hint: {'center_x':0.5,'center_y':0.875}
        font_size: "20sp"
        on_release: app.change_mode(self)

    MDFillRoundFlatIconButton:
        icon: "wifi"
        text: "Connect"
        pos_hint: {'center_x':0.5,'center_y':0.875}
        user_font_size: "40sp"
        on_release: app.connect()
'''


class TouchScreenApp(MDApp):

    def build(self):
        self.mode = 0 # 0/1 for mouse/screen
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Green"

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

        self.cursor = None
        self.focused_button = None
        self.hold_N = 0
        self.N_for_action = 5
        self.calibrating = False

        self.colors = [(1,1,1), (1,0,0), (0,1,0), (0,0,1), (0,0,0)]
        self.cursor_color = (1,1,1)
        self.bg_color = (0,0,0)

        Clock.schedule_interval(self.handler,.2)

        return Builder.load_string(layout)

    def handler(self,dt):
        if not self.connected:
            return
        data = self.client_socket.recv(102).decode()
        print(data)
        if data == "-1": # Out of bounds data
            if self.cursor:
                self.root.canvas.remove(self.cursor)  
                self.cursor = None
        elif data == "0": # Calibration starting
            if self.mode == 0:
                toast("Hold Top-Left position")
            elif self.mode == 1:
                toast("Hold Bottom-Left position")
        elif data == "1": # Upper-left point calibration
            if self.mode == 0:
                toast("Hold Bottom-Right position")
            elif self.mode == 1:
                toast("Hold Top-Right position")
        elif data == "2": # Bottom-right point calibration
            toast("Calibration complete!")
            self.calibrating = False
        else: # Data is a position
            try:
                position = [int(i) for i in data.split()]
                #print(position)
                self.draw_cursor(position)
                if not self.calibrating:
                    self.simulate_button_press(position)
            except:
                pass

    def toast(self):
        toast("Button clicked")
    
    def change_cursor(self):
        idx = self.colors.index(self.cursor_color)
        idx = (idx+1)%len(self.colors)
        self.cursor_color = self.colors[idx]

    def change_background(self):
        idx = self.colors.index(self.bg_color)
        idx = (idx+1)%len(self.colors)
        self.bg_color = self.colors[idx]
        self.root.md_bg_color = self.bg_color

    def change_mode(self, btn):
        if self.mode == 0:
            self.mode = 1
            self.N_for_action = 1
            btn.text = "Screen mode"
        elif self.mode == 1:
            self.mode = 0
            self.N_for_action = 5
            btn.text = "Mouse mode"

    def connect(self):
        if self.connected:
            return
        try:
            self.client_socket.connect(("raspberrypi.local", 123))
            self.connected = True
            toast("Connected!")
        except:
            toast("Failed to connect")
        
    def draw_cursor(self, position):
        if self.cursor:
            self.root.canvas.remove(self.cursor)  

        with self.root.canvas:
            x = position[0]
            if self.mode:
                y = position[1]
            else:    
                y = Window.size[1] - position[1] # Not pretty, but needed since the way Kivy coordinate system is like this...
            Color(*self.cursor_color)
            d = 30.
            self.cursor = Ellipse(pos=(x - d / 2, y - d / 2), size=(d, d), id="cursor")

    def simulate_button_press(self, position):
        def action(btn):
            self.hold_N = 0 # Reseting
            self.focused_button = None

            btn_name = btn.text # Performing desired action
            if (btn_name =="Button"):
                self.toast()
            elif (btn_name =="Change background color"):
                self.change_background()
            elif (btn_name =="Change cursor color"):
                self.change_cursor()

        x = position[0]
        if self.mode:
            y = position[1]
        else:    
            y = Window.size[1] - position[1] # Not pretty, but needed since the way Kivy coordinate system is like this...
        for child in self.root.children: # Iterate over all children of the root widget
            if hasattr(child, 'collide_point') and child.collide_point(x, y):
                if self.focused_button == child: # Focus on selected button. Action is taken after user holds this position
                    if self.hold_N == self.N_for_action:
                        action(child)
                    else:
                        self.hold_N += 1
                else:
                    self.focused_button = child
                    self.hold_N = 1

    def get_position(self):
        if not self.connected:
            return
        message = "1  -1"
        self.client_socket.sendall(message.encode())
        data = self.client_socket.recv(1024) 
        position = data.decode()
        try:
            x, y = position.split()
            return int(x), int(y)
        except:
            return
            

    def request_calibration(self):
        if not self.connected:
            toast("Not connected!")
            return
        self.calibrating = True
        W,H = Window.size
        message = f"0 {W} {H}"
        self.client_socket.sendall(message.encode()) # Sends a message to the server requeting a calibration

TouchScreenApp().run()
