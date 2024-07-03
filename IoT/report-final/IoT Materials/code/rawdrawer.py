import tkinter

class Scribble:

    def on_pressed(self, event):
        self.sx = event.x
        self.sy = event.y
        self.canvas.create_oval(self.sx, self.sy, event.x, event.y,
                                outline = self.color.get(),
                                width = self.width.get())


    def on_dragged(self, event):
        self.canvas.create_line(self.sx, self.sy, event.x, event.y,
                            fill = self.color.get(),
                            width = self.width.get())
        self.sx = event.x
        self.sy = event.y

    """
    def click1(self,event):
        self.canvas.create_oval(self.sx-20,self.sy-20,self.sx+20,self.sy+20,
                            fill = self.color.get(),
                            width = self.width.get())

    def click2(self,event):
        self.canvas.create_oval(self.sx-40,self.sy-30,self.sx+40,self.sy+30,
                           fill = self.color.get(),
                           width = self.width.get())
        """



    def create_window(self):
        window = tkinter.Tk()
        window.title("お絵かきソフト")
        self.canvas = tkinter.Canvas(window, bg = "black", 
                                 width = 600, height = 400)                             
        self.canvas.pack()
        quit_button = tkinter.Button(window, text = "終了",
                                 command = window.quit)
        quit_button.pack(side = tkinter.RIGHT)

        self.canvas.bind("<ButtonPress-1>", self.on_pressed)
        self.canvas.bind("<B1-Motion>", self.on_dragged)
        



        COLORS = ["red", "white", "blue", "pink", "green","black"]
        self.color = tkinter.StringVar()                    
        self.color.set(COLORS[1])                             
        b = tkinter.OptionMenu(window, self.color, *COLORS) 
        b.pack(side = tkinter.LEFT)


        self.width = tkinter.Scale(window, from_ = 1, to = 10,
                               orient = tkinter.HORIZONTAL) 
        self.width.set(5)                                       
        self.width.pack(side = tkinter.LEFT)

        return window;

    def __init__(self):
        self.window = self.create_window();  

    def run(self):
        self.window.mainloop()
Scribble().run()