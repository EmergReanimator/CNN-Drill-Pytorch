from tkinter import *
import tkinter as tk
from PIL import ImageGrab, ImageOps
from torchvision import transforms
import torch
import matplotlib.image as mpimg


batch_size = 128
mean = 0.1307
std = 0.3081

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def predict_digit(image, model, transforms):
    image = image.resize((28,28))

    image = image.convert('L')

    image = ImageOps.invert(image)

    image = transforms(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    output = model(image)

    probs = torch.nn.functional.softmax(output, dim=1)

    return probs.argmax().item(), probs.max().item()


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0
        self.model = torch.load(f'./models/{device}-model.pth')
        self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(mean,), std=(std,))])

        # Creating elements
        self.canvas = tk.Canvas(self, width=300, height=300, bg = "white", cursor="cross")
        self.label = tk.Label(self, text="Draw..", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting)
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=1,pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_clear.grid(row=1, column=0, pady=2)

        #self.canvas.bind("<Motion>", self.start_pos)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        l = self.winfo_x()
        t = self.winfo_y()
        b = t + self.winfo_height()
        r = l + self.winfo_width()

        b = t + 300
        r = l + 300

        rect=(l+4,t+4,r-4,b-4)
        im = ImageGrab.grab(rect)
        # im.save('/tmp/grab.png')

        digit, acc = predict_digit(im, self.model, self.transforms)
        self.label.configure(text= str(digit)+', '+ str(int(acc*100))+'%')

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')

app = App()
mainloop()
