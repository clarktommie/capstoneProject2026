from manim import *

class Test(Scene):
    def construct(self):
        text = Text("Manim Works")
        self.play(Write(text))
        self.wait()