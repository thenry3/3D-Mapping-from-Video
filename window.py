import sdl2.ext


class Window():
    def __init__(self, height, width):
        self.height, self.width = height, width

        sdl2.ext.init()
        self.win = sdl2.ext.Window("SLAM", size=(height, width))
        self.win.show()

    def render(self, image):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surface = sdl2.ext.pixels3d(self.win.get_surface())
        surface[:, :, 0:3] = image.swapaxes(0, 1)

        self.win.refresh()
