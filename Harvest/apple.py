class Apple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.name = "Apple"

    def get_info(self):
        return {"x": self.x, "y": self.y, "name": self.name}
