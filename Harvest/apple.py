class Apple:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.object_type = "Apple"

    def get_info(self):
        return {"x": self.x, "y": self.y, "object_type": self.object_type}
