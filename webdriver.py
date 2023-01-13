from selenium import webdriver

class InstaBot:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get("https://r.mtdv.me/g1Kro2O7hZ")

InstaBot()

