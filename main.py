import cv2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
import time
from time import sleep


class RickBot:
    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get("https://r.mtdv.me/NZQoDVKnp2")
        sleep(1)
        self.driver.find_element(By.XPATH, "/html/body/main/div[2]/div/div/div[3]/button[2]")\
            .click()

class InstaBot:
    def __init__(self, username, pw):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option("detach", True)
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get("https://instagram.com")
        sleep(2)
        self.driver.find_element(By.XPATH, "//input[@name=\"username\"]")\
            .send_keys(username)
        self.driver.find_element(By.XPATH, "//input[@name=\"password\"]")\
            .send_keys(pw)
        self.driver.find_element(By.XPATH, "//button[@type=\"submit\"]")\
            .click()
        sleep(4)
        self.actions = ActionChains(self.driver)
        self.actions.move_by_offset(24, 452)
        self.actions.click()
        self.actions.perform()
        sleep(2)
        self.driver.find_element(By.XPATH, "//input[@type=\"file\"]")\
            .send_keys("C:\\Users\\super\\Desktop\\WakeUpCharm\\A7401694.jpg")
        sleep(1)
        self.driver.find_element(By.XPATH, "/html/body/div[2]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/div/div/div[1]/div/div/div[3]/div/button")\
            .click()
        sleep(1)
        self.driver.find_element(By.XPATH, "/html/body/div[2]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/div/div/div[1]/div/div/div[3]/div/button")\
            .click()
        """  self.driver.find_element(By.XPATH, "/html/body/div[2]/div/div/div/div[2]/div/div/div[1]/div/div[3]/div/div/div/div/div[2]/div/div/div/div[2]/div[2]/div/div/div/div[2]/div[1]/div[1]")\
            .send_keys("lol") """

        sleep(5)
        self.driver.quit()




def main():
    sleep(10)
    rick_bot_initial = RickBot()
    second = time.time()
    original_time = int(second)
    objectDetected = False
    releaseHell = False


    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)

    classNames= []
    classFile = "coco.names"
    with open(classFile,"rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/ 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success,img = cap.read()
        classIds, confs, bbox = net.detect(img,confThreshold=.50)
        list_of_ids = classIds.tolist()
        print(list_of_ids)
        if 90 in list_of_ids:
            objectDetected = True
            rick_bot_initial.driver.quit()
            break
        if (int(time.time()) - original_time) > 10:
            releaseHell = True
            rick_bot_initial.driver.quit()
            break



        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        cv2.imshow("Output", img)
        cv2.waitKey(1)

    if releaseHell:
         InstaBot("monkeymediafilms", "m0nkeyeatsbanana")

if __name__ == "__main__":
    main()








